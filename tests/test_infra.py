"""
Unit tests for the training/eval/export/inference infrastructure built on
top of the core model (photon2perception.models.*), which is already
covered by tests/test_core.py.

Covers:
- photon2perception.losses.{detection_loss,segmentation_loss}
- photon2perception.models.model_wrapper (PerceptionModel / build_perception_model)
- photon2perception.utils.{checkpoint,config,distributed,logger}
- photon2perception.evaluation.{metrics,efficiency}
- photon2perception.models.heads.postprocess
- photon2perception.inference (PerceptionInferencer, pytorch/torchscript/
  onnxruntime backends -- CPU only, no GPU/TensorRT required)
- tools/export.py (TorchScript + ONNX export + numerical parity), invoked
  as a subprocess to exercise it exactly as a user would from the CLI.

All configs use tiny dims (embed_dim=32-64, depth=1-2, small img_size) so
the full suite runs in a few seconds on CPU, matching the CPU-only dev
workflow documented in CLAUDE.md.
"""

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from photon2perception.evaluation.efficiency import (
    compute_input_bandwidth,
    full_efficiency_report,
    measure_latency,
    measure_memory,
)
from photon2perception.evaluation.metrics import DetectionEvaluator, SegmentationEvaluator
from photon2perception.losses.detection_loss import (
    DetectionLoss,
    box_iou,
    decode_boxes,
    encode_boxes,
    generate_anchors,
)
from photon2perception.losses.segmentation_loss import RMILoss, SegmentationLoss
from photon2perception.models.heads.postprocess import postprocess_detections
from photon2perception.models.model_wrapper import PerceptionModel, build_perception_model
from photon2perception.utils.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint,
    load_weights_only,
    save_checkpoint,
    strip_ddp_prefix,
)
from photon2perception.utils.config import apply_cli_overrides, load_config, save_config


# ----------------------------------------------------------------------------
# Shared tiny configs
# ----------------------------------------------------------------------------

def _tiny_det_config() -> dict:
    return {
        'task': 'detection',
        'model': {
            'img_size': [64, 64],
            'patch_size': 16,
            # embed_dim=64 (not 32): SimpleFeaturePyramidNeck's scale_factor=4
            # branch feeds `embed_dim // 2` channels into a GroupNorm(num_groups=32),
            # which requires `embed_dim // 2` to be divisible by (and >=) 32.
            'embed_dim': 64,
            'depth': 2,
            'num_heads': 2,
            'mlp_ratio': 2.0,
            'cfa_pattern': 'rggb',
            'use_rope_2d': True,
            'use_directional': False,
            'use_sparse_routing': False,
            'neck_out_channels': 32,
            'feat_channels': 32,
            'num_anchors': 9,
        },
        'data': {'num_classes': 3},
    }


def _tiny_seg_config() -> dict:
    return {
        'task': 'segmentation',
        'model': {
            'img_size': [64, 64],
            'patch_size': 16,
            'embed_dim': 32,
            'depth': 2,
            'num_heads': 2,
            'mlp_ratio': 2.0,
            'cfa_pattern': 'rggb',
            'use_rope_2d': True,
            'seg_hidden_dim': 32,
            'seg_output_size': [64, 64],
        },
        'data': {'num_classes': 5},
    }


# ----------------------------------------------------------------------------
# Detection loss
# ----------------------------------------------------------------------------

class TestDetectionLossPrimitives(unittest.TestCase):
    def test_generate_anchors_shape(self):
        anchors = generate_anchors((4, 4), stride=16, anchor_scales=(1.0, 1.26), anchor_ratios=(0.5, 1.0, 2.0))
        self.assertEqual(anchors.shape, (4 * 4 * 6, 4))

    def test_box_iou_identity(self):
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
        iou = box_iou(boxes, boxes)
        self.assertTrue(torch.allclose(torch.diag(iou), torch.ones(2), atol=1e-5))
        self.assertGreater(iou[0, 1].item(), 0.0)
        self.assertLess(iou[0, 1].item(), 1.0)

    def test_encode_decode_boxes_roundtrip(self):
        anchors = torch.tensor([[0.0, 0.0, 20.0, 20.0], [10.0, 10.0, 30.0, 40.0]])
        gt_boxes = torch.tensor([[2.0, 2.0, 18.0, 22.0], [12.0, 15.0, 28.0, 35.0]])
        deltas = encode_boxes(anchors, gt_boxes)
        decoded = decode_boxes(anchors, deltas)
        self.assertTrue(torch.allclose(decoded, gt_boxes, atol=1e-3))


class TestDetectionLoss(unittest.TestCase):
    def setUp(self):
        self.num_classes = 3
        self.strides = (16, 32)
        self.loss_fn = DetectionLoss(num_classes=self.num_classes, strides=self.strides)

    def _make_preds(self, batch_size=2, num_anchors=9):
        # Feature maps at strides 16 (4x4) and 32 (2x2) for a 64x64 input.
        cls_scores = [
            torch.randn(batch_size, num_anchors * self.num_classes, 4, 4, requires_grad=True),
            torch.randn(batch_size, num_anchors * self.num_classes, 2, 2, requires_grad=True),
        ]
        bbox_preds = [
            torch.randn(batch_size, num_anchors * 4, 4, 4, requires_grad=True),
            torch.randn(batch_size, num_anchors * 4, 2, 2, requires_grad=True),
        ]
        return cls_scores, bbox_preds

    def test_forward_with_positive_targets(self):
        cls_scores, bbox_preds = self._make_preds()
        targets = [
            {'boxes': torch.tensor([[5.0, 5.0, 30.0, 30.0]]), 'labels': torch.tensor([1])},
            {'boxes': torch.tensor([[10.0, 10.0, 40.0, 50.0]]), 'labels': torch.tensor([0])},
        ]
        out = self.loss_fn(cls_scores, bbox_preds, targets)
        for key in ('loss_cls', 'loss_reg', 'loss_total', 'num_pos'):
            self.assertIn(key, out)
        self.assertGreater(out['num_pos'].item(), 0)
        self.assertTrue(torch.isfinite(out['loss_total']))
        # Gradients should flow back to the raw predictions.
        out['loss_total'].backward()
        self.assertIsNotNone(cls_scores[0].grad)
        self.assertIsNotNone(bbox_preds[0].grad)

    def test_forward_with_empty_targets(self):
        cls_scores, bbox_preds = self._make_preds()
        targets = [
            {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)},
            {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)},
        ]
        out = self.loss_fn(cls_scores, bbox_preds, targets)
        self.assertEqual(out['num_pos'].item(), 0)
        self.assertTrue(torch.isfinite(out['loss_total']))

    def test_giou_reg_loss_type(self):
        loss_fn = DetectionLoss(num_classes=self.num_classes, strides=self.strides, reg_loss_type='giou')
        cls_scores, bbox_preds = self._make_preds()
        targets = [
            {'boxes': torch.tensor([[5.0, 5.0, 30.0, 30.0]]), 'labels': torch.tensor([1])},
            {'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)},
        ]
        out = loss_fn(cls_scores, bbox_preds, targets)
        self.assertTrue(torch.isfinite(out['loss_total']))


# ----------------------------------------------------------------------------
# Segmentation loss
# ----------------------------------------------------------------------------

class TestSegmentationLoss(unittest.TestCase):
    def test_ce_only(self):
        loss_fn = SegmentationLoss(num_classes=5, ce_weight=1.0, rmi_weight=0.0)
        logits = torch.randn(2, 5, 16, 16, requires_grad=True)
        targets = torch.randint(0, 5, (2, 16, 16))
        out = loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(out['loss_total']))
        out['loss_total'].backward()
        self.assertIsNotNone(logits.grad)

    def test_ce_plus_rmi(self):
        # Use a larger spatial size than the other tests here: RMI's patch
        # covariance matrices can be near-singular (Cholesky failure) when
        # there are too few patches to estimate a stable covariance from,
        # which is a small-test-tensor numerical-conditioning artifact, not
        # a property of the loss itself (real segmentation inputs are far
        # larger than 16x16).
        loss_fn = SegmentationLoss(num_classes=4, ce_weight=1.0, rmi_weight=0.1, rmi_downsampling_ratio=2)
        logits = torch.randn(2, 4, 64, 64, requires_grad=True)
        targets = torch.randint(0, 4, (2, 64, 64))
        out = loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(out['loss_total']))
        out['loss_total'].backward()

    def test_ignore_index_excluded(self):
        loss_fn = SegmentationLoss(num_classes=3, ce_weight=1.0, rmi_weight=0.0, ignore_index=255)
        logits = torch.randn(1, 3, 8, 8)
        targets = torch.full((1, 8, 8), 255, dtype=torch.long)
        out = loss_fn(logits, targets)
        # All pixels ignored -> CrossEntropyLoss returns nan by convention
        # when there are zero valid pixels; just check it doesn't raise.
        self.assertIn('loss_total', out)

    def test_auxiliary_loss(self):
        # rmi_weight defaults > 0, and RMI's -log-det term can be negative
        # (it's a mutual-information lower bound, not a non-negative
        # divergence), so `loss_aux` isn't guaranteed positive -- just check
        # it's actually computed (finite, and combined into loss_total).
        loss_fn = SegmentationLoss(num_classes=3, aux_weight=0.4)
        logits = torch.randn(1, 3, 8, 8)
        aux_logits = torch.randn(1, 3, 4, 4)
        targets = torch.randint(0, 3, (1, 8, 8))
        out = loss_fn(logits, targets, aux_logits=aux_logits)
        self.assertTrue(torch.isfinite(out['loss_aux']))
        expected_total = out['loss_main'] + 0.4 * out['loss_aux']
        self.assertTrue(torch.allclose(out['loss_total'], expected_total))

    def test_resizes_logits_to_target(self):
        loss_fn = SegmentationLoss(num_classes=3, rmi_weight=0.0)
        logits = torch.randn(1, 3, 4, 4)  # coarser than target
        targets = torch.randint(0, 3, (1, 16, 16))
        out = loss_fn(logits, targets)
        self.assertTrue(torch.isfinite(out['loss_total']))

    def test_rmi_loss_standalone_shape(self):
        rmi = RMILoss(num_classes=4, rmi_radius=2, downsampling_ratio=1)
        logits = torch.randn(1, 4, 8, 8)
        targets = torch.randint(0, 4, (1, 8, 8))
        loss = rmi(logits, targets)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))


# ----------------------------------------------------------------------------
# PerceptionModel / build_perception_model
# ----------------------------------------------------------------------------

class TestPerceptionModel(unittest.TestCase):
    def test_build_detection_model_forward(self):
        config = _tiny_det_config()
        model = build_perception_model(config)
        self.assertIsInstance(model, PerceptionModel)
        x = torch.randn(2, 1, 64, 64)
        cls_scores, bbox_preds = model(x)
        self.assertEqual(len(cls_scores), len(model.get_strides()))
        for cs, bp in zip(cls_scores, bbox_preds):
            self.assertEqual(cs.shape[0], 2)
            self.assertEqual(bp.shape[0], 2)

    def test_build_segmentation_model_forward(self):
        config = _tiny_seg_config()
        model = build_perception_model(config)
        x = torch.randn(2, 1, 64, 64)
        seg_logits = model(x)
        self.assertEqual(seg_logits.shape, (2, 5, 64, 64))

    def test_get_strides_matches_neck(self):
        config = _tiny_det_config()
        model = build_perception_model(config)
        strides = model.get_strides()
        self.assertEqual(len(strides), 4)  # default 4-level pyramid

    def test_get_strides_raises_for_segmentation(self):
        config = _tiny_seg_config()
        model = build_perception_model(config)
        with self.assertRaises(ValueError):
            model.get_strides()

    def test_routing_active_property(self):
        config = _tiny_det_config()
        config['model']['use_sparse_routing'] = True
        config['model']['router_type'] = 'saliency'
        config['model']['route_at_inference'] = False
        model = build_perception_model(config)

        model.train()
        self.assertTrue(model.routing_active)
        model.eval()
        self.assertFalse(model.routing_active)  # route_at_inference=False -> dense at eval

        config['model']['route_at_inference'] = True
        model2 = build_perception_model(config)
        model2.eval()
        self.assertTrue(model2.routing_active)

    def test_invalid_task_raises(self):
        backbone = build_perception_model(_tiny_det_config()).backbone
        with self.assertRaises(ValueError):
            PerceptionModel(backbone=backbone, neck=None, head=torch.nn.Identity(), task='bogus')


# ----------------------------------------------------------------------------
# Detection postprocessing
# ----------------------------------------------------------------------------

class TestPostprocessDetections(unittest.TestCase):
    def test_postprocess_returns_valid_structure(self):
        config = _tiny_det_config()
        model = build_perception_model(config)
        model.eval()
        x = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            cls_scores, bbox_preds = model(x)
        results = postprocess_detections(
            cls_scores, bbox_preds, strides=model.get_strides(), num_classes=3,
            score_thresh=0.0, nms_thresh=0.5, image_size=(64, 64),
        )
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn('boxes', r)
            self.assertIn('scores', r)
            self.assertIn('labels', r)
            self.assertEqual(r['boxes'].shape[-1], 4)

    def test_high_score_thresh_yields_no_boxes(self):
        config = _tiny_det_config()
        model = build_perception_model(config)
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            cls_scores, bbox_preds = model(x)
        results = postprocess_detections(
            cls_scores, bbox_preds, strides=model.get_strides(), num_classes=3,
            score_thresh=1.1, image_size=(64, 64),
        )
        self.assertEqual(results[0]['boxes'].shape[0], 0)


# ----------------------------------------------------------------------------
# Checkpoint utils
# ----------------------------------------------------------------------------

class TestCheckpointUtils(unittest.TestCase):
    def test_strip_ddp_prefix(self):
        sd = {'module.a.weight': 1, 'module.b.bias': 2}
        stripped = strip_ddp_prefix(sd)
        self.assertEqual(set(stripped.keys()), {'a.weight', 'b.bias'})

    def test_strip_ddp_prefix_noop_without_prefix(self):
        sd = {'a.weight': 1}
        self.assertEqual(strip_ddp_prefix(sd), sd)

    def test_save_and_load_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            ckpt_path = os.path.join(tmp, 'ckpt.pth')
            save_checkpoint(ckpt_path, model, optimizer, epoch=3, global_step=100,
                             best_metric=0.5, config={'foo': 'bar'})

            new_model = torch.nn.Linear(4, 2)
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
            ckpt = load_checkpoint(ckpt_path, model=new_model, optimizer=new_optimizer)

            self.assertEqual(ckpt['epoch'], 3)
            self.assertEqual(ckpt['global_step'], 100)
            self.assertEqual(ckpt['best_metric'], 0.5)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_load_weights_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(4, 2)
            ckpt_path = os.path.join(tmp, 'ckpt.pth')
            save_checkpoint(ckpt_path, model, epoch=0, global_step=0)

            new_model = torch.nn.Linear(4, 2)
            load_weights_only(ckpt_path, new_model)
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    def test_find_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_latest_checkpoint(tmp))
            model = torch.nn.Linear(2, 2)
            for epoch in range(3):
                save_checkpoint(os.path.join(tmp, f'checkpoint_epoch_{epoch:04d}.pth'), model, epoch=epoch, global_step=0)
            latest = find_latest_checkpoint(tmp)
            self.assertIsNotNone(latest)
            self.assertIn('0002', latest)


# ----------------------------------------------------------------------------
# Config utils
# ----------------------------------------------------------------------------

class TestConfigUtils(unittest.TestCase):
    def test_load_yaml_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'cfg.yaml')
            with open(path, 'w') as f:
                yaml.safe_dump({'a': 1, 'b': {'c': 2}}, f)
            cfg = load_config(path)
            self.assertEqual(cfg['a'], 1)
            self.assertEqual(cfg.b.c, 2)  # attribute-style access

    def test_base_inheritance_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_path = os.path.join(tmp, 'base.yaml')
            child_path = os.path.join(tmp, 'child.yaml')
            with open(base_path, 'w') as f:
                yaml.safe_dump({'model': {'embed_dim': 768, 'depth': 12}}, f)
            with open(child_path, 'w') as f:
                yaml.safe_dump({'_base_': 'base.yaml', 'model': {'embed_dim': 64}}, f)

            cfg = load_config(child_path)
            self.assertEqual(cfg['model']['embed_dim'], 64)   # overridden
            self.assertEqual(cfg['model']['depth'], 12)       # inherited

    def test_circular_inheritance_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_path = os.path.join(tmp, 'a.yaml')
            b_path = os.path.join(tmp, 'b.yaml')
            with open(a_path, 'w') as f:
                yaml.safe_dump({'_base_': 'b.yaml'}, f)
            with open(b_path, 'w') as f:
                yaml.safe_dump({'_base_': 'a.yaml'}, f)
            with self.assertRaises(ValueError):
                load_config(a_path)

    def test_apply_cli_overrides_types(self):
        cfg = load_config.__globals__['_to_config_dict']({'training': {'lr': 0.1, 'epochs': 10}})
        apply_cli_overrides(cfg, [
            'training.lr=0.01', 'training.epochs=20', 'training.flag=true',
            'training.name=null', 'model.img_size=[256,256]',
        ])
        self.assertAlmostEqual(cfg['training']['lr'], 0.01)
        self.assertEqual(cfg['training']['epochs'], 20)
        self.assertIs(cfg['training']['flag'], True)
        self.assertIsNone(cfg['training']['name'])
        self.assertEqual(cfg['model']['img_size'], [256, 256])

    def test_apply_cli_overrides_noop_on_none(self):
        cfg = load_config.__globals__['_to_config_dict']({'a': 1})
        result = apply_cli_overrides(cfg, None)
        self.assertEqual(result, cfg)

    def test_save_config_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config.__globals__['_to_config_dict']({'a': 1, 'b': {'c': [1, 2, 3]}})
            out_path = os.path.join(tmp, 'out.yaml')
            save_config(cfg, out_path)
            reloaded = load_config(out_path)
            self.assertEqual(reloaded['a'], 1)
            self.assertEqual(reloaded['b']['c'], [1, 2, 3])


# ----------------------------------------------------------------------------
# Evaluation metrics
# ----------------------------------------------------------------------------

class TestSegmentationEvaluator(unittest.TestCase):
    def test_perfect_prediction_gives_miou_one(self):
        evaluator = SegmentationEvaluator(num_classes=3, ignore_index=255)
        targets = torch.randint(0, 3, (2, 8, 8))
        evaluator.update(targets.clone(), targets)
        metrics = evaluator.compute()
        self.assertAlmostEqual(metrics['mIoU'], 1.0, places=5)
        self.assertAlmostEqual(metrics['pixel_acc'], 1.0, places=5)

    def test_ignore_index_excluded(self):
        evaluator = SegmentationEvaluator(num_classes=2, ignore_index=255)
        targets = torch.full((1, 4, 4), 255, dtype=torch.long)
        targets[0, 0, 0] = 0
        preds = torch.zeros_like(targets)
        evaluator.update(preds, targets)
        metrics = evaluator.compute()
        self.assertGreaterEqual(metrics['pixel_acc'], 0.0)

    def test_reset(self):
        evaluator = SegmentationEvaluator(num_classes=2)
        targets = torch.zeros(1, 2, 2, dtype=torch.long)
        evaluator.update(targets, targets)
        evaluator.reset()
        self.assertEqual(evaluator.confusion_matrix.sum(), 0)


class TestDetectionEvaluator(unittest.TestCase):
    def test_perfect_predictions_high_ap(self):
        evaluator = DetectionEvaluator(num_classes=2)
        gt_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        gt_labels = torch.tensor([0])
        preds = [{
            'boxes': gt_boxes.clone(),
            'scores': torch.tensor([0.99]),
            'labels': gt_labels.clone(),
        }]
        targets = [{'boxes': gt_boxes, 'labels': gt_labels}]
        evaluator.update([1], preds, targets, image_sizes=[(64, 64)])
        metrics = evaluator.compute()
        # Either the pycocotools path ('mAP') or the fallback ('approx_AP50') is used.
        self.assertTrue('mAP' in metrics or 'approx_AP50' in metrics)
        score = metrics.get('mAP', metrics.get('approx_AP50'))
        self.assertGreater(score, 0.5)

    def test_no_predictions_zero_ap(self):
        evaluator = DetectionEvaluator(num_classes=2)
        gt_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        gt_labels = torch.tensor([0])
        preds = [{'boxes': torch.zeros(0, 4), 'scores': torch.zeros(0), 'labels': torch.zeros(0, dtype=torch.long)}]
        targets = [{'boxes': gt_boxes, 'labels': gt_labels}]
        evaluator.update([1], preds, targets, image_sizes=[(64, 64)])
        metrics = evaluator.compute()
        score = metrics.get('mAP', metrics.get('approx_AP50'))
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_reset(self):
        evaluator = DetectionEvaluator(num_classes=2)
        gt_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        gt_labels = torch.tensor([0])
        preds = [{'boxes': gt_boxes, 'scores': torch.tensor([0.9]), 'labels': gt_labels}]
        evaluator.update([1], preds, [{'boxes': gt_boxes, 'labels': gt_labels}], image_sizes=[(64, 64)])
        evaluator.reset()
        self.assertEqual(len(evaluator.predictions), 0)
        self.assertEqual(len(evaluator.ground_truth), 0)


# ----------------------------------------------------------------------------
# Efficiency metrics (CPU-safe per photon2perception/evaluation/efficiency.py)
# ----------------------------------------------------------------------------

class TestEfficiencyMetrics(unittest.TestCase):
    def test_measure_latency_cpu(self):
        model = torch.nn.Linear(16, 16)

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x.flatten(1))

        result = measure_latency(Wrap(model), input_shape=(2, 1, 4, 4), num_warmup=2, num_runs=3, device='cpu')
        self.assertIn('mean_latency_ms', result)
        self.assertGreaterEqual(result['mean_latency_ms'], 0.0)
        self.assertEqual(result['batch_size'], 2)

    def test_measure_memory_cpu_zero_peak(self):
        model = torch.nn.Conv2d(1, 4, 3, padding=1)
        result = measure_memory(model, input_shape=(1, 1, 8, 8), device='cpu')
        self.assertEqual(result['peak_gpu_memory_mb'], 0.0)
        self.assertGreater(result['total_params'], 0)

    def test_compute_input_bandwidth_bayer_vs_rgb(self):
        bayer = compute_input_bandwidth((1, 1, 100, 100), input_format='bayer')
        rgb = compute_input_bandwidth((1, 3, 100, 100), input_format='rgb')
        self.assertLess(bayer['input_bytes_per_image_8bit'], rgb['input_bytes_per_image_8bit'])

    def test_full_efficiency_report_smoke(self):
        model = build_perception_model(_tiny_seg_config())
        report = full_efficiency_report(model, input_shape=(1, 1, 64, 64), device='cpu')
        self.assertIn('latency', report)
        self.assertIn('memory', report)
        self.assertIn('bandwidth', report)
        self.assertIn('flops', report)


# ----------------------------------------------------------------------------
# tools/export.py (subprocess, exercises the real CLI end-to-end)
# ----------------------------------------------------------------------------

class TestExportScript(unittest.TestCase):
    def _run_export(self, config_dict, output_stem, extra_args=None):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        cfg_path = os.path.join(tmp, 'cfg.yaml')
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(config_dict, f)
        output_path = os.path.join(tmp, output_stem)
        cmd = [
            sys.executable, str(REPO_ROOT / 'tools' / 'export.py'),
            '--config', cfg_path, '--output', output_path,
            '--format', 'both', '--device', 'cpu',
        ] + (extra_args or [])
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        return result, tmp

    def test_export_detection_both_formats(self):
        result, tmp = self._run_export(_tiny_det_config(), 'det_out')
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertTrue(os.path.exists(os.path.join(tmp, 'det_out.pt')))
        self.assertTrue(os.path.exists(os.path.join(tmp, 'det_out.onnx')))
        self.assertIn('parity check passed', result.stdout)

    def test_export_segmentation_both_formats(self):
        result, tmp = self._run_export(_tiny_seg_config(), 'seg_out')
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertTrue(os.path.exists(os.path.join(tmp, 'seg_out.pt')))
        self.assertTrue(os.path.exists(os.path.join(tmp, 'seg_out.onnx')))

    def test_export_routing_without_route_at_inference_fails(self):
        config = _tiny_det_config()
        config['model']['use_sparse_routing'] = True
        config['model']['route_at_inference'] = False
        result, _ = self._run_export(config, 'should_fail')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('route_at_inference', result.stdout + result.stderr)

    def test_export_routing_with_route_at_inference_succeeds(self):
        config = _tiny_det_config()
        config['model']['use_sparse_routing'] = True
        config['model']['route_at_inference'] = True
        config['model']['router_type'] = 'saliency'
        result, tmp = self._run_export(config, 'routed_out', extra_args=['--attn_backend', 'math'])
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_print_npu_notes_standalone(self):
        cmd = [sys.executable, str(REPO_ROOT / 'tools' / 'export.py'), '--print_npu_notes']
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn('Ascend', result.stdout)


# ----------------------------------------------------------------------------
# photon2perception.inference (PerceptionInferencer)
# ----------------------------------------------------------------------------

class TestPerceptionInferencer(unittest.TestCase):
    """Exercises pytorch/torchscript/onnxruntime backends. TensorRT is
    excluded (requires an NVIDIA GPU + tensorrt/pycuda, not available in CI/
    CPU-only dev environments -- see `_TensorRTBackend`'s deferred import).
    """

    @classmethod
    def setUpClass(cls):
        from photon2perception.inference import PerceptionInferencer
        cls.PerceptionInferencer = PerceptionInferencer

        cls.tmp_dir = tempfile.mkdtemp()
        cls.cfg_path = os.path.join(cls.tmp_dir, 'det_cfg.yaml')
        with open(cls.cfg_path, 'w') as f:
            yaml.safe_dump(_tiny_det_config(), f)

        export_cmd = [
            sys.executable, str(REPO_ROOT / 'tools' / 'export.py'),
            '--config', cls.cfg_path,
            '--output', os.path.join(cls.tmp_dir, 'model'),
            '--format', 'both', '--device', 'cpu',
        ]
        result = subprocess.run(export_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, result.stdout + result.stderr
        cls.ts_path = os.path.join(cls.tmp_dir, 'model.pt')
        cls.onnx_path = os.path.join(cls.tmp_dir, 'model.onnx')

    def test_pytorch_backend_predict(self):
        inferencer = self.PerceptionInferencer(self.cfg_path, backend='pytorch', device='cpu')
        dummy = np.random.rand(64, 64).astype(np.float32)
        results = inferencer.predict(dummy)
        self.assertEqual(len(results), 1)
        self.assertIn('boxes', results[0])

    def test_preprocess_handles_various_shapes(self):
        inferencer = self.PerceptionInferencer(self.cfg_path, backend='pytorch', device='cpu')
        for shape in [(64, 64), (1, 64, 64), (1, 1, 64, 64)]:
            arr = np.random.rand(*shape).astype(np.float32)
            out = inferencer.preprocess(arr)
            self.assertEqual(out.shape, (1, 1, 64, 64))

    def test_preprocess_resizes_mismatched_input(self):
        inferencer = self.PerceptionInferencer(self.cfg_path, backend='pytorch', device='cpu')
        arr = np.random.rand(32, 32).astype(np.float32)
        out = inferencer.preprocess(arr)
        self.assertEqual(out.shape, (1, 1, 64, 64))

    def test_torchscript_backend_matches_pytorch(self):
        pt = self.PerceptionInferencer(self.cfg_path, backend='pytorch', device='cpu')
        ts = self.PerceptionInferencer(self.cfg_path, backend='torchscript', weights_path=self.ts_path, device='cpu')
        dummy = np.random.rand(64, 64).astype(np.float32)
        raw_pt = pt.predict_raw(dummy)
        raw_ts = ts.predict_raw(dummy)
        for a, b in zip(raw_pt, raw_ts):
            self.assertTrue(np.allclose(a, b, atol=1e-3))

    def test_onnxruntime_backend_matches_pytorch(self):
        pt = self.PerceptionInferencer(self.cfg_path, backend='pytorch', device='cpu')
        ort = self.PerceptionInferencer(self.cfg_path, backend='onnxruntime', weights_path=self.onnx_path, device='cpu')
        dummy = np.random.rand(64, 64).astype(np.float32)
        raw_pt = pt.predict_raw(dummy)
        raw_ort = ort.predict_raw(dummy)
        for a, b in zip(raw_pt, raw_ort):
            self.assertTrue(np.allclose(a, b, atol=1e-3))
        ort.close()

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            self.PerceptionInferencer(self.cfg_path, backend='bogus_backend')

    def test_missing_weights_path_raises_for_non_pytorch_backend(self):
        with self.assertRaises(ValueError):
            self.PerceptionInferencer(self.cfg_path, backend='torchscript', weights_path=None)

    def test_benchmark_returns_latency_stats(self):
        inferencer = self.PerceptionInferencer(self.cfg_path, backend='pytorch', device='cpu')
        stats = inferencer.benchmark(num_warmup=1, num_runs=2)
        self.assertIn('mean_latency_ms', stats)
        self.assertEqual(stats['backend'], 'pytorch')

    def test_segmentation_task_predict(self):
        seg_cfg_path = os.path.join(self.tmp_dir, 'seg_cfg.yaml')
        with open(seg_cfg_path, 'w') as f:
            yaml.safe_dump(_tiny_seg_config(), f)
        inferencer = self.PerceptionInferencer(seg_cfg_path, backend='pytorch', device='cpu')
        dummy = np.random.rand(64, 64).astype(np.float32)
        seg_map = inferencer.predict(dummy)
        self.assertEqual(seg_map.shape, (1, 64, 64))


if __name__ == '__main__':
    unittest.main()
