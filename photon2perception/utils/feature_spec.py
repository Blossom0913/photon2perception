"""
Feature-spec (`.pb.txt`) read/write utilities.

Why this exists
----------------
`configs/{detection,segmentation}/*.yaml` describe the *model* (embed_dim,
depth, ...) but say nothing machine-readable about the exact tensors the
model consumes/produces -- their names, shapes, and dtypes. That
information is needed by:

- `tools/export_features.py` (this project's feature exporter), to know
  what tensor layout to write to disk for a given `img_size`/`patch_size`.
- Downstream serving / feature-platform tooling (e.g. a feature store, or
  an NPU/edge deployment toolchain) that wants a single source of truth
  for "what does this model expect as input" without importing Python or
  parsing YAML.

We describe each input/output feature in a small, dependency-free
protobuf-text-*like* format (`.pb.txt`), the same idea as TensorFlow's
`FeatureConfig`/`GraphDef` `.pbtxt` files or ONNX's human-readable
`ValueInfoProto` dumps, but with a minimal hand-rolled schema (no `protobuf`
package dependency, so this works on a bare Python + PyYAML install):

    feature {
      name: "bayer_raw"
      dtype: "float32"
      layout: "NCHW"
      dim { name: "batch" size: -1 }
      dim { name: "channel" size: 1 }
      dim { name: "height" size: 512 }
      dim { name: "width" size: 512 }
      description: "Bayer RAW input, CFA-mosaicked, normalized to [-1, 1]"
    }
    feature {
      name: "cls_scores_p3"
      ...
    }

`size: -1` denotes a dynamic dimension (batch dimension). This module
provides:
- `FeatureDim` / `FeatureSpec` / `FeatureSpecCollection` dataclasses.
- `dump_pbtxt` / `load_pbtxt` to write/read the `.pb.txt` text format above.
- `build_input_feature_specs(config)` / `build_output_feature_specs(config)`
  to derive the canonical I/O feature specs directly from an experiment
  config dict (`configs/**/*.yaml`), so the `.pb.txt` files are always
  regenerable and never hand-drift from the model definition.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class FeatureDim:
    """A single named dimension of a feature's shape.

    `size == -1` means dynamic (typically the batch dimension).
    """
    name: str
    size: int

    def to_pbtxt(self, indent: str = '  ') -> str:
        return f'{indent}dim {{ name: "{self.name}" size: {self.size} }}'


@dataclass
class FeatureSpec:
    """Describes one model input or output tensor."""
    name: str
    dtype: str                      # e.g. 'float32', 'int64', 'uint8'
    layout: str                     # e.g. 'NCHW', 'NHWC', 'N' (list-of-dicts), 'N,num_classes,H,W'
    dims: List[FeatureDim] = field(default_factory=list)
    description: str = ""

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape tuple, with dynamic dims reported as -1."""
        return tuple(d.size for d in self.dims)

    def to_pbtxt(self) -> str:
        lines = ["feature {"]
        lines.append(f'  name: "{self.name}"')
        lines.append(f'  dtype: "{self.dtype}"')
        lines.append(f'  layout: "{self.layout}"')
        for d in self.dims:
            lines.append(d.to_pbtxt())
        if self.description:
            escaped = self.description.replace('"', '\\"')
            lines.append(f'  description: "{escaped}"')
        lines.append("}")
        return "\n".join(lines)


@dataclass
class FeatureSpecCollection:
    """A named group of `FeatureSpec`s (e.g. all model inputs, or all
    detection-head outputs), serializable to/from a single `.pb.txt` file.
    """
    group: str                      # e.g. 'inputs', 'outputs'
    task: str = ""                  # 'detection' | 'segmentation'
    features: List[FeatureSpec] = field(default_factory=list)

    def to_pbtxt(self) -> str:
        header = [
            "# Auto-generated feature spec (pb.txt). Do not hand-edit;",
            "# regenerate via `python -m photon2perception.utils.feature_spec` or",
            "# `tools/export_features.py --emit_spec_only`.",
            f'group: "{self.group}"',
        ]
        if self.task:
            header.append(f'task: "{self.task}"')
        body = "\n".join(f.to_pbtxt() for f in self.features)
        return "\n".join(header) + "\n" + body + "\n"

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, 'w') as f:
            f.write(self.to_pbtxt())

    def get(self, name: str) -> Optional[FeatureSpec]:
        for feat in self.features:
            if feat.name == name:
                return feat
        return None


# ----------------------------------------------------------------------------
# pb.txt parsing (minimal, dependency-free)
# ----------------------------------------------------------------------------

_FEATURE_BLOCK_RE = re.compile(r"feature\s*\{(.*?)\n\}", re.DOTALL)
# Matches a scalar `key: "value"` line, but NOT one nested inside a `dim { ... }`
# block on the same line (those are consumed separately by `_DIM_RE`).
_SCALAR_FIELD_RE = re.compile(r'^\s*(\w+)\s*:\s*"((?:[^"\\]|\\.)*)"\s*$', re.MULTILINE)
_DIM_RE = re.compile(r'dim\s*\{\s*name:\s*"([^"]*)"\s*size:\s*(-?\d+)\s*\}')
_TOPLEVEL_FIELD_RE = re.compile(r'^(\w+)\s*:\s*"((?:[^"\\]|\\.)*)"\s*$', re.MULTILINE)


def load_pbtxt(path: str) -> FeatureSpecCollection:
    """Parse a `.pb.txt` file written by `FeatureSpecCollection.to_pbtxt`."""
    with open(path, 'r') as f:
        text = f.read()

    # Top-level fields (group/task) live before the first `feature {` block.
    preamble = text.split('feature {', 1)[0]
    top_fields = dict(_TOPLEVEL_FIELD_RE.findall(preamble))
    collection = FeatureSpecCollection(
        group=top_fields.get('group', ''),
        task=top_fields.get('task', ''),
    )

    for block_match in _FEATURE_BLOCK_RE.finditer(text):
        block = block_match.group(1)
        dims = [FeatureDim(name=n, size=int(s)) for n, s in _DIM_RE.findall(block)]
        # Strip `dim { ... }` sub-blocks before matching scalar fields, so a
        # dim's `name: "..."` never gets mistaken for the feature's own `name`.
        scalar_block = _DIM_RE.sub('', block)
        scalars = dict(_SCALAR_FIELD_RE.findall(scalar_block))
        collection.features.append(FeatureSpec(
            name=scalars.get('name', ''),
            dtype=scalars.get('dtype', ''),
            layout=scalars.get('layout', ''),
            dims=dims,
            description=scalars.get('description', '').replace('\\"', '"'),
        ))
    return collection


def dump_pbtxt(collection: FeatureSpecCollection, path: str) -> None:
    collection.save(path)


# ----------------------------------------------------------------------------
# Deriving feature specs from an experiment config
# ----------------------------------------------------------------------------

def build_input_feature_specs(config: Dict[str, Any]) -> FeatureSpecCollection:
    """Derive the canonical *input* feature spec from an experiment config.

    The model's sole tensor input is the Bayer RAW image, shape
    `(B, in_chans, H, W)`. An optional second input, the unnormalized RAW
    (used only by `PhysicalPriorRouter`), shares the same shape.
    """
    task = config.get('task', 'detection')
    model_cfg = config['model']
    img_h, img_w = tuple(model_cfg['img_size'])
    in_chans = model_cfg.get('in_chans', 1)
    patch_size = model_cfg['patch_size']
    normalize = config.get('data', {}).get('normalize', True)

    specs = [
        FeatureSpec(
            name='bayer_raw',
            dtype='float32',
            layout='NCHW',
            dims=[
                FeatureDim('batch', -1),
                FeatureDim('channel', in_chans),
                FeatureDim('height', img_h),
                FeatureDim('width', img_w),
            ],
            description=(
                f"Bayer RAW input, CFA pattern={model_cfg.get('cfa_pattern', 'rggb')}, "
                f"patch_size={patch_size} (H and W must be divisible by it), "
                + ("normalized to [-1, 1]" if normalize else "raw range [0, 1]")
            ),
        ),
        FeatureSpec(
            name='raw_image_unnormalized',
            dtype='float32',
            layout='NCHW',
            dims=[
                FeatureDim('batch', -1),
                FeatureDim('channel', in_chans),
                FeatureDim('height', img_h),
                FeatureDim('width', img_w),
            ],
            description=(
                "Optional unnormalized Bayer RAW in [0, 1], used only by "
                "PhysicalPriorRouter's local-variance saliency prior when "
                "model.use_sparse_routing=true and model.router_type=physical. "
                "If omitted at inference time, `bayer_raw` is reused."
            ),
        ),
    ]
    return FeatureSpecCollection(group='inputs', task=task, features=specs)


def build_output_feature_specs(config: Dict[str, Any]) -> FeatureSpecCollection:
    """Derive the canonical *output* feature spec from an experiment config.

    - detection: one (cls_scores, bbox_preds) pair of tensors per FPN level,
      strides derived the same way `tools/train.py::build_loss` does.
    - segmentation: a single (B, num_classes, H, W) logits tensor.
    """
    task = config.get('task', 'detection')
    model_cfg = config['model']
    num_classes = config['data']['num_classes']
    img_h, img_w = tuple(model_cfg['img_size'])
    patch_size = model_cfg['patch_size']
    specs: List[FeatureSpec] = []

    if task == 'detection':
        scale_factors = model_cfg.get('neck_scale_factors', (4.0, 2.0, 1.0, 0.5))
        strides = model_cfg.get(
            'neck_strides', [int(round(patch_size / sf)) for sf in scale_factors]
        )
        num_anchors = model_cfg.get('num_anchors', 9)
        for level, stride in enumerate(strides):
            level_h, level_w = img_h // stride, img_w // stride
            # Naming matches tools/export.py::_output_names (`cls_scores_{i}` /
            # `bbox_preds_{i}`) so the .pb.txt and the exported ONNX/TorchScript
            # I/O names refer to the same tensors under the same names.
            specs.append(FeatureSpec(
                name=f'cls_scores_{level}',
                dtype='float32',
                layout='NCHW',
                dims=[
                    FeatureDim('batch', -1),
                    FeatureDim('channel', num_anchors * num_classes),
                    FeatureDim('height', level_h),
                    FeatureDim('width', level_w),
                ],
                description=f"Classification logits, FPN level {level}, stride={stride}",
            ))
            specs.append(FeatureSpec(
                name=f'bbox_preds_{level}',
                dtype='float32',
                layout='NCHW',
                dims=[
                    FeatureDim('batch', -1),
                    FeatureDim('channel', num_anchors * 4),
                    FeatureDim('height', level_h),
                    FeatureDim('width', level_w),
                ],
                description=f"Box regression (dx, dy, dw, dh), FPN level {level}, stride={stride}",
            ))
    else:
        seg_h, seg_w = tuple(model_cfg.get('seg_output_size', (img_h, img_w)))
        specs.append(FeatureSpec(
            name='seg_logits',
            dtype='float32',
            layout='NCHW',
            dims=[
                FeatureDim('batch', -1),
                FeatureDim('channel', num_classes),
                FeatureDim('height', seg_h),
                FeatureDim('width', seg_w),
            ],
            description="Per-pixel class logits (pre-softmax/argmax)",
        ))

    return FeatureSpecCollection(group='outputs', task=task, features=specs)


def write_feature_specs_for_config(
    config: Dict[str, Any],
    output_dir: str,
    basename: Optional[str] = None,
) -> Tuple[str, str]:
    """Generate and write `{basename}.inputs.pb.txt` / `{basename}.outputs.pb.txt`
    for `config` into `output_dir`. Returns the two written paths.

    `basename` defaults to `config['data'].get('feature_spec_name')` if set,
    else the task name (e.g. 'detection').
    """
    basename = basename or config.get('data', {}).get('feature_spec_name') or config.get('task', 'model')
    os.makedirs(output_dir, exist_ok=True)

    input_path = os.path.join(output_dir, f'{basename}.inputs.pb.txt')
    output_path = os.path.join(output_dir, f'{basename}.outputs.pb.txt')

    build_input_feature_specs(config).save(input_path)
    build_output_feature_specs(config).save(output_path)
    return input_path, output_path


def _main():
    """CLI: `python -m photon2perception.utils.feature_spec --config <cfg.yaml> --output_dir <dir>`."""
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description="Generate input/output feature-spec .pb.txt files")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./feature_specs')
    parser.add_argument('--basename', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path, output_path = write_feature_specs_for_config(config, args.output_dir, args.basename)
    print(f"Wrote {input_path}")
    print(f"Wrote {output_path}")


if __name__ == '__main__':
    _main()
