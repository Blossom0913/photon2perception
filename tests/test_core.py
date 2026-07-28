#!/usr/bin/env python3
"""
Unit tests for Photon2Perception core modules.

Run with: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest


# ----- Tokenization Tests -----

class TestBayerPatchEmbed:
    """Test CFA-aware patch embedding."""

    def test_basic_forward(self):
        from photon2perception.models.tokenization.bayer_patch_embed import BayerPatchEmbed

        embed = BayerPatchEmbed(
            img_size=(224, 224),
            patch_size=16,
            in_chans=1,
            embed_dim=768,
        )

        x = torch.randn(2, 1, 224, 224)
        tokens = embed(x)

        assert tokens.shape == (2, 196, 768)  # (B, 14*14, D)
        assert tokens.dtype == torch.float32

    def test_patch_count(self):
        from photon2perception.models.tokenization.bayer_patch_embed import BayerPatchEmbed

        embed = BayerPatchEmbed(img_size=(512, 512), patch_size=32, embed_dim=384)
        assert embed.num_patches == 256  # 16x16

    def test_odd_patch_raises(self):
        from photon2perception.models.tokenization.bayer_patch_embed import BayerPatchEmbed

        with pytest.raises(ValueError, match='even'):
            BayerPatchEmbed(img_size=(224, 224), patch_size=15, embed_dim=768)

    def test_size_mismatch_raises(self):
        from photon2perception.models.tokenization.bayer_patch_embed import BayerPatchEmbed

        embed = BayerPatchEmbed(img_size=(224, 224), patch_size=16, embed_dim=768)
        x = torch.randn(2, 1, 256, 256)  # Wrong size
        with pytest.raises(ValueError):
            embed(x)

    def test_cfa_patterns(self):
        from photon2perception.models.tokenization.bayer_patch_embed import BayerPatchEmbed

        for pattern in ['rggb', 'bggr', 'grbg', 'gbrg']:
            embed = BayerPatchEmbed(img_size=(224, 224), patch_size=16,
                                    embed_dim=768, cfa_pattern=pattern)
            x = torch.randn(1, 1, 224, 224)
            tokens = embed(x)
            assert tokens.shape == (1, 196, 768)

    def test_without_cfa_embed(self):
        from photon2perception.models.tokenization.bayer_patch_embed import BayerPatchEmbed

        embed = BayerPatchEmbed(
            img_size=(224, 224), patch_size=16, embed_dim=768,
            use_cfa_embed=False,
        )
        x = torch.randn(1, 1, 224, 224)
        tokens = embed(x)
        assert tokens.shape == (1, 196, 768)


class TestBayerFineTokenize:
    """Test fine-grained Bayer tokenization."""

    def test_basic_forward(self):
        from photon2perception.models.tokenization.bayer_patch_embed import BayerFineTokenize

        tokenizer = BayerFineTokenize(embed_dim=192, merge_phases=False)
        x = torch.randn(2, 1, 224, 224)
        tokens = tokenizer(x)

        # 112*112 quads, 4 phases each, 192 dim
        assert tokens.shape == (2, 112 * 112, 4, 192)

    def test_merge_phases(self):
        from photon2perception.models.tokenization.bayer_patch_embed import BayerFineTokenize

        tokenizer = BayerFineTokenize(embed_dim=192, merge_phases=True)
        x = torch.randn(2, 1, 224, 224)
        tokens = tokenizer(x)

        assert tokens.shape == (2, 112 * 112, 768)  # 4 * 192 = 768


# ----- Position Encoding Tests -----

class TestRoPE2D:
    """Test 2D Rotary Position Embedding."""

    def test_basic_forward(self):
        from photon2perception.models.position_encoding.rope_2d import RoPE2D

        rope = RoPE2D(dim=256, grid_h=14, grid_w=14)
        x = torch.randn(2, 196, 256)
        x_rot = rope(x)

        assert x_rot.shape == x.shape
        # Rotation should preserve norm approximately
        assert torch.allclose(
            x.norm(dim=-1), x_rot.norm(dim=-1), atol=1e-4
        )

    def test_dim_not_divisible_by_4(self):
        from photon2perception.models.position_encoding.rope_2d import RoPE2D

        # dim=250 is divisible by neither 4 nor 8; RoPE2D requires the
        # embedding dim to be divisible by 8 (see precompute_2d_freqs_cis
        # docstring: D/2 must split evenly into 4 x/y/diag/anti-diag bands).
        with pytest.raises(ValueError, match='divisible by 8'):
            RoPE2D(dim=250, grid_h=14, grid_w=14)

    def test_grid_mismatch(self):
        from photon2perception.models.position_encoding.rope_2d import RoPE2D

        rope = RoPE2D(dim=256, grid_h=14, grid_w=14)
        x = torch.randn(2, 200, 256)  # Wrong token count (200 != 196)
        with pytest.raises(ValueError):
            rope(x)


# ----- Sparse Routing Tests -----

class TestSaliencyRouter:
    """Test saliency-aware token router."""

    def test_training_mode(self):
        from photon2perception.models.routing.router import SaliencyRouter

        router = SaliencyRouter(dim=256, keep_ratio=0.7)
        x = torch.randn(4, 100, 256)
        x_routed, scores = router(x, training=True)

        assert x_routed.shape == x.shape
        assert scores.shape == (4, 100, 1)

    def test_inference_mode(self):
        from photon2perception.models.routing.router import SaliencyRouter

        router = SaliencyRouter(dim=256, keep_ratio=0.7)
        x = torch.randn(4, 100, 256)
        x_routed, scores = router(x, training=False)

        # Check that approximately keep_ratio tokens are kept
        kept = (x_routed.abs().sum(dim=-1) > 0).float().mean()
        assert 0.5 < kept < 0.9  # Rough range

    def test_gradient_flow(self):
        from photon2perception.models.routing.router import SaliencyRouter

        router = SaliencyRouter(dim=256, keep_ratio=0.7)
        x = torch.randn(4, 100, 256, requires_grad=True)
        x_routed, scores = router(x, training=True)

        loss = x_routed.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.any(x.grad != 0)


class TestPhysicalPriorRouter:
    """Test physics-driven router."""

    def test_with_raw_image(self):
        from photon2perception.models.routing.router import PhysicalPriorRouter

        router = PhysicalPriorRouter(dim=256, patch_size=16, keep_ratio=0.7)
        x = torch.randn(2, 196, 256)
        raw = torch.randn(2, 1, 224, 224)

        x_routed, scores = router(x, raw_image=raw, grid_h=14, grid_w=14, training=False)

        assert x_routed.shape == x.shape


# ----- Data Pipeline Tests -----

class TestUnprocessing:
    """Test RGB-to-Bayer unprocessing pipeline."""

    def test_basic_unprocessing(self):
        from photon2perception.datasets.unprocessing import UnprocessPipeline

        unprocess = UnprocessPipeline(pattern='rggb', add_noise=False)
        srgb = torch.rand(2, 3, 256, 256)

        bayer = unprocess(srgb)

        assert bayer.shape == (2, 1, 256, 256)
        assert bayer.min() >= 0.0
        assert bayer.max() <= 1.0

    def test_with_noise(self):
        from photon2perception.datasets.unprocessing import UnprocessPipeline

        unprocess = UnprocessPipeline(
            pattern='rggb', add_noise=True,
            noise_params=(0.01, 0.001),
        )
        srgb = torch.rand(1, 3, 256, 256)
        bayer_clean = UnprocessPipeline(pattern='rggb', add_noise=False)(srgb)
        bayer_noisy = unprocess(srgb)

        # Noisy should differ from clean
        assert not torch.allclose(bayer_clean, bayer_noisy)

    def test_intermediates(self):
        from photon2perception.datasets.unprocessing import UnprocessPipeline

        unprocess = UnprocessPipeline(pattern='rggb', add_noise=True)
        srgb = torch.rand(1, 3, 256, 256)

        bayer, intermediates = unprocess(srgb, return_intermediates=True)

        assert 'linear_rgb' in intermediates
        assert 'camera_rgb' in intermediates
        assert 'sensor_raw' in intermediates
        assert 'bayer_clean' in intermediates
        assert 'bayer_noisy' in intermediates

    def test_bayer_mosaic(self):
        from photon2perception.datasets.unprocessing import bayer_mosaic

        rgb = torch.ones(1, 3, 4, 4)
        bayer = bayer_mosaic(rgb, pattern='rggb')

        assert bayer.shape == (1, 1, 4, 4)
        # At R position (0,0): should be from channel 0
        assert bayer[0, 0, 0, 0] == rgb[0, 0, 0, 0]
        # At B position (1,1): should be from channel 2 (B=G channel in RGGB)
        assert bayer[0, 0, 1, 1] == rgb[0, 1, 1, 1]  # G channels averaged


class TestMultiConditionUnprocess:
    """Test multi-condition unprocessing."""

    def test_conditions(self):
        from photon2perception.datasets.unprocessing import MultiConditionUnprocess

        srgb = torch.rand(1, 3, 256, 256)

        for condition in ['normal', 'dark', 'over_exp']:
            unprocess = MultiConditionUnprocess(condition=condition)
            bayer = unprocess(srgb)
            assert bayer.shape == (1, 1, 256, 256)
            assert bayer.min() >= 0.0
            assert bayer.max() <= 1.0


# ----- Full Model Tests -----

class TestRawViT:
    """Test the full RAW-adapted ViT backbone."""

    def test_basic_forward(self):
        from photon2perception.models.backbones.raw_vit import RawViT

        model = RawViT(
            img_size=(224, 224),
            patch_size=16,
            embed_dim=384,  # Smaller for testing
            depth=4,
            num_heads=6,
        )
        x = torch.randn(2, 1, 224, 224)
        cls_token, hidden_states = model(x)

        assert cls_token.shape == (2, 384)
        assert len(hidden_states) == 4
        assert hidden_states[0].shape == (2, 197, 384)  # 196 patches + CLS

    def test_with_rope(self):
        from photon2perception.models.backbones.raw_vit import RawViT

        model = RawViT(
            img_size=(224, 224),
            patch_size=16,
            embed_dim=384,
            depth=4,
            num_heads=6,
            use_rope_2d=True,
        )
        x = torch.randn(2, 1, 224, 224)
        cls_token, hidden_states = model(x)
        assert cls_token.shape == (2, 384)

    def test_with_directional(self):
        from photon2perception.models.backbones.raw_vit import RawViT

        model = RawViT(
            img_size=(224, 224),
            patch_size=16,
            embed_dim=384,
            depth=4,
            num_heads=6,
            use_directional=True,
        )
        x = torch.randn(1, 1, 224, 224)
        cls_token, _ = model(x)
        assert cls_token.shape == (1, 384)

    def test_with_sparse_routing(self):
        from photon2perception.models.backbones.raw_vit import RawViT

        model = RawViT(
            img_size=(224, 224),
            patch_size=16,
            embed_dim=384,
            depth=4,
            num_heads=6,
            use_sparse_routing=True,
            router_type='saliency',
            keep_ratio=0.7,
        )
        model.train()
        x = torch.randn(2, 1, 224, 224)
        cls_token, _ = model(x)
        assert cls_token.shape == (2, 384)

    def test_get_num_tokens(self):
        from photon2perception.models.backbones.raw_vit import RawViT

        model = RawViT(img_size=(224, 224), patch_size=16, embed_dim=384, depth=4, num_heads=6)
        assert model.get_num_tokens() == 196  # (224/16)^2


class TestRawViTWithoutOptional:
    """Test RawViT without optional components."""

    def test_minimal_config(self):
        from photon2perception.models.backbones.raw_vit import RawViT

        model = RawViT(
            img_size=(224, 224),
            patch_size=16,
            embed_dim=384,
            depth=4,
            num_heads=6,
            use_rope_2d=False,
            use_directional=False,
            use_sparse_routing=False,
        )
        x = torch.randn(2, 1, 224, 224)
        cls_token, hidden_states = model(x)
        assert cls_token.shape == (2, 384)
        assert len(hidden_states) == 4


# ----- Sanity Check Tests -----

class TestSanityCheck:
    """End-to-end sanity checks."""

    def test_overfit_tiny_batch(self):
        """Verify model can overfit a tiny batch (gradient flow check)."""
        from photon2perception.models.backbones.raw_vit import RawViT
        import torch.optim as optim

        model = RawViT(
            img_size=(64, 64),
            patch_size=8,  # 8x8 grid
            embed_dim=128,
            depth=2,
            num_heads=4,
        )
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        # Tiny batch
        x = torch.randn(4, 1, 64, 64)
        target = torch.randn(4, 128)

        model.train()
        for _ in range(10):
            optimizer.zero_grad()
            cls_token, _ = model(x)
            loss = torch.nn.functional.mse_loss(cls_token, target)
            loss.backward()
            optimizer.step()

        # Loss should decrease
        assert loss.item() < 5.0  # Should be learning


if __name__ == '__main__':
    # Run tests manually
    pytest.main([__file__, '-v', '--tb=short'])
