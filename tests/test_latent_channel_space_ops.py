import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_channel_space_ops import (  # noqa: E402
    LatentChannelMatch,
    LatentChannelLinearTransform,
    LatentChannelMerge,
    LatentChannelNonlinearTransform,
    LatentPackedSlotTransform,
)


def test_signed_permutation_preserves_channel_values():
    samples = torch.zeros((1, 6, 2, 2), dtype=torch.float32)
    for ch in range(6):
        samples[0, ch] = float(ch + 1)
    latent = {"samples": samples}

    node = LatentChannelLinearTransform()
    (out1,) = node.transform(
        latent,
        operation="signed_permute",
        seed=123,
        sign_flip_prob=0.0,
        tile_size=0,
        block_size=0,
        alpha=0.5,
        selection_mode="all",
        selection_fraction=1.0,
        selection_count=0,
        selection_order="highest",
        selection_indices="",
        mix=1.0,
        match_stats=False,
    )
    (out2,) = node.transform(
        latent,
        operation="signed_permute",
        seed=123,
        sign_flip_prob=0.0,
        tile_size=0,
        block_size=0,
        alpha=0.5,
        selection_mode="all",
        selection_fraction=1.0,
        selection_count=0,
        selection_order="highest",
        selection_indices="",
        mix=1.0,
        match_stats=False,
    )

    assert torch.equal(out1["samples"], out2["samples"])

    original_means = samples.mean(dim=(0, 2, 3)).tolist()
    out_means = out1["samples"].mean(dim=(0, 2, 3)).tolist()
    assert sorted(out_means) == sorted(original_means)


def test_orthogonal_rotation_preserves_per_pixel_norms():
    torch.manual_seed(0)
    samples = torch.randn(1, 4, 3, 3, dtype=torch.float32)
    latent = {"samples": samples}

    node = LatentChannelLinearTransform()
    (out,) = node.transform(
        latent,
        operation="orthogonal_rotate",
        seed=42,
        sign_flip_prob=0.0,
        tile_size=0,
        block_size=0,
        alpha=0.5,
        selection_mode="all",
        selection_fraction=1.0,
        selection_count=0,
        selection_order="highest",
        selection_indices="",
        mix=1.0,
        match_stats=False,
    )

    original_norm = samples.norm(dim=1)
    rotated_norm = out["samples"].norm(dim=1)
    assert torch.allclose(rotated_norm, original_norm, atol=1e-4, rtol=1e-4)


def test_quantize_respects_selection_indices():
    samples = torch.tensor(
        [[
            [[0.1, 0.6], [1.1, -0.2]],
            [[0.2, 0.7], [1.2, -0.3]],
            [[0.3, 0.8], [1.3, -0.4]],
        ]],
        dtype=torch.float32,
    )
    latent = {"samples": samples.clone()}

    node = LatentChannelNonlinearTransform()
    (out,) = node.transform(
        latent,
        operation="quantize",
        seed=0,
        gate_strength=1.0,
        beta=1.0,
        blur_radius=0,
        quantize_step=0.5,
        clip_threshold=2.0,
        selection_mode="indices",
        selection_fraction=1.0,
        selection_count=0,
        selection_order="highest",
        selection_indices="1",
        mix=1.0,
        match_stats=False,
    )

    assert torch.equal(out["samples"][:, 0], samples[:, 0])
    assert torch.equal(out["samples"][:, 2], samples[:, 2])
    quantized = out["samples"][:, 1] / 0.5
    assert torch.allclose(quantized, torch.round(quantized))


def test_quantize_respects_mask():
    samples = torch.tensor(
        [[
            [[0.1, 0.6], [1.1, -0.2]],
            [[0.2, 0.7], [1.2, -0.3]],
        ]],
        dtype=torch.float32,
    )
    latent = {"samples": samples.clone()}
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)

    node = LatentChannelNonlinearTransform()
    (out,) = node.transform(
        latent,
        operation="quantize",
        seed=0,
        gate_strength=1.0,
        beta=1.0,
        blur_radius=0,
        quantize_step=0.5,
        clip_threshold=2.0,
        selection_mode="all",
        selection_fraction=1.0,
        selection_count=0,
        selection_order="highest",
        selection_indices="",
        mix=1.0,
        match_stats=False,
        mask=mask,
    )

    quantized = torch.round(samples / 0.5) * 0.5
    mask_broadcast = mask.unsqueeze(1)
    expected = samples * (1.0 - mask_broadcast) + quantized * mask_broadcast
    assert torch.allclose(out["samples"], expected)


def test_latent_channel_merge_indices_blend_strength():
    dest_samples = torch.tensor(
        [[
            [[0.0, 0.0], [0.0, 0.0]],
            [[10.0, 10.0], [10.0, 10.0]],
            [[20.0, 20.0], [20.0, 20.0]],
        ]],
        dtype=torch.float32,
    )
    source_samples = torch.tensor(
        [[
            [[1.0, 1.0], [1.0, 1.0]],
            [[2.0, 2.0], [2.0, 2.0]],
            [[3.0, 3.0], [3.0, 3.0]],
        ]],
        dtype=torch.float32,
    )

    node = LatentChannelMerge()
    (out,) = node.merge(
        destination={"samples": dest_samples.clone()},
        source={"samples": source_samples},
        seed=0,
        selection_mode="indices",
        selection_fraction=1.0,
        selection_count=0,
        selection_order="highest",
        selection_indices="1,2",
        blend_strength=0.5,
    )

    expected = dest_samples.clone()
    expected[:, 1] = dest_samples[:, 1] + (source_samples[:, 1] - dest_samples[:, 1]) * 0.5
    expected[:, 2] = dest_samples[:, 2] + (source_samples[:, 2] - dest_samples[:, 2]) * 0.5
    assert torch.allclose(out["samples"], expected)


def test_latent_channel_merge_top_variance_selection_count():
    dest_samples = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    source_samples = torch.tensor(
        [[
            [[1.0, 1.0], [1.0, 1.0]],
            [[0.0, 2.0], [0.0, 2.0]],
        ]],
        dtype=torch.float32,
    )

    node = LatentChannelMerge()
    (out,) = node.merge(
        destination={"samples": dest_samples.clone()},
        source={"samples": source_samples},
        seed=0,
        selection_mode="top_variance",
        selection_fraction=0.0,
        selection_count=1,
        selection_order="highest",
        selection_indices="",
        blend_strength=1.0,
    )

    expected = dest_samples.clone()
    expected[:, 1] = source_samples[:, 1]
    assert torch.allclose(out["samples"], expected)


def test_latent_channel_match_transfers_reference_channel_stats():
    target_samples = torch.tensor(
        [[
            [[0.0, 1.0], [2.0, 3.0]],
            [[10.0, 12.0], [14.0, 16.0]],
        ]],
        dtype=torch.float32,
    )
    reference_samples = torch.tensor(
        [[
            [[100.0, 100.0], [100.0, 100.0]],
            [[0.0, 2.0], [4.0, 6.0]],
        ]],
        dtype=torch.float32,
    )

    node = LatentChannelMatch()
    (out,) = node.match(
        target={"samples": target_samples.clone(), "noise_mask": torch.ones(1, 2, 2)},
        reference={"samples": reference_samples},
        mix=1.0,
    )

    out_means = out["samples"].mean(dim=(0, 2, 3))
    out_stds = out["samples"].std(dim=(0, 2, 3), unbiased=False)
    ref_means = reference_samples.mean(dim=(0, 2, 3))
    ref_stds = reference_samples.std(dim=(0, 2, 3), unbiased=False)

    assert torch.allclose(out_means, ref_means, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_stds, ref_stds, atol=1e-5, rtol=1e-5)
    assert torch.equal(out["noise_mask"], torch.ones(1, 2, 2))


def test_latent_channel_match_broadcasts_single_reference_sample():
    target_samples = torch.tensor(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 5.0], [6.0, 7.0]],
            ],
            [
                [[8.0, 9.0], [10.0, 11.0]],
                [[12.0, 13.0], [14.0, 15.0]],
            ],
        ],
        dtype=torch.float32,
    )
    reference_samples = torch.tensor(
        [[
            [[20.0, 22.0], [24.0, 26.0]],
            [[100.0, 100.0], [100.0, 100.0]],
        ]],
        dtype=torch.float32,
    )

    node = LatentChannelMatch()
    (out,) = node.match(
        target={"samples": target_samples.clone()},
        reference={"samples": reference_samples},
        mix=1.0,
    )

    ref_means = reference_samples.mean(dim=(0, 2, 3))
    ref_stds = reference_samples.std(dim=(0, 2, 3), unbiased=False)
    out_means = out["samples"].mean(dim=(2, 3))
    out_stds = out["samples"].std(dim=(2, 3), unbiased=False)

    assert torch.allclose(out_means[0], ref_means, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_means[1], ref_means, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_stds[0], ref_stds, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_stds[1], ref_stds, atol=1e-5, rtol=1e-5)


def test_latent_channel_match_respects_mix_and_mask():
    target_samples = torch.tensor(
        [[[[0.0, 2.0], [4.0, 6.0]]]],
        dtype=torch.float32,
    )
    reference_samples = torch.tensor(
        [[[[10.0, 10.0], [10.0, 10.0]]]],
        dtype=torch.float32,
    )
    mask = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)

    node = LatentChannelMatch()
    (out,) = node.match(
        target={"samples": target_samples.clone()},
        reference={"samples": reference_samples},
        mix=0.5,
        mask=mask,
    )

    fully_matched = torch.full_like(target_samples, 10.0)
    mixed = target_samples + (fully_matched - target_samples) * 0.5
    expected = target_samples.clone()
    expected[:, :, :, 0] = mixed[:, :, :, 0]

    assert torch.allclose(out["samples"], expected)


def test_packed_slot_rotate_cw_reorders_slots():
    samples = torch.tensor([[[[10.0]], [[20.0]], [[30.0]], [[40.0]]]], dtype=torch.float32)
    latent = {"samples": samples}

    node = LatentPackedSlotTransform()
    (out,) = node.transform(
        latent,
        operation="rotate_cw",
        patch_size=2,
        base_channels=1,
        seed=0,
        mix=1.0,
        match_stats=False,
    )

    expected = torch.tensor([[[[30.0]], [[10.0]], [[40.0]], [[20.0]]]], dtype=torch.float32)
    assert torch.equal(out["samples"], expected)
