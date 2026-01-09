import pathlib
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_mesh_drag import (  # noqa: E402
    ImageMeshDrag,
    LatentMeshDrag,
    _make_control_displacement,
    mesh_drag_warp,
    mesh_drag_warp_image,
)
from src.masking import prepare_mask_nchw  # noqa: E402


def test_mesh_drag_warp_noop_when_disabled():
    tensor = torch.randn(2, 4, 16, 16)

    out_points_zero = mesh_drag_warp(tensor, points=0, drag_min=0.0, drag_max=4.0, seed=123)
    assert torch.equal(out_points_zero, tensor)

    out_range_zero = mesh_drag_warp(tensor, points=8, drag_min=0.0, drag_max=0.0, seed=123)
    assert torch.equal(out_range_zero, tensor)


def test_mesh_drag_warp_deterministic_for_seed():
    tensor = torch.randn(1, 4, 32, 32)
    out1 = mesh_drag_warp(tensor, points=12, drag_min=1.0, drag_max=4.0, seed=999)
    out2 = mesh_drag_warp(tensor, points=12, drag_min=1.0, drag_max=4.0, seed=999)
    assert torch.allclose(out1, out2)


def test_mesh_drag_warp_deterministic_for_seed_with_brush_strokes():
    tensor = torch.randn(1, 4, 32, 32)
    out1 = mesh_drag_warp(
        tensor,
        points=12,
        drag_min=1.0,
        drag_max=4.0,
        seed=999,
        direction=90.0,
        stroke_width=8.0,
    )
    out2 = mesh_drag_warp(
        tensor,
        points=12,
        drag_min=1.0,
        drag_max=4.0,
        seed=999,
        direction=90.0,
        stroke_width=8.0,
    )
    assert torch.allclose(out1, out2)


def test_mesh_drag_control_displacement_direction_up_enforced():
    disp = _make_control_displacement(
        grid_h=8,
        grid_w=8,
        points=12,
        drag_min=1.0,
        drag_max=3.0,
        seed=123,
        direction=0.0,
    )

    mask = disp.abs().sum(dim=0) > 0
    dx = disp[0][mask]
    dy = disp[1][mask]

    assert torch.max(torch.abs(dx)).item() < 1e-5
    assert torch.all(dy < 0)


def test_mesh_drag_control_displacement_direction_right_enforced():
    disp = _make_control_displacement(
        grid_h=8,
        grid_w=8,
        points=12,
        drag_min=1.0,
        drag_max=3.0,
        seed=123,
        direction=90.0,
    )

    mask = disp.abs().sum(dim=0) > 0
    dx = disp[0][mask]
    dy = disp[1][mask]

    assert torch.max(torch.abs(dy)).item() < 1e-5
    assert torch.all(dx > 0)


def test_mesh_drag_warp_stroke_width_narrows_affected_area():
    base = torch.arange(0, 64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64)
    tensor = base.repeat(1, 4, 1, 1)

    out_narrow = mesh_drag_warp(
        tensor,
        points=64,
        drag_min=1.0,
        drag_max=6.0,
        seed=42,
        direction=90.0,
        stroke_width=6.0,
    )
    out_wide = mesh_drag_warp(
        tensor,
        points=64,
        drag_min=1.0,
        drag_max=6.0,
        seed=42,
        direction=90.0,
        stroke_width=30.0,
    )

    changed_narrow = (out_narrow - tensor).abs().amax(dim=1) > 1e-6
    changed_wide = (out_wide - tensor).abs().amax(dim=1) > 1e-6

    count_narrow = int(changed_narrow.sum().item())
    count_wide = int(changed_wide.sum().item())

    assert count_narrow > 0
    assert count_wide > count_narrow


def test_mesh_drag_warp_stroke_width_scales_with_point_count():
    base = torch.arange(0, 64 * 64, dtype=torch.float32).reshape(1, 1, 64, 64)
    tensor = base.repeat(1, 4, 1, 1)

    out_single = mesh_drag_warp(
        tensor,
        points=1,
        drag_min=1.0,
        drag_max=6.0,
        seed=42,
        direction=90.0,
        stroke_width=6.0,
    )
    out_many = mesh_drag_warp(
        tensor,
        points=24,
        drag_min=1.0,
        drag_max=6.0,
        seed=42,
        direction=90.0,
        stroke_width=6.0,
    )

    changed_single = (out_single - tensor).abs().amax(dim=1) > 1e-6
    changed_many = (out_many - tensor).abs().amax(dim=1) > 1e-6

    rows_single = int(changed_single[0].any(dim=1).sum().item())
    rows_many = int(changed_many[0].any(dim=1).sum().item())

    assert rows_single > 0
    assert rows_many > rows_single


def test_mesh_drag_warp_supports_bspline_interpolation():
    base = torch.linspace(0.0, 1.0, 48 * 48, dtype=torch.float32).reshape(1, 1, 48, 48)
    tensor = base.repeat(1, 4, 1, 1)

    out_bicubic = mesh_drag_warp(
        tensor,
        points=12,
        drag_min=1.0,
        drag_max=6.0,
        seed=123,
        displacement_interpolation="bicubic",
        sampling_interpolation="bilinear",
    )
    out_bspline = mesh_drag_warp(
        tensor,
        points=12,
        drag_min=1.0,
        drag_max=6.0,
        seed=123,
        displacement_interpolation="bspline",
        spline_passes=2,
        sampling_interpolation="bilinear",
    )
    assert not torch.allclose(out_bicubic, out_bspline)


def test_mesh_drag_warp_changes_tensor_for_nonzero_drag():
    base = torch.arange(0, 32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
    tensor = base.repeat(1, 4, 1, 1)

    out = mesh_drag_warp(tensor, points=8, drag_min=2.0, drag_max=6.0, seed=42)
    assert not torch.allclose(out, tensor)


def test_node_warps_noise_mask_in_lockstep():
    base = torch.linspace(0.0, 1.0, 16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    samples = base.repeat(1, 4, 1, 1)
    noise_mask = base.squeeze(1)
    latent = {"samples": samples, "noise_mask": noise_mask}

    node = LatentMeshDrag()
    (out_latent,) = node.drag(latent, seed=123, points=10, drag_min=1.0, drag_max=3.0)

    assert isinstance(out_latent, dict)
    assert out_latent["samples"].shape == samples.shape
    assert out_latent["noise_mask"].shape == noise_mask.shape
    assert torch.allclose(out_latent["noise_mask"], out_latent["samples"][:, 0])


def test_latent_mesh_drag_mask_limits_warp_region_and_respects_noise_mask():
    base = torch.linspace(0.0, 1.0, 16 * 16, dtype=torch.float32).reshape(1, 1, 16, 16)
    samples = base.repeat(1, 4, 1, 1)
    noise_mask = base.squeeze(1)
    latent = {"samples": samples, "noise_mask": noise_mask}

    mask = torch.zeros((1, 16, 16), dtype=torch.float32)
    mask[:, :, :8] = 1.0

    node = LatentMeshDrag()
    (out_latent,) = node.drag(latent, seed=123, points=10, drag_min=1.0, drag_max=3.0, mask=mask)

    expected_warped = mesh_drag_warp(samples, points=10, drag_min=1.0, drag_max=3.0, seed=123)
    expected_samples = samples * (1.0 - mask.unsqueeze(1)) + expected_warped * mask.unsqueeze(1)
    assert torch.allclose(out_latent["samples"], expected_samples)

    expected_noise_warped = mesh_drag_warp(noise_mask, points=10, drag_min=1.0, drag_max=3.0, seed=123)
    expected_noise = noise_mask * (1.0 - mask) + expected_noise_warped * mask
    assert torch.allclose(out_latent["noise_mask"], expected_noise)
    assert torch.allclose(out_latent["noise_mask"], out_latent["samples"][:, 0])


def test_mask_downscales_with_bicubic_interpolation():
    torch.manual_seed(0)
    mask = torch.rand((1, 64, 64), dtype=torch.float32)
    out = prepare_mask_nchw(mask, batch_size=1, height=16, width=16, device=torch.device("cpu"))

    expected = F.interpolate(mask.unsqueeze(1), size=(16, 16), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    assert torch.allclose(out, expected)


def test_mesh_drag_warp_repeats_per_batch_across_extra_dims():
    base_frame = torch.randn(1, 2, 1, 16, 16).repeat(1, 1, 3, 1, 1)
    out = mesh_drag_warp(base_frame, points=6, drag_min=0.5, drag_max=2.0, seed=7)
    assert torch.allclose(out[:, :, 0], out[:, :, 1])
    assert torch.allclose(out[:, :, 1], out[:, :, 2])


def test_mesh_drag_warp_image_noop_when_disabled():
    image = torch.rand(1, 32, 32, 3)
    out_points_zero = mesh_drag_warp_image(image, points=0, drag_min=0.0, drag_max=32.0, seed=1)
    assert torch.equal(out_points_zero, image)

    out_range_zero = mesh_drag_warp_image(image, points=8, drag_min=0.0, drag_max=0.0, seed=1)
    assert torch.equal(out_range_zero, image)


def test_mesh_drag_warp_image_deterministic_for_seed():
    image = torch.linspace(0.0, 1.0, 32 * 32, dtype=torch.float32).reshape(1, 32, 32, 1).repeat(1, 1, 1, 3)
    out1 = mesh_drag_warp_image(image, points=12, drag_min=4.0, drag_max=32.0, seed=999)
    out2 = mesh_drag_warp_image(image, points=12, drag_min=4.0, drag_max=32.0, seed=999)
    assert torch.allclose(out1, out2)


def test_image_mesh_drag_node_preserves_shape_and_changes_content():
    image = torch.linspace(0.0, 1.0, 64 * 64, dtype=torch.float32).reshape(1, 64, 64, 1).repeat(1, 1, 1, 3)
    node = ImageMeshDrag()
    (out_image,) = node.drag(
        image,
        seed=123,
        points=10,
        drag_min=8.0,
        drag_max=48.0,
        displacement_interpolation="bspline",
        spline_passes=2,
        sampling_interpolation="bicubic",
    )
    assert out_image.shape == image.shape
    assert not torch.allclose(out_image, image)
