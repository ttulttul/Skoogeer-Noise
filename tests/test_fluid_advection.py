import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fluid_advection import (  # noqa: E402
    FluidLatentAdvection,
    ImageSmokeSimulation,
    LatentSmokeSimulation,
)


def test_fluid_latent_advection_noop_when_steps_zero():
    samples = torch.randn(2, 4, 16, 16)
    latent = {"samples": samples}

    node = FluidLatentAdvection()
    (out_latent, preview) = node.run(
        latent,
        steps=0,
        dt=1.0,
        resolution_scale=0.5,
        force_count=3,
        force_strength=5.0,
        force_radius=0.1,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.0,
        seed=123,
        wrap_mode="clamp",
    )

    assert torch.equal(out_latent["samples"], samples)
    assert preview.shape == (2, 16, 16, 3)
    assert torch.allclose(preview, torch.zeros_like(preview))


def test_fluid_latent_advection_deterministic_for_seed():
    torch.manual_seed(0)
    samples = torch.randn(1, 4, 32, 32)
    latent = {"samples": samples}

    node = FluidLatentAdvection()
    out1, _ = node.run(
        latent,
        steps=10,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=4.0,
        force_radius=0.2,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.05,
        vorticity=0.5,
        seed=999,
        wrap_mode="wrap",
    )
    out2, _ = node.run(
        latent,
        steps=10,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=4.0,
        force_radius=0.2,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.05,
        vorticity=0.5,
        seed=999,
        wrap_mode="wrap",
    )

    assert torch.allclose(out1["samples"], out2["samples"])


def test_fluid_latent_advection_warps_noise_mask_in_lockstep():
    base = torch.linspace(0.0, 1.0, 32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
    samples = base.repeat(1, 4, 1, 1)
    noise_mask = base.squeeze(1)
    latent = {"samples": samples, "noise_mask": noise_mask}

    node = FluidLatentAdvection()
    out_latent, preview = node.run(
        latent,
        steps=8,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=3.0,
        force_radius=0.25,
        swirl_strength=1.5,
        velocity_damping=0.99,
        diffusion=0.02,
        vorticity=0.75,
        seed=42,
        wrap_mode="mirror",
    )

    assert out_latent["samples"].shape == samples.shape
    assert out_latent["noise_mask"].shape == noise_mask.shape
    assert torch.allclose(out_latent["noise_mask"], out_latent["samples"][:, 0], atol=1e-5)
    assert preview.shape == (1, 32, 32, 3)
    assert preview.min().item() >= 0.0
    assert preview.max().item() <= 1.0


def test_fluid_latent_advection_mask_limits_effect_region():
    torch.manual_seed(0)
    samples = torch.randn(1, 4, 24, 24)
    latent = {"samples": samples}

    mask = torch.zeros((1, 24, 24), dtype=torch.float32)
    mask[:, :, :12] = 1.0

    node = FluidLatentAdvection()
    out_latent, _ = node.run(
        latent,
        steps=6,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=4.0,
        force_radius=0.3,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.0,
        seed=123,
        wrap_mode="clamp",
        mask=mask,
    )

    out_samples = out_latent["samples"]
    assert out_samples.shape == samples.shape
    outside = (mask.unsqueeze(1) < 0.5).expand_as(out_samples)
    assert torch.allclose(out_samples[outside], samples[outside], atol=1e-6)
    assert not torch.allclose(out_samples, samples)


def test_smoke_latent_simulation_deterministic_for_seed():
    torch.manual_seed(0)
    samples = torch.randn(1, 4, 24, 24)
    latent = {"samples": samples}

    node = LatentSmokeSimulation()
    out1, _, _ = node.run(
        latent,
        steps=12,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=4.0,
        force_radius=0.2,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.3,
        buoyancy=1.5,
        ambient_updraft=0.1,
        density_fade=0.01,
        temperature_strength=0.0,
        cooling_rate=0.05,
        smoke_source_strength=1.0,
        smoke_source_radius=0.1,
        smoke_source_mode="random",
        seed=999,
        wrap_mode="wrap",
    )
    out2, _, _ = node.run(
        latent,
        steps=12,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=4.0,
        force_radius=0.2,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.3,
        buoyancy=1.5,
        ambient_updraft=0.1,
        density_fade=0.01,
        temperature_strength=0.0,
        cooling_rate=0.05,
        smoke_source_strength=1.0,
        smoke_source_radius=0.1,
        smoke_source_mode="random",
        seed=999,
        wrap_mode="wrap",
    )
    assert torch.allclose(out1["samples"], out2["samples"])


def test_smoke_latent_simulation_warps_noise_mask_in_lockstep():
    base = torch.linspace(0.0, 1.0, 20 * 20, dtype=torch.float32).reshape(1, 1, 20, 20)
    samples = base.repeat(1, 4, 1, 1)
    noise_mask = base.squeeze(1)
    latent = {"samples": samples, "noise_mask": noise_mask}

    node = LatentSmokeSimulation()
    out_latent, density_preview, velocity_preview = node.run(
        latent,
        steps=8,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=4.0,
        force_radius=0.25,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.6,
        buoyancy=1.5,
        ambient_updraft=0.1,
        density_fade=0.01,
        temperature_strength=0.0,
        cooling_rate=0.05,
        smoke_source_strength=1.0,
        smoke_source_radius=0.1,
        smoke_source_mode="image",
        seed=123,
        wrap_mode="mirror",
    )

    assert out_latent["samples"].shape == samples.shape
    assert out_latent["noise_mask"].shape == noise_mask.shape
    assert torch.allclose(out_latent["noise_mask"], out_latent["samples"][:, 0], atol=1e-5)
    assert density_preview.shape == (1, 20, 20, 3)
    assert velocity_preview.shape == (1, 20, 20, 3)


def test_smoke_latent_simulation_requires_mask_for_mask_mode():
    samples = torch.randn(1, 4, 16, 16)
    latent = {"samples": samples}

    node = LatentSmokeSimulation()
    with pytest.raises(ValueError):
        node.run(
            latent,
            steps=4,
            dt=1.0,
            resolution_scale=1.0,
            force_count=3,
            force_strength=4.0,
            force_radius=0.2,
            swirl_strength=2.0,
            velocity_damping=0.98,
            diffusion=0.0,
            vorticity=0.3,
            buoyancy=1.5,
            ambient_updraft=0.1,
            density_fade=0.01,
            temperature_strength=0.0,
            cooling_rate=0.05,
            smoke_source_strength=1.0,
            smoke_source_radius=0.1,
            smoke_source_mode="mask",
            seed=123,
            wrap_mode="clamp",
        )


def test_smoke_image_simulation_preserves_shape_and_is_deterministic():
    image = torch.linspace(0.0, 1.0, 32 * 32, dtype=torch.float32).reshape(1, 32, 32, 1).repeat(1, 1, 1, 3)

    node = ImageSmokeSimulation()
    out1, density1, vel1 = node.run(
        image,
        steps=10,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=6.0,
        force_radius=0.2,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.3,
        buoyancy=1.5,
        ambient_updraft=0.1,
        density_fade=0.01,
        temperature_strength=0.0,
        cooling_rate=0.05,
        smoke_source_strength=1.0,
        smoke_source_radius=0.1,
        smoke_source_mode="image",
        seed=7,
        wrap_mode="wrap",
    )
    out2, density2, vel2 = node.run(
        image,
        steps=10,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=6.0,
        force_radius=0.2,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.3,
        buoyancy=1.5,
        ambient_updraft=0.1,
        density_fade=0.01,
        temperature_strength=0.0,
        cooling_rate=0.05,
        smoke_source_strength=1.0,
        smoke_source_radius=0.1,
        smoke_source_mode="image",
        seed=7,
        wrap_mode="wrap",
    )

    assert out1.shape == image.shape
    assert density1.shape == (1, 32, 32, 3)
    assert vel1.shape == (1, 32, 32, 3)
    assert torch.allclose(out1, out2)
    assert torch.allclose(density1, density2)
    assert torch.allclose(vel1, vel2)


def test_smoke_image_simulation_can_return_step_batch():
    image = (
        torch.linspace(0.0, 1.0, 2 * 16 * 16, dtype=torch.float32)
        .reshape(2, 16, 16, 1)
        .repeat(1, 1, 1, 3)
    )
    steps = 4

    node = ImageSmokeSimulation()
    out_batch, density_batch, vel_batch = node.run(
        image,
        steps=steps,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=6.0,
        force_radius=0.2,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.3,
        buoyancy=1.5,
        ambient_updraft=0.1,
        density_fade=0.01,
        temperature_strength=0.0,
        cooling_rate=0.05,
        smoke_source_strength=1.0,
        smoke_source_radius=0.1,
        smoke_source_mode="image",
        seed=7,
        wrap_mode="wrap",
        output_mode="batch",
    )
    out_final, density_final, vel_final = node.run(
        image,
        steps=steps,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=6.0,
        force_radius=0.2,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.3,
        buoyancy=1.5,
        ambient_updraft=0.1,
        density_fade=0.01,
        temperature_strength=0.0,
        cooling_rate=0.05,
        smoke_source_strength=1.0,
        smoke_source_radius=0.1,
        smoke_source_mode="image",
        seed=7,
        wrap_mode="wrap",
    )

    assert out_batch.shape == (int(image.shape[0]) * steps, 16, 16, 3)
    assert density_batch.shape == (int(image.shape[0]) * steps, 16, 16, 3)
    assert vel_batch.shape == (int(image.shape[0]) * steps, 16, 16, 3)
    assert torch.allclose(out_batch[-int(image.shape[0]):], out_final)
    assert torch.allclose(density_batch[-int(image.shape[0]):], density_final)
    assert torch.allclose(vel_batch[-int(image.shape[0]):], vel_final)


def test_smoke_latent_simulation_can_return_step_batch():
    base = torch.linspace(0.0, 1.0, 2 * 16 * 16, dtype=torch.float32).reshape(2, 1, 16, 16)
    samples = base.repeat(1, 4, 1, 1)
    noise_mask = base.squeeze(1)
    latent = {"samples": samples, "noise_mask": noise_mask}
    steps = 5

    node = LatentSmokeSimulation()
    out_batch, density_batch, vel_batch = node.run(
        latent,
        steps=steps,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=4.0,
        force_radius=0.25,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.6,
        buoyancy=1.5,
        ambient_updraft=0.1,
        density_fade=0.01,
        temperature_strength=0.0,
        cooling_rate=0.05,
        smoke_source_strength=1.0,
        smoke_source_radius=0.1,
        smoke_source_mode="image",
        seed=123,
        wrap_mode="mirror",
        output_mode="batch",
    )
    out_final, density_final, vel_final = node.run(
        latent,
        steps=steps,
        dt=1.0,
        resolution_scale=1.0,
        force_count=3,
        force_strength=4.0,
        force_radius=0.25,
        swirl_strength=2.0,
        velocity_damping=0.98,
        diffusion=0.0,
        vorticity=0.6,
        buoyancy=1.5,
        ambient_updraft=0.1,
        density_fade=0.01,
        temperature_strength=0.0,
        cooling_rate=0.05,
        smoke_source_strength=1.0,
        smoke_source_radius=0.1,
        smoke_source_mode="image",
        seed=123,
        wrap_mode="mirror",
    )

    assert out_batch["samples"].shape == (int(samples.shape[0]) * steps, 4, 16, 16)
    assert out_batch["noise_mask"].shape == (int(samples.shape[0]) * steps, 16, 16)
    assert density_batch.shape == (int(samples.shape[0]) * steps, 16, 16, 3)
    assert vel_batch.shape == (int(samples.shape[0]) * steps, 16, 16, 3)
    assert torch.allclose(out_batch["samples"][-int(samples.shape[0]):], out_final["samples"])
    assert torch.allclose(out_batch["noise_mask"][-int(samples.shape[0]):], out_final["noise_mask"])
    assert torch.allclose(density_batch[-int(samples.shape[0]):], density_final)
    assert torch.allclose(vel_batch[-int(samples.shape[0]):], vel_final)
    assert torch.allclose(out_batch["noise_mask"], out_batch["samples"][:, 0], atol=1e-5)
