import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.seeded_noise import ImageNoise, LatentNoise  # noqa: E402


def test_latent_noise_deterministic_for_seed():
    latent = {"samples": torch.ones((1, 4, 8, 8), dtype=torch.float32)}
    node = LatentNoise()
    (out1,) = node.add_noise(latent, seed=123, strength=0.5)
    (out2,) = node.add_noise(latent, seed=123, strength=0.5)
    assert torch.allclose(out1["samples"], out2["samples"])


def test_latent_noise_strength_zero_noop():
    samples = torch.randn((1, 4, 8, 8), dtype=torch.float32)
    latent = {"samples": samples}
    node = LatentNoise()
    (out,) = node.add_noise(latent, seed=123, strength=0.0)
    assert torch.equal(out["samples"], samples)


def test_latent_noise_constant_tensor_still_changes():
    samples = torch.ones((1, 4, 8, 8), dtype=torch.float32)
    latent = {"samples": samples}
    node = LatentNoise()
    (out,) = node.add_noise(latent, seed=42, strength=1.0)
    assert not torch.allclose(out["samples"], samples)


def test_latent_noise_respects_mask():
    samples = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
    latent = {"samples": samples}
    mask = torch.zeros((1, 8, 8), dtype=torch.float32)
    mask[:, :4, :] = 1.0

    node = LatentNoise()
    (full,) = node.add_noise(latent, seed=123, strength=1.0)
    (masked,) = node.add_noise(latent, seed=123, strength=1.0, mask=mask)

    expected = samples * (1.0 - mask.unsqueeze(1)) + full["samples"] * mask.unsqueeze(1)
    assert torch.allclose(masked["samples"], expected)


def test_image_noise_deterministic_for_seed():
    image = torch.linspace(0.0, 1.0, 32 * 32, dtype=torch.float32).reshape(1, 32, 32, 1).repeat(1, 1, 1, 3)
    node = ImageNoise()
    (out1,) = node.add_noise(image, seed=999, strength=0.25)
    (out2,) = node.add_noise(image, seed=999, strength=0.25)
    assert torch.allclose(out1, out2)
    assert out1.shape == image.shape


def test_image_noise_strength_zero_noop():
    image = torch.rand(1, 16, 16, 3, dtype=torch.float32)
    node = ImageNoise()
    (out,) = node.add_noise(image, seed=1, strength=0.0)
    assert torch.equal(out, image)
