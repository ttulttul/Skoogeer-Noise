import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_frequency_domain import (  # noqa: E402
    CombineLatentPhaseMagnitude,
    SplitLatentPhaseMagnitude,
)


def test_split_outputs_magnitude_and_phase_and_preserves_keys():
    samples = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    noise_mask = torch.rand(1, 8, 8, dtype=torch.float32)
    latent = {"samples": samples, "noise_mask": noise_mask}

    node = SplitLatentPhaseMagnitude()
    magnitude_latent, phase_latent = node.split(latent)

    assert magnitude_latent["samples"].shape == samples.shape
    assert phase_latent["samples"].shape == samples.shape
    assert torch.equal(magnitude_latent["noise_mask"], noise_mask)
    assert torch.equal(phase_latent["noise_mask"], noise_mask)


def test_split_then_combine_roundtrips_latent_samples():
    torch.manual_seed(0)
    samples = torch.randn(2, 4, 16, 16, dtype=torch.float32)
    latent = {"samples": samples}

    split_node = SplitLatentPhaseMagnitude()
    combine_node = CombineLatentPhaseMagnitude()

    magnitude_latent, phase_latent = split_node.split(latent)
    (out_latent,) = combine_node.combine(magnitude_latent, phase_latent)

    assert torch.allclose(out_latent["samples"], samples, atol=1e-5, rtol=1e-5)


def test_combine_rejects_mismatched_shapes():
    magnitude_latent = {"samples": torch.randn(1, 4, 8, 8, dtype=torch.float32)}
    phase_latent = {"samples": torch.randn(1, 4, 8, 9, dtype=torch.float32)}

    node = CombineLatentPhaseMagnitude()
    with pytest.raises(ValueError):
        node.combine(magnitude_latent, phase_latent)
