import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_frequency_domain import (  # noqa: E402
    CombineLatentPhaseMagnitude,
    FrequencySelectiveStructuredNoise,
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


def test_fss_noise_outputs_latents_and_preserves_keys():
    phase_samples = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    noise_mask = torch.rand(1, 8, 8, dtype=torch.float32)
    phase_latent_ref = {"samples": phase_samples, "noise_mask": noise_mask}

    node = FrequencySelectiveStructuredNoise()
    magnitude_latent_noise, phase_latent_noise = node.generate_fss_noise(
        phase_latent_ref,
        cutoff_radius_r=2,
        sigma=2.0,
        seed=123,
    )

    assert magnitude_latent_noise["samples"].shape == phase_samples.shape
    assert phase_latent_noise["samples"].shape == phase_samples.shape
    assert torch.equal(magnitude_latent_noise["noise_mask"], noise_mask)
    assert torch.equal(phase_latent_noise["noise_mask"], noise_mask)


def test_fss_noise_is_deterministic_for_seed():
    phase_samples = torch.randn(2, 4, 16, 16, dtype=torch.float32)
    phase_latent_ref = {"samples": phase_samples}

    node = FrequencySelectiveStructuredNoise()
    out1_mag, out1_phase = node.generate_fss_noise(phase_latent_ref, cutoff_radius_r=4, sigma=2.0, seed=0)
    out2_mag, out2_phase = node.generate_fss_noise(phase_latent_ref, cutoff_radius_r=4, sigma=2.0, seed=0)

    assert torch.equal(out1_mag["samples"], out2_mag["samples"])
    assert torch.equal(out1_phase["samples"], out2_phase["samples"])


def test_fss_noise_magnitude_is_independent_of_reference_phase_values():
    phase_a = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    phase_b = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    latent_a = {"samples": phase_a}
    latent_b = {"samples": phase_b}

    node = FrequencySelectiveStructuredNoise()
    mag_a, _ = node.generate_fss_noise(latent_a, cutoff_radius_r=3, sigma=2.0, seed=42)
    mag_b, _ = node.generate_fss_noise(latent_b, cutoff_radius_r=3, sigma=2.0, seed=42)

    assert torch.equal(mag_a["samples"], mag_b["samples"])


def test_fss_noise_preserves_reference_phase_when_cutoff_covers_all_frequencies():
    phase_samples = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    phase_latent_ref = {"samples": phase_samples}

    node = FrequencySelectiveStructuredNoise()
    _, phase_latent_noise = node.generate_fss_noise(
        phase_latent_ref,
        cutoff_radius_r=100,
        sigma=2.0,
        seed=0,
    )

    assert torch.equal(phase_latent_noise["samples"], phase_samples)
