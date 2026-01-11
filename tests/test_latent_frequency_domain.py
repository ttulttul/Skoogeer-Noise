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


def test_split_doubles_channel_count_and_preserves_keys():
    samples = torch.randn(1, 4, 8, 8, dtype=torch.float32)
    noise_mask = torch.rand(1, 8, 8, dtype=torch.float32)
    latent = {"samples": samples, "noise_mask": noise_mask}

    node = SplitLatentPhaseMagnitude()
    (freq_latent,) = node.split(latent)

    assert freq_latent["samples"].shape == (1, 8, 8, 8)
    assert torch.equal(freq_latent["noise_mask"], noise_mask)


def test_split_then_combine_roundtrips_latent_samples():
    torch.manual_seed(0)
    samples = torch.randn(2, 4, 16, 16, dtype=torch.float32)
    latent = {"samples": samples}

    split_node = SplitLatentPhaseMagnitude()
    combine_node = CombineLatentPhaseMagnitude()

    (freq_latent,) = split_node.split(latent)
    (out_latent,) = combine_node.combine(freq_latent)

    assert torch.allclose(out_latent["samples"], samples, atol=1e-5, rtol=1e-5)


def test_combine_rejects_odd_channel_count():
    packed = torch.randn(1, 5, 8, 8, dtype=torch.float32)
    freq_latent = {"samples": packed}

    node = CombineLatentPhaseMagnitude()
    with pytest.raises(ValueError):
        node.combine(freq_latent)

