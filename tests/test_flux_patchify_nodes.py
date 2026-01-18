import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qwen_noise_nodes import PatchifyFlux2Latent, UnpatchifyFlux2Latent  # noqa: E402


def test_flux_unpatchify_patchify_roundtrip():
    samples = torch.arange(1 * 128 * 8 * 8, dtype=torch.float32).reshape(1, 128, 8, 8)
    latent = {"samples": samples}

    unpatch_node = UnpatchifyFlux2Latent()
    patch_node = PatchifyFlux2Latent()

    (unpatchified,) = unpatch_node.unpatchify(latent)
    assert unpatchified["samples"].shape == (1, 32, 16, 16)

    (repatchified,) = patch_node.patchify(unpatchified)
    assert torch.equal(repatchified["samples"], samples)


def test_flux_unpatchify_patchify_video_roundtrip():
    samples = torch.arange(1 * 128 * 2 * 8 * 8, dtype=torch.float32).reshape(1, 128, 2, 8, 8)
    latent = {"samples": samples}

    unpatch_node = UnpatchifyFlux2Latent()
    patch_node = PatchifyFlux2Latent()

    (unpatchified,) = unpatch_node.unpatchify(latent)
    assert unpatchified["samples"].shape == (1, 32, 2, 16, 16)

    (repatchified,) = patch_node.patchify(unpatchified)
    assert torch.equal(repatchified["samples"], samples)
