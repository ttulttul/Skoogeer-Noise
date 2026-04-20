import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.seeded_noise import (  # noqa: E402
    ImageNoise,
    LatentNoise,
    NextSeeds,
    generate_seed,
    next_seed_output_lists,
    next_seed_values,
)


def test_latent_noise_deterministic_for_seed():
    latent = {"samples": torch.ones((1, 4, 8, 8), dtype=torch.float32)}
    node = LatentNoise()
    (out1,) = node.add_noise(latent, seed=123, strength=0.5)
    (out2,) = node.add_noise(latent, seed=123, strength=0.5)
    assert torch.allclose(out1["samples"], out2["samples"])


def test_latent_noise_batch_seed_offsets():
    base = torch.linspace(0.0, 1.0, 4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8)
    latent = {"samples": base.repeat(2, 1, 1, 1)}
    node = LatentNoise()

    (batched,) = node.add_noise(latent, seed=10, strength=0.5)
    (single,) = node.add_noise({"samples": base.clone()}, seed=11, strength=0.5)

    assert torch.allclose(batched["samples"][1], single["samples"][0])


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


def test_image_noise_batch_seed_offsets():
    base = torch.linspace(0.0, 1.0, 16 * 16 * 3, dtype=torch.float32).reshape(1, 16, 16, 3)
    image = base.repeat(2, 1, 1, 1)
    node = ImageNoise()

    (batched,) = node.add_noise(image, seed=3, strength=0.4)
    (single,) = node.add_noise(base.clone(), seed=4, strength=0.4)

    assert torch.allclose(batched[1], single[0])


def test_image_noise_strength_zero_noop():
    image = torch.rand(1, 16, 16, 3, dtype=torch.float32)
    node = ImageNoise()
    (out,) = node.add_noise(image, seed=1, strength=0.0)
    assert torch.equal(out, image)


def test_image_noise_respects_mask():
    image = torch.zeros(1, 16, 16, 3, dtype=torch.float32)
    mask = torch.zeros((1, 16, 16), dtype=torch.float32)
    mask[:, :8, :] = 1.0

    node = ImageNoise()
    (full,) = node.add_noise(image, seed=42, strength=0.5)
    (masked,) = node.add_noise(image, seed=42, strength=0.5, mask=mask)

    expected = image * (1.0 - mask.unsqueeze(-1)) + full * mask.unsqueeze(-1)
    assert torch.allclose(masked, expected)


def test_generate_seed_matches_expected_mixer_output():
    assert generate_seed(1) == 0xB456BCFC34C2CB2C
    assert generate_seed(0xFFFFFFFFFFFFFFFF) == 0x64B5720B4B825F21


def test_next_seed_values_expand_zero_seed_into_distinct_nonzero_values():
    seeds = next_seed_values(0, count=4)

    assert len(seeds) == 4
    assert len(set(seeds)) == 4
    assert all(0 <= seed <= 0xFFFFFFFFFFFFFFFF for seed in seeds)
    assert all(seed != 0 for seed in seeds)


def test_next_seed_values_wrap_input_at_64_bits():
    wrapped = next_seed_values(0xFFFFFFFFFFFFFFFF, count=4)
    masked = next_seed_values(-1, count=4)

    assert wrapped == masked


def test_next_seed_output_lists_preserve_original_fanout_at_count_one():
    grouped = next_seed_output_lists(123456789, count_per_output=1)

    assert grouped == (
        (0x781E82BBFF6E8F64,),
        (0x780CE7A5D373EB6F,),
        (0x83E08B7E6BEBC732,),
        (0x026450D10CF97BDE,),
    )


def test_next_seed_output_lists_extend_each_output_stream():
    grouped = next_seed_output_lists(123456789, count_per_output=3)

    assert grouped == (
        (
            0x781E82BBFF6E8F64,
            0xDFF0B61891E95BE4,
            0x42B9B6A561E48821,
        ),
        (
            0x780CE7A5D373EB6F,
            0xA735D784BAFF3551,
            0x6782D1C819E6708A,
        ),
        (
            0x83E08B7E6BEBC732,
            0x36D7E9014DB7EFA0,
            0x5FB45D7F824D87BD,
        ),
        (
            0x026450D10CF97BDE,
            0xE105735CD01953A0,
            0x0F65D13E1C77D0AF,
        ),
    )


def test_next_seeds_node_returns_expected_seed_fanout():
    node = NextSeeds()

    outputs = node.next_seeds(123456789, count=1)

    assert outputs == (
        [0x781E82BBFF6E8F64],
        [0x780CE7A5D373EB6F],
        [0x83E08B7E6BEBC732],
        [0x026450D10CF97BDE],
    )
    assert node.OUTPUT_IS_LIST == (True, True, True, True)


def test_next_seeds_node_returns_list_outputs_when_count_is_greater_than_one():
    node = NextSeeds()

    outputs = node.next_seeds(123456789, count=3)

    assert outputs == (
        [0x781E82BBFF6E8F64, 0xDFF0B61891E95BE4, 0x42B9B6A561E48821],
        [0x780CE7A5D373EB6F, 0xA735D784BAFF3551, 0x6782D1C819E6708A],
        [0x83E08B7E6BEBC732, 0x36D7E9014DB7EFA0, 0x5FB45D7F824D87BD],
        [0x026450D10CF97BDE, 0xE105735CD01953A0, 0x0F65D13E1C77D0AF],
    )
