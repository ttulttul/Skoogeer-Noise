import pathlib
import sys

import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_to_image import ImageBatchToLatent, LatentToImage  # noqa: E402


def test_latent_to_image_outputs_channel_batch_bhwc():
    samples = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    latent = {"samples": samples}

    node = LatentToImage()
    (images,) = node.render(latent=latent, normalize=False, output_channels="1")

    assert images.shape == (6, 2, 2, 1)
    assert torch.equal(images[0, :, :, 0], samples[0, 0])
    assert torch.equal(images[1, :, :, 0], samples[0, 1])
    assert torch.equal(images[-1, :, :, 0], samples[1, 2])


def test_latent_to_image_normalize_minmax_rgb():
    samples = torch.tensor([[[[0.0, 1.0], [2.0, 4.0]]]], dtype=torch.float32)
    latent = {"samples": samples}

    node = LatentToImage()
    (images,) = node.render(latent=latent, normalize=True, output_channels="3")

    expected_gray = torch.tensor([[[0.0, 0.25], [0.5, 1.0]]], dtype=torch.float32)
    assert images.shape == (1, 2, 2, 3)
    assert torch.allclose(images[0, :, :, 0], expected_gray[0], atol=1e-6)
    assert torch.allclose(images[0, :, :, 1], expected_gray[0], atol=1e-6)
    assert torch.allclose(images[0, :, :, 2], expected_gray[0], atol=1e-6)
    assert torch.all(images >= 0.0)
    assert torch.all(images <= 1.0)


def test_latent_to_image_flattens_extra_dims():
    samples = torch.arange(1 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 2, 3, 2, 2)
    latent = {"samples": samples}

    node = LatentToImage()
    (images,) = node.render(latent=latent, normalize=False, output_channels="1")

    assert images.shape == (6, 2, 2, 1)
    assert torch.equal(images[0, :, :, 0], samples[0, 0, 0])
    assert torch.equal(images[1, :, :, 0], samples[0, 1, 0])


def test_image_batch_to_latent_reconstructs_single_channel_batch():
    samples = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    latent = {"samples": samples}

    to_image = LatentToImage()
    (images,) = to_image.render(latent=latent, normalize=False, output_channels="1")

    to_latent = ImageBatchToLatent()
    (merged,) = to_latent.merge(image_batch=images, batch_size=2, channels=3, channel_source="r")

    assert merged["samples"].shape == samples.shape
    assert torch.equal(merged["samples"], samples)


def test_image_batch_to_latent_inferring_channels():
    samples = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    latent = {"samples": samples}

    to_image = LatentToImage()
    (images,) = to_image.render(latent=latent, normalize=False, output_channels="1")

    to_latent = ImageBatchToLatent()
    (merged,) = to_latent.merge(image_batch=images, batch_size=2, channels=0, channel_source="r")

    assert merged["samples"].shape == samples.shape
    assert torch.equal(merged["samples"], samples)


def test_image_batch_to_latent_channel_source_mean():
    images = torch.tensor(
        [[[[1.0, 2.0, 3.0]]], [[[4.0, 5.0, 6.0]]]],
        dtype=torch.float32,
    )

    to_latent = ImageBatchToLatent()
    (merged,) = to_latent.merge(image_batch=images, batch_size=1, channels=2, channel_source="mean")

    expected = torch.tensor([[[[2.0]], [[5.0]]]], dtype=torch.float32)
    assert merged["samples"].shape == (1, 2, 1, 1)
    assert torch.allclose(merged["samples"], expected, atol=1e-6)
