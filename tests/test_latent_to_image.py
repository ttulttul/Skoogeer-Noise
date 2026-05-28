import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.latent_to_image import ImageBatchToLatent, ImageToBatch, LatentToBatch, LatentToImage  # noqa: E402


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


def test_image_to_batch_inserts_image_at_middle_index():
    batch = torch.tensor(
        [
            [[[1.0, 1.0, 1.0]]],
            [[[3.0, 3.0, 3.0]]],
        ],
        dtype=torch.float32,
    )
    image = torch.tensor([[[[2.0, 2.0, 2.0]]]], dtype=torch.float32)

    node = ImageToBatch()
    (output,) = node.insert(image_batch=batch, image=image, index=1, mode="insert")

    assert output.shape == (3, 1, 1, 3)
    assert torch.equal(output[:, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_image_to_batch_can_prepend_append_and_insert_image_batches():
    batch = torch.arange(2 * 1 * 1 * 3, dtype=torch.float32).reshape(2, 1, 1, 3)
    images = torch.full((2, 1, 1, 3), 99.0, dtype=torch.float32)

    node = ImageToBatch()
    (prepended,) = node.insert(image_batch=batch, image=images[:1], index=0, mode="insert")
    (appended,) = node.insert(image_batch=batch, image=images[:1], index=2, mode="insert")
    (multi_inserted,) = node.insert(image_batch=batch, image=images, index=1, mode="insert")

    assert torch.equal(prepended[0], images[0])
    assert torch.equal(prepended[1:], batch)
    assert torch.equal(appended[:2], batch)
    assert torch.equal(appended[2], images[0])
    assert multi_inserted.shape == (4, 1, 1, 3)
    assert torch.equal(multi_inserted[0], batch[0])
    assert torch.equal(multi_inserted[1:3], images)
    assert torch.equal(multi_inserted[3], batch[1])


def test_image_to_batch_rejects_out_of_range_index():
    batch = torch.zeros((2, 1, 1, 3), dtype=torch.float32)
    image = torch.ones((1, 1, 1, 3), dtype=torch.float32)

    node = ImageToBatch()

    with pytest.raises(ValueError, match="replace index"):
        node.insert(image_batch=batch, image=image, index=2, mode="replace")


def test_image_to_batch_replaces_image_at_index_by_default():
    batch = torch.tensor(
        [
            [[[1.0, 1.0, 1.0]]],
            [[[2.0, 2.0, 2.0]]],
            [[[3.0, 3.0, 3.0]]],
        ],
        dtype=torch.float32,
    )
    image = torch.tensor([[[[9.0, 9.0, 9.0]]]], dtype=torch.float32)

    node = ImageToBatch()
    (output,) = node.insert(image_batch=batch, image=image, index=1)

    assert output.shape == batch.shape
    assert torch.equal(output[:, 0, 0, 0], torch.tensor([1.0, 9.0, 3.0]))


def test_image_to_batch_replaces_multiple_items():
    batch = torch.arange(4 * 1 * 1 * 3, dtype=torch.float32).reshape(4, 1, 1, 3)
    images = torch.full((2, 1, 1, 3), 99.0, dtype=torch.float32)

    node = ImageToBatch()
    (output,) = node.insert(image_batch=batch, image=images, index=1, mode="replace")

    assert output.shape == batch.shape
    assert torch.equal(output[0], batch[0])
    assert torch.equal(output[1:3], images)
    assert torch.equal(output[3], batch[3])


def test_latent_to_batch_inserts_latent_samples_and_preserves_metadata():
    latent_batch = {
        "samples": torch.tensor(
            [
                [[[1.0]]],
                [[[3.0]]],
            ],
            dtype=torch.float32,
        ),
        "noise_mask": torch.tensor([[[0.1]], [[0.3]]], dtype=torch.float32),
        "batch_index": [10, 30],
        "extra": "preserved",
    }
    latent = {
        "samples": torch.tensor([[[[2.0]]]], dtype=torch.float32),
        "noise_mask": torch.tensor([[[0.2]]], dtype=torch.float32),
        "batch_index": [20],
    }

    node = LatentToBatch()
    (output,) = node.insert(latent_batch=latent_batch, latent=latent, index=1, mode="insert")

    assert output["extra"] == "preserved"
    assert output["batch_index"] == [10, 20, 30]
    assert torch.equal(output["samples"][:, 0, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(output["noise_mask"][:, 0, 0], torch.tensor([0.1, 0.2, 0.3]))


def test_latent_to_batch_expands_singleton_noise_mask_for_inserted_batch():
    latent_batch = {
        "samples": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
        "noise_mask": torch.zeros((2, 1, 1), dtype=torch.float32),
    }
    latent = {
        "samples": torch.ones((2, 1, 1, 1), dtype=torch.float32),
        "noise_mask": torch.ones((1, 1, 1), dtype=torch.float32),
    }

    node = LatentToBatch()
    (output,) = node.insert(latent_batch=latent_batch, latent=latent, index=1, mode="insert")

    assert output["samples"].shape == (4, 1, 1, 1)
    assert torch.equal(output["samples"][:, 0, 0, 0], torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert torch.equal(output["noise_mask"][:, 0, 0], torch.tensor([0.0, 1.0, 1.0, 0.0]))


def test_latent_to_batch_replaces_latent_samples_and_metadata_by_default():
    latent_batch = {
        "samples": torch.tensor(
            [
                [[[1.0]]],
                [[[2.0]]],
                [[[3.0]]],
            ],
            dtype=torch.float32,
        ),
        "noise_mask": torch.tensor([[[0.1]], [[0.2]], [[0.3]]], dtype=torch.float32),
        "batch_index": [10, 20, 30],
    }
    latent = {
        "samples": torch.tensor([[[[9.0]]]], dtype=torch.float32),
        "noise_mask": torch.tensor([[[0.9]]], dtype=torch.float32),
        "batch_index": [90],
    }

    node = LatentToBatch()
    (output,) = node.insert(latent_batch=latent_batch, latent=latent, index=1)

    assert output["samples"].shape == latent_batch["samples"].shape
    assert output["batch_index"] == [10, 90, 30]
    assert torch.equal(output["samples"][:, 0, 0, 0], torch.tensor([1.0, 9.0, 3.0]))
    assert torch.allclose(output["noise_mask"][:, 0, 0], torch.tensor([0.1, 0.9, 0.3]))


def test_latent_to_batch_rejects_replace_that_would_extend_past_batch_end():
    latent_batch = {"samples": torch.zeros((2, 1, 1, 1), dtype=torch.float32)}
    latent = {"samples": torch.ones((2, 1, 1, 1), dtype=torch.float32)}

    node = LatentToBatch()

    with pytest.raises(ValueError, match="cannot fit"):
        node.insert(latent_batch=latent_batch, latent=latent, index=1, mode="replace")
