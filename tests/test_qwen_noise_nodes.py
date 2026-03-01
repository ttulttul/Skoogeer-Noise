import pathlib
import sys

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.qwen_noise_nodes as qnn  # noqa: E402


FLOW_MATCHING_CHANNELS = 16
FLOW_MATCHING_TEMPORAL = 1
FLOW_MATCHING_WIDTH = 8
FLOW_MATCHING_HEIGHT = 8


def make_flow_matching_latent(
    batch_size=1,
    temporal=FLOW_MATCHING_TEMPORAL,
    width=FLOW_MATCHING_WIDTH,
    height=FLOW_MATCHING_HEIGHT,
):
    samples = torch.randn(batch_size, FLOW_MATCHING_CHANNELS, temporal, width, height)
    return {"samples": samples}


def assert_flow_matching_shape(
    tensor,
    batch_size=1,
    temporal=FLOW_MATCHING_TEMPORAL,
    width=FLOW_MATCHING_WIDTH,
    height=FLOW_MATCHING_HEIGHT,
):
    expected = (batch_size, FLOW_MATCHING_CHANNELS, temporal, width, height)
    assert tensor.shape == expected


def make_latent_from_tensor(tensor):
    return {"samples": tensor}


def make_image(batch_size=1, height=32, width=32, channels=3):
    return torch.rand(batch_size, height, width, channels)


def test_latent_gaussian_blur_modifies_values():
    node = qnn.LatentGaussianBlur()
    latent = make_flow_matching_latent()

    (result,) = node.blur_latent(latent, sigma=1.5, blur_mode="Spatial Only")

    original = latent["samples"]
    blurred = result["samples"]

    assert_flow_matching_shape(original)
    assert_flow_matching_shape(blurred)
    assert not torch.allclose(blurred, original)


def test_latent_gaussian_blur_rejects_rank2_tensor():
    node = qnn.LatentGaussianBlur()
    malformed = make_latent_from_tensor(torch.randn(16, 16))

    with pytest.raises((RuntimeError, IndexError)):
        node.blur_latent(malformed, sigma=1.0, blur_mode="Spatial Only")


def test_latent_gaussian_blur_zero_sigma_returns_original_object():
    node = qnn.LatentGaussianBlur()
    latent = make_flow_matching_latent()

    (result,) = node.blur_latent(latent, sigma=0.0, blur_mode="Spatial Only")

    assert result is latent


def test_latent_gaussian_blur_handles_video_latent():
    node = qnn.LatentGaussianBlur()
    video = make_flow_matching_latent(temporal=3, width=4, height=4)

    (result,) = node.blur_latent(video, sigma=1.0, blur_mode="Spatial Only")

    blurred = result["samples"]
    assert_flow_matching_shape(blurred, temporal=3, width=4, height=4)


def test_latent_add_noise_reproducible_with_seed():
    node = qnn.LatentAddNoise()
    latent = make_flow_matching_latent()

    (first,) = node.add_noise(latent, seed=123, strength=0.8)
    (second,) = node.add_noise(latent, seed=123, strength=0.8)
    (third,) = node.add_noise(latent, seed=321, strength=0.8)

    assert_flow_matching_shape(first["samples"])
    assert_flow_matching_shape(second["samples"])
    assert_flow_matching_shape(third["samples"])
    assert torch.allclose(first["samples"], second["samples"])
    assert not torch.allclose(first["samples"], third["samples"])


def test_latent_add_noise_batch_seed_offsets():
    node = qnn.LatentAddNoise()
    base = torch.linspace(
        0.0,
        1.0,
        FLOW_MATCHING_CHANNELS * FLOW_MATCHING_TEMPORAL * FLOW_MATCHING_WIDTH * FLOW_MATCHING_HEIGHT,
        dtype=torch.float32,
    ).reshape(1, FLOW_MATCHING_CHANNELS, FLOW_MATCHING_TEMPORAL, FLOW_MATCHING_WIDTH, FLOW_MATCHING_HEIGHT)
    latent = make_latent_from_tensor(base.repeat(2, 1, 1, 1, 1))

    (batched,) = node.add_noise(latent, seed=101, strength=0.5)
    (single,) = node.add_noise(make_latent_from_tensor(base.clone()), seed=102, strength=0.5)

    assert batched["samples"].shape[0] == 2
    assert torch.allclose(batched["samples"][1], single["samples"][0])


def test_latent_add_noise_respects_mask():
    node = qnn.LatentAddNoise()
    latent = make_flow_matching_latent()

    mask = torch.zeros((1, FLOW_MATCHING_WIDTH, FLOW_MATCHING_HEIGHT), dtype=torch.float32)
    mask[:, : FLOW_MATCHING_WIDTH // 2, :] = 1.0

    (full,) = node.add_noise(latent, seed=123, strength=0.8)
    (masked,) = node.add_noise(latent, seed=123, strength=0.8, mask=mask)

    mask_view = mask.unsqueeze(1).unsqueeze(2)
    expected = latent["samples"] * (1.0 - mask_view) + full["samples"] * mask_view
    assert torch.allclose(masked["samples"], expected)


def test_latent_add_noise_negative_strength_inverts_delta():
    node = qnn.LatentAddNoise()
    latent = make_flow_matching_latent()

    (positive,) = node.add_noise(latent, seed=42, strength=0.75)
    (negative,) = node.add_noise(latent, seed=42, strength=-0.75)

    positive_delta = positive["samples"] - latent["samples"]
    negative_delta = negative["samples"] - latent["samples"]

    assert torch.allclose(negative_delta, -positive_delta, atol=1e-6)


def test_latent_add_noise_with_zero_variance_latent_is_stable():
    node = qnn.LatentAddNoise()
    zero_latent = make_latent_from_tensor(
        torch.zeros(
            1,
            FLOW_MATCHING_CHANNELS,
            FLOW_MATCHING_TEMPORAL,
            FLOW_MATCHING_WIDTH,
            FLOW_MATCHING_HEIGHT,
        )
    )

    (result,) = node.add_noise(zero_latent, seed=0, strength=50.0)

    assert torch.allclose(result["samples"], zero_latent["samples"])


def test_latent_add_noise_rejects_integer_latent():
    node = qnn.LatentAddNoise()
    latent = make_latent_from_tensor(
        torch.zeros(
            1,
            FLOW_MATCHING_CHANNELS,
            FLOW_MATCHING_TEMPORAL,
            FLOW_MATCHING_WIDTH,
            FLOW_MATCHING_HEIGHT,
            dtype=torch.int32,
        )
    )

    with pytest.raises(RuntimeError):
        node.add_noise(latent, seed=0, strength=1.0)


def test_latent_frequency_split_zero_sigma_returns_zero_high_pass():
    node = qnn.LatentFrequencySplit()
    constant = torch.ones(1, FLOW_MATCHING_CHANNELS, FLOW_MATCHING_TEMPORAL, 4, 4)
    latent = make_latent_from_tensor(constant)

    low, high = node.split(latent, sigma=0.0)

    assert_flow_matching_shape(low["samples"], width=4, height=4)
    assert_flow_matching_shape(high["samples"], width=4, height=4)
    assert torch.allclose(low["samples"], constant)
    assert torch.allclose(high["samples"], torch.zeros_like(constant))


def test_latent_frequency_merge_reconstructs_original():
    split_node = qnn.LatentFrequencySplit()
    merge_node = qnn.LatentFrequencyMerge()
    latent = make_flow_matching_latent()

    low, high = split_node.split(latent, sigma=1.0)
    (merged,) = merge_node.merge(low, high, low_gain=1.0, high_gain=1.0)

    assert_flow_matching_shape(merged["samples"])
    assert torch.allclose(merged["samples"], latent["samples"], atol=1e-5)


def test_perlin_noise_strength_affects_latent():
    node = qnn.LatentPerlinFractalNoise()
    latent = make_flow_matching_latent()

    (result,) = node.add_perlin_noise(
        latent,
        seed=7,
        frequency=1.5,
        octaves=2,
        persistence=0.5,
        lacunarity=2.0,
        strength=0.5,
        channel_mode="shared",
    )

    assert_flow_matching_shape(result["samples"])
    assert not torch.allclose(result["samples"], latent["samples"])


def test_perlin_noise_negative_strength_inverts_delta():
    node = qnn.LatentPerlinFractalNoise()
    latent = make_flow_matching_latent()

    args = dict(
        seed=11,
        frequency=2.5,
        octaves=3,
        persistence=0.4,
        lacunarity=2.0,
    )

    (positive,) = node.add_perlin_noise(latent, strength=0.8, channel_mode="shared", **args)
    (negative,) = node.add_perlin_noise(latent, strength=-0.8, channel_mode="shared", **args)

    positive_delta = positive["samples"] - latent["samples"]
    negative_delta = negative["samples"] - latent["samples"]

    assert torch.allclose(negative_delta, -positive_delta, atol=1e-6)


def test_perlin_noise_per_channel_channels_diverge():
    node = qnn.LatentPerlinFractalNoise()
    latent_tensor = torch.randn(1, 4, 1, 8, 8)
    latent = make_latent_from_tensor(latent_tensor.clone())

    (result,) = node.add_perlin_noise(
        latent,
        seed=3,
        frequency=1.5,
        octaves=2,
        persistence=0.4,
        lacunarity=2.0,
        strength=0.75,
        channel_mode="per_channel",
    )

    samples = result["samples"]
    assert not torch.isnan(samples).any()
    deltas = samples - latent_tensor
    assert not torch.allclose(deltas[0, 0], deltas[0, 1])


def test_simplex_noise_zero_strength_is_noop():
    node = qnn.LatentSimplexNoise()
    latent = make_flow_matching_latent()

    (result,) = node.add_simplex_noise(
        latent,
        seed=42,
        frequency=1.0,
        octaves=2,
        persistence=0.5,
        lacunarity=2.0,
        strength=0.0,
        channel_mode="shared",
        temporal_mode="locked",
    )

    assert_flow_matching_shape(result["samples"])
    assert torch.allclose(result["samples"], latent["samples"])


def test_simplex_noise_negative_strength_inverts_delta():
    node = qnn.LatentSimplexNoise()
    latent = make_flow_matching_latent()

    args = dict(
        seed=9,
        frequency=1.0,
        octaves=2,
        persistence=0.3,
        lacunarity=2.0,
        channel_mode="per_channel",
        temporal_mode="animated",
    )

    (positive,) = node.add_simplex_noise(latent, strength=0.5, **args)
    (negative,) = node.add_simplex_noise(latent, strength=-0.5, **args)

    positive_delta = positive["samples"] - latent["samples"]
    negative_delta = negative["samples"] - latent["samples"]

    assert torch.allclose(negative_delta, -positive_delta, atol=1e-6)


def test_image_add_noise_reproducible_with_seed():
    node = qnn.ImageAddNoise()
    image = make_image()

    (first,) = node.add_noise(image, seed=123, strength=0.8)
    (second,) = node.add_noise(image, seed=123, strength=0.8)
    (third,) = node.add_noise(image, seed=321, strength=0.8)

    assert first.shape == image.shape
    assert torch.allclose(first, second)
    assert not torch.allclose(first, third)


def test_image_add_noise_batch_seed_offsets():
    node = qnn.ImageAddNoise()
    base = torch.linspace(0.0, 1.0, 16 * 16 * 3, dtype=torch.float32).reshape(1, 16, 16, 3)
    image = base.repeat(2, 1, 1, 1)

    (batched,) = node.add_noise(image, seed=55, strength=0.4)
    (single,) = node.add_noise(base.clone(), seed=56, strength=0.4)

    assert batched.shape[0] == 2
    assert torch.allclose(batched[1], single[0])


def test_image_add_noise_respects_mask():
    node = qnn.ImageAddNoise()
    image = make_image(height=24, width=24)

    mask = torch.zeros((1, 24, 24), dtype=torch.float32)
    mask[:, :, :12] = 1.0

    (full,) = node.add_noise(image, seed=123, strength=0.8)
    (masked,) = node.add_noise(image, seed=123, strength=0.8, mask=mask)

    expected = image * (1.0 - mask.unsqueeze(-1)) + full * mask.unsqueeze(-1)
    assert torch.allclose(masked, expected)


def test_image_add_noise_zero_variance_is_stable():
    node = qnn.ImageAddNoise()
    image = torch.zeros(1, 16, 16, 3)

    (result,) = node.add_noise(image, seed=0, strength=10.0)

    assert torch.allclose(result, image)


def test_image_perlin_noise_strength_affects_image():
    node = qnn.ImagePerlinFractalNoise()
    image = make_image()

    (result,) = node.add_perlin_noise(
        image,
        seed=7,
        frequency=1.5,
        octaves=2,
        persistence=0.5,
        lacunarity=2.0,
        strength=0.5,
        channel_mode="shared",
        temporal_mode="locked",
    )

    assert result.shape == image.shape
    assert not torch.allclose(result, image)


def test_fbm_noise_zero_strength_returns_original_object():
    node = qnn.LatentFractalBrownianMotion()
    latent = make_flow_matching_latent()

    (result,) = node.add_fbm_noise(
        latent,
        seed=0,
        base_noise="simplex",
        frequency=1.0,
        feature_points=8,
        octaves=2,
        persistence=0.5,
        lacunarity=2.0,
        distance_metric="euclidean",
        jitter=0.25,
        strength=0.0,
        channel_mode="shared",
        temporal_mode="locked",
    )

    assert result is not latent
    assert torch.allclose(result["samples"], latent["samples"])


def test_image_simplex_noise_zero_strength_is_noop():
    node = qnn.ImageSimplexNoise()
    image = make_image()

    (result,) = node.add_simplex_noise(
        image,
        seed=11,
        frequency=1.0,
        octaves=2,
        persistence=0.5,
        lacunarity=2.0,
        strength=0.0,
        channel_mode="shared",
        temporal_mode="locked",
    )

    assert torch.allclose(result, image)


def test_image_worley_noise_negative_strength_inverts_delta():
    node = qnn.ImageWorleyNoise()
    image = make_image()

    kwargs = dict(
        seed=19,
        feature_points=10,
        octaves=3,
        persistence=0.5,
        lacunarity=2.0,
        distance_metric="euclidean",
        jitter=0.3,
        channel_mode="shared",
        temporal_mode="locked",
    )

    (positive,) = node.add_worley_noise(image, strength=0.5, **kwargs)
    (negative,) = node.add_worley_noise(image, strength=-0.5, **kwargs)

    positive_delta = positive - image
    negative_delta = negative - image

    assert torch.allclose(negative_delta, -positive_delta, atol=1e-6)


def test_fbm_noise_with_constant_input_generates_variation():
    node = qnn.LatentFractalBrownianMotion()
    latent = make_latent_from_tensor(torch.zeros(1, 4, 1, 10, 10))

    (result,) = node.add_fbm_noise(
        latent,
        seed=1,
        base_noise="perlin",
        frequency=1.5,
        feature_points=4,
        octaves=3,
        persistence=0.45,
        lacunarity=2.2,
        distance_metric="euclidean",
        jitter=0.1,
        strength=0.8,
        channel_mode="per_channel",
        temporal_mode="locked",
    )

    delta = result["samples"] - latent["samples"]
    assert not torch.allclose(delta, torch.zeros_like(delta))


def test_fbm_noise_animated_temporal_frames_differ():
    node = qnn.LatentFractalBrownianMotion()
    latent = make_flow_matching_latent(temporal=2, width=6, height=6)

    (result,) = node.add_fbm_noise(
        latent,
        seed=7,
        base_noise="simplex",
        frequency=1.0,
        feature_points=8,
        octaves=2,
        persistence=0.6,
        lacunarity=2.0,
        distance_metric="manhattan",
        jitter=0.25,
        channel_mode="per_channel",
        strength=0.75,
        temporal_mode="animated",
    )

    samples = result["samples"]
    assert samples.shape[2] == 2
    first_frame = samples[:, :, 0]
    second_frame = samples[:, :, 1]
    assert not torch.allclose(first_frame, second_frame)


def test_image_reaction_diffusion_changes_image():
    node = qnn.ImageReactionDiffusion()
    image = make_image(height=24, width=24)

    (result,) = node.add_reaction_diffusion(
        image,
        seed=0,
        iterations=20,
        feed_rate=0.037,
        kill_rate=0.065,
        diffusion_u=0.16,
        diffusion_v=0.08,
        time_step=1.0,
        strength=0.8,
        channel_mode="shared",
        temporal_mode="locked",
    )

    assert result.shape == image.shape
    assert not torch.allclose(result, image)


def test_image_fractal_brownian_motion_respects_base_noise():
    node = qnn.ImageFractalBrownianMotion()
    image = make_image()

    (simplex,) = node.add_fbm_noise(
        image,
        seed=3,
        base_noise="simplex",
        frequency=2.0,
        feature_points=12,
        octaves=3,
        persistence=0.5,
        lacunarity=2.0,
        distance_metric="euclidean",
        jitter=0.3,
        strength=0.5,
        channel_mode="shared",
        temporal_mode="locked",
    )

    (worley,) = node.add_fbm_noise(
        image,
        seed=3,
        base_noise="worley",
        frequency=1.5,
        feature_points=12,
        octaves=3,
        persistence=0.5,
        lacunarity=2.0,
        distance_metric="euclidean",
        jitter=0.3,
        strength=0.5,
        channel_mode="shared",
        temporal_mode="locked",
    )

    assert simplex.shape == image.shape
    assert worley.shape == image.shape
    assert not torch.allclose(simplex, worley)


def test_image_fractal_brownian_motion_batch_reproducible_with_seed():
    node = qnn.ImageFractalBrownianMotion()
    base = torch.linspace(0.0, 1.0, 24 * 24 * 3, dtype=torch.float32).reshape(1, 24, 24, 3)
    image = base.repeat(3, 1, 1, 1)

    args = dict(
        seed=3,
        base_noise="simplex",
        frequency=2.0,
        feature_points=12,
        octaves=3,
        persistence=0.5,
        lacunarity=2.0,
        distance_metric="euclidean",
        jitter=0.3,
        strength=0.5,
        channel_mode="shared",
        temporal_mode="locked",
    )

    (first,) = node.add_fbm_noise(image, **args)
    (second,) = node.add_fbm_noise(image, **args)

    assert first.shape == image.shape
    assert torch.allclose(first, second)
    assert not torch.allclose(first[0], first[1])


def test_image_swirl_noise_modifies_pixels():
    node = qnn.ImageSwirlNoise()
    image = make_image()

    (result,) = node.add_swirl_noise(
        image,
        seed=42,
        vortices=2,
        channel_mode="global",
        channel_fraction=1.0,
        strength=0.5,
        radius=0.5,
        center_spread=0.3,
        direction_bias=0.0,
        mix=1.0,
    )

    assert result.shape == image.shape
    assert not torch.allclose(result, image)


def test_reaction_diffusion_per_channel_channels_diverge():
    node = qnn.LatentReactionDiffusion()
    latent = make_latent_from_tensor(torch.zeros(1, 3, 1, 12, 12))

    (result,) = node.add_reaction_diffusion(
        latent,
        seed=5,
        iterations=5,
        feed_rate=0.03,
        kill_rate=0.058,
        diffusion_u=0.16,
        diffusion_v=0.08,
        time_step=1.0,
        strength=1.0,
        channel_mode="per_channel",
        temporal_mode="locked",
    )

    samples = result["samples"]
    assert not torch.isnan(samples).any()
    assert not torch.allclose(samples[0, 0], samples[0, 1])


def test_swirl_noise_channel_fraction_zero_is_noop():
    node = qnn.LatentSwirlNoise()
    latent = make_flow_matching_latent(width=6, height=6)

    (result,) = node.add_swirl_noise(
        latent,
        seed=9,
        vortices=3,
        channel_mode="global",
        channel_fraction=0.0,
        strength=1.0,
        radius=0.5,
        center_spread=0.25,
        direction_bias=0.0,
        mix=1.0,
    )

    assert torch.allclose(result["samples"], latent["samples"])


def test_latent_forward_diffusion_batch_seed_offsets():
    node = qnn.LatentForwardDiffusion()
    base = torch.linspace(0.0, 1.0, 4 * 8 * 8, dtype=torch.float32).reshape(1, 4, 8, 8)
    latent = make_latent_from_tensor(base.repeat(2, 1, 1, 1))

    (batched,) = node.add_scheduled_noise(None, latent, seed=7, steps=5, noise_strength=0.6)
    (single,) = node.add_scheduled_noise(None, make_latent_from_tensor(base.clone()), seed=8, steps=5, noise_strength=0.6)

    assert batched["samples"].shape[0] == 2
    assert torch.allclose(batched["samples"][1], single["samples"][0])


def test_conditioning_nodes_modify_embeddings_and_metadata():
    noise_node = qnn.ConditioningAddNoise()
    blur_node = qnn.ConditioningGaussianBlur()

    embedding = torch.randn(2, 4, 6)
    pooled = torch.randn(4)
    conditioning = [[embedding.clone(), {"pooled_output": pooled.clone()}]]

    (noised,) = noise_node.add_noise(conditioning, seed=0, strength=0.5)
    (blurred,) = blur_node.blur(noised, sigma=1.0)

    noised_embedding = noised[0][0]
    noised_pooled = noised[0][1]["pooled_output"]

    assert not torch.allclose(noised_embedding, embedding)
    assert not torch.allclose(noised_pooled, pooled)

    blurred_embedding = blurred[0][0]
    assert blurred_embedding.shape == embedding.shape


def test_conditioning_add_noise_negative_strength_reverses_delta():
    node = qnn.ConditioningAddNoise()

    embedding = torch.randn(3, 2, 5)
    pooled = torch.randn(5)
    conditioning = [[embedding.clone(), {"pooled_output": pooled.clone()}]]

    (positive,) = node.add_noise(conditioning, seed=123, strength=0.6)
    (negative,) = node.add_noise(conditioning, seed=123, strength=-0.6)

    positive_embed_delta = positive[0][0] - embedding
    negative_embed_delta = negative[0][0] - embedding
    assert torch.allclose(negative_embed_delta, -positive_embed_delta, atol=1e-6)

    positive_pooled_delta = positive[0][1]["pooled_output"] - pooled
    negative_pooled_delta = negative[0][1]["pooled_output"] - pooled
    assert torch.allclose(negative_pooled_delta, -positive_pooled_delta, atol=1e-6)


def test_conditioning_add_noise_leaves_non_tensor_entries_untouched():
    node = qnn.ConditioningAddNoise()

    conditioning = [["not a tensor", {"pooled_output": "still not a tensor"}]]

    (result,) = node.add_noise(conditioning, seed=0, strength=1.0)

    assert result[0][0] == "not a tensor"
    assert result[0][1]["pooled_output"] == "still not a tensor"


def test_conditioning_gaussian_blur_zero_sigma_returns_original_list():
    node = qnn.ConditioningGaussianBlur()

    embedding = torch.randn(2, 4, 6)
    pooled = torch.randn(6)
    conditioning = [[embedding.clone(), {"pooled_output": pooled.clone()}]]

    (result,) = node.blur(conditioning, sigma=0.0)

    assert result is conditioning


def test_conditioning_frequency_split_handles_non_tensor_entries():
    node = qnn.ConditioningFrequencySplit()

    conditioning = [["prompt", {"pooled_output": "meta"}]]

    low, high = node.split(conditioning, sigma=0.5)

    assert low[0][0] == "prompt"
    assert high[0][0] == "prompt"
    assert low[0][1]["pooled_output"] == "meta"
    assert high[0][1]["pooled_output"] == "meta"


def test_conditioning_frequency_merge_mismatched_lengths_raises():
    node = qnn.ConditioningFrequencyMerge()

    low_pass = [[torch.zeros(2, 3), {}]]
    high_pass = [[torch.zeros(2, 3), {}], [torch.zeros(2, 3), {}]]

    with pytest.raises(ValueError):
        node.merge(low_pass, high_pass, low_gain=1.0, high_gain=1.0)


def test_ksampler_lora_sigma_inverse_strength_curve():
    sigmas = torch.tensor([4.0, 2.0, 1.0, 0.0], dtype=torch.float32)

    strengths = qnn.KSamplerLoraSigmaInverse._compute_inverse_sigma_strengths(sigmas, max_lora_strength=1.5)

    expected = torch.tensor([0.0, 0.75, 1.125, 1.5], dtype=torch.float32)
    assert torch.allclose(strengths, expected, atol=1e-6)


def test_ksampler_lora_sigma_inverse_strength_curve_with_min():
    sigmas = torch.tensor([4.0, 2.0, 1.0, 0.0], dtype=torch.float32)

    strengths = qnn.KSamplerLoraSigmaInverse._compute_inverse_sigma_strengths(
        sigmas,
        max_lora_strength=1.2,
        min_lora_strength=0.2,
    )

    expected = torch.tensor([0.2, 0.7, 0.95, 1.2], dtype=torch.float32)
    assert torch.allclose(strengths, expected, atol=1e-6)


def test_ksampler_lora_sigma_inverse_strength_curve_with_step_window():
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], dtype=torch.float32)

    strengths = qnn.KSamplerLoraSigmaInverse._compute_inverse_sigma_strengths(
        sigmas,
        max_lora_strength=1.2,
        min_lora_strength=0.2,
        min_lora_step=1,
        max_lora_step=3,
    )

    expected = torch.tensor([0.0, 0.45, 0.7, 0.95, 0.0], dtype=torch.float32)
    assert torch.allclose(strengths, expected, atol=1e-6)


def test_ksampler_lora_sigma_inverse_builds_percent_schedule():
    class FakeModelSampling:
        @staticmethod
        def percent_to_sigma(percent):
            return 1.0 - float(percent)

    sigmas = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)
    schedule = qnn.KSamplerLoraSigmaInverse._build_percent_strength_schedule(
        FakeModelSampling(),
        sigmas=sigmas,
        max_lora_strength=1.0,
        min_lora_strength=0.0,
        min_lora_step=-1,
        max_lora_step=-1,
    )

    percents = [value[0] for value in schedule]
    strengths = [value[1] for value in schedule]

    assert percents == pytest.approx([0.0, 0.5, 1.0], abs=1e-4)
    assert strengths == pytest.approx([0.0, 0.5, 1.0], abs=1e-6)


def test_ksampler_lora_sigma_inverse_applies_scheduled_hooks(monkeypatch):
    captured = {}

    class FakeHookKeyframe:
        def __init__(self, strength, start_percent, guarantee_steps=1):
            self.strength = strength
            self.start_percent = start_percent
            self.guarantee_steps = guarantee_steps

    class FakeHookKeyframeGroup:
        def __init__(self):
            self.keyframes = []

        def add(self, keyframe):
            self.keyframes.append(keyframe)

    class FakeHook:
        def __init__(self):
            self.hook_keyframe = None

    class FakeHookGroup:
        def __init__(self):
            self.hook = FakeHook()

        def set_keyframes_on_hooks(self, hook_kf):
            self.hook.hook_keyframe = hook_kf

    class FakeHooksModule:
        HookKeyframe = FakeHookKeyframe
        HookKeyframeGroup = FakeHookKeyframeGroup

        @staticmethod
        def create_hook_lora(lora, strength_model, strength_clip):
            captured["create_hook_strength_model"] = strength_model
            captured["create_hook_strength_clip"] = strength_clip
            captured["create_hook_payload"] = lora
            return FakeHookGroup()

        @staticmethod
        def set_hooks_for_conditioning(cond, hooks, append_hooks=True, cache=None):
            out = []
            for embedding, metadata in cond:
                new_meta = dict(metadata)
                new_meta["hooks"] = hooks
                out.append([embedding, new_meta])
            return out

    class FakeSamplerClass:
        SAMPLERS = ("euler",)
        SCHEDULERS = ("normal",)

        def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options=None):
            self.sigmas = torch.tensor([1.0, 0.4, 0.0], dtype=torch.float32)

    class FakeSamplersModule:
        KSampler = FakeSamplerClass

    class FakeComfySample:
        @staticmethod
        def fix_empty_latent_channels(model, latent, downscale_ratio_spacial=None):
            return latent

        @staticmethod
        def prepare_noise(latent, seed, batch_inds=None):
            return torch.zeros_like(latent)

        @staticmethod
        def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, **kwargs):
            captured["positive"] = positive
            captured["negative"] = negative
            captured["sigmas"] = kwargs.get("sigmas")
            return latent_image + 1.0

    class FakeComfyUtils:
        PROGRESS_BAR_ENABLED = False

        @staticmethod
        def load_torch_file(path, safe_load=True):
            captured["loaded_lora_path"] = path
            return {"loaded": True}

    class FakeFolderPaths:
        @staticmethod
        def get_full_path_or_raise(category, filename):
            assert category == "loras"
            return f"/virtual/loras/{filename}"

    class FakeModelSampling:
        @staticmethod
        def percent_to_sigma(percent):
            return 1.0 - float(percent)

    class FakeModel:
        load_device = torch.device("cpu")
        model_options = {}

        @staticmethod
        def get_model_object(name):
            assert name == "model_sampling"
            return FakeModelSampling()

    monkeypatch.setattr(qnn, "comfy_hooks", FakeHooksModule)
    monkeypatch.setattr(qnn, "comfy_samplers", FakeSamplersModule)
    monkeypatch.setattr(qnn, "comfy_sample", FakeComfySample)
    monkeypatch.setattr(qnn, "comfy_utils", FakeComfyUtils)
    monkeypatch.setattr(qnn, "folder_paths", FakeFolderPaths)
    monkeypatch.setattr(qnn, "latent_preview", None)

    node = qnn.KSamplerLoraSigmaInverse()
    latent = {"samples": torch.zeros((1, 4, 8, 8), dtype=torch.float32)}
    positive = [[torch.zeros((1, 1, 1), dtype=torch.float32), {}]]
    negative = [[torch.zeros((1, 1, 1), dtype=torch.float32), {}]]

    (out,) = node.sample(
        model=FakeModel(),
        seed=7,
        steps=2,
        cfg=8.0,
        sampler_name="euler",
        scheduler="normal",
        positive=positive,
        negative=negative,
        latent_image=latent,
        lora_name="test_lora.safetensors",
        min_lora_strength=0.0,
        max_lora_strength=2.0,
        min_lora_step=-1,
        max_lora_step=-1,
        denoise=1.0,
    )

    assert torch.allclose(out["samples"], torch.ones_like(latent["samples"]))
    assert captured["loaded_lora_path"].endswith("/test_lora.safetensors")
    assert captured["create_hook_strength_model"] == 1.0
    assert captured["create_hook_strength_clip"] == 0.0
    assert torch.allclose(captured["sigmas"], torch.tensor([1.0, 0.4, 0.0], dtype=torch.float32))

    positive_hooks = captured["positive"][0][1]["hooks"]
    keyframes = positive_hooks.hook.hook_keyframe.keyframes
    strengths = [keyframe.strength for keyframe in keyframes]
    percents = [keyframe.start_percent for keyframe in keyframes]

    assert strengths == pytest.approx([0.0, 1.2, 2.0], abs=1e-6)
    assert percents == pytest.approx([0.0, 0.6, 1.0], abs=1e-4)


def test_ksampler_lora_sigma_inverse_uses_bypass_path(monkeypatch):
    captured = {
        "multipliers": [],
        "set_injections_called": 0,
        "hook_fallback_called": False,
    }

    class FakeWeightAdapterBase:
        def __init__(self):
            self.multiplier = -1.0

    class FakeBypassInjectionManager:
        def __init__(self):
            self.adapters = {}
            self.hooks = []

        def add_adapter(self, key, adapter, strength=1.0):
            adapter.multiplier = strength
            self.adapters[key] = adapter

        def create_injections(self, model):
            self.hooks = [
                type("FakeHook", (), {"adapter": adapter})()
                for adapter in self.adapters.values()
            ]
            return ["fake_injection"]

        def get_hook_count(self):
            return len(self.hooks)

    class FakeWeightAdapterModule:
        WeightAdapterBase = FakeWeightAdapterBase
        BypassInjectionManager = FakeBypassInjectionManager

    class FakeComfyLora:
        @staticmethod
        def model_lora_keys_unet(model, key_map=None):
            return {"test_key": "layer.weight"}

        @staticmethod
        def load_lora(lora, key_map, log_missing=False):
            return {"layer.weight": FakeWeightAdapterBase()}

    class FakeSamplerClass:
        SAMPLERS = ("euler",)
        SCHEDULERS = ("normal",)

        def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options=None):
            self.sigmas = torch.tensor([1.0, 0.4, 0.0], dtype=torch.float32)

    class FakeSamplersModule:
        KSampler = FakeSamplerClass

    class FakeComfySample:
        @staticmethod
        def fix_empty_latent_channels(model, latent, downscale_ratio_spacial=None):
            return latent

        @staticmethod
        def prepare_noise(latent, seed, batch_inds=None):
            return torch.zeros_like(latent)

        @staticmethod
        def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, **kwargs):
            wrapper = model.model_options.get("model_function_wrapper")
            assert wrapper is not None

            def fake_apply_model(x, timestep, **extra):
                adapter = next(iter(model.bypass_manager.adapters.values()))
                captured["multipliers"].append(float(adapter.multiplier))
                return x

            wrapper(fake_apply_model, {"input": latent_image, "timestep": torch.tensor([1.0]), "c": {}})
            wrapper(fake_apply_model, {"input": latent_image, "timestep": torch.tensor([0.25]), "c": {}})
            return latent_image + 1.0

    class FakeComfyUtils:
        PROGRESS_BAR_ENABLED = False

        @staticmethod
        def load_torch_file(path, safe_load=True):
            return {"loaded": True}

    class FakeFolderPaths:
        @staticmethod
        def get_full_path_or_raise(category, filename):
            return f"/virtual/loras/{filename}"

    class FakeHooks:
        @staticmethod
        def set_hooks_for_conditioning(*args, **kwargs):
            captured["hook_fallback_called"] = True
            raise AssertionError("Hook fallback should not be used in bypass test")

    class FakeModelInner:
        @staticmethod
        def state_dict():
            return {"layer.weight": torch.zeros((1, 1), dtype=torch.float32)}

    class FakeModel:
        def __init__(self):
            self.load_device = torch.device("cpu")
            self.model_options = {}
            self.model = FakeModelInner()
            self.bypass_manager = None

        def clone(self):
            clone = FakeModel()
            clone.model_options = dict(self.model_options)
            return clone

        def set_injections(self, key, injections):
            captured["set_injections_called"] += 1

        def set_model_unet_function_wrapper(self, wrapper):
            self.model_options["model_function_wrapper"] = wrapper

    def patched_build_bypass(
        self,
        model,
        adapter_patches,
        max_sigma,
        max_lora_strength,
        min_lora_strength,
        reference_sigmas,
        min_lora_step,
        max_lora_step,
    ):
        model_with_lora = model.clone()
        manager = FakeBypassInjectionManager()
        for key, adapter in adapter_patches.items():
            manager.add_adapter(key, adapter, strength=min_lora_strength)
        injections = manager.create_injections(model_with_lora.model)
        model_with_lora.set_injections("skoogeer_lora_sigma_inverse", injections)
        model_with_lora.bypass_manager = manager
        self._install_bypass_strength_wrapper(
            model_with_lora=model_with_lora,
            adapters=[hook.adapter for hook in manager.hooks],
            max_sigma=max_sigma,
            max_lora_strength=max_lora_strength,
            min_lora_strength=min_lora_strength,
            reference_sigmas=reference_sigmas,
            min_lora_step=min_lora_step,
            max_lora_step=max_lora_step,
        )
        return model_with_lora, len(manager.hooks), 0

    monkeypatch.setattr(qnn, "comfy_weight_adapter", FakeWeightAdapterModule)
    monkeypatch.setattr(qnn, "comfy_lora", FakeComfyLora)
    monkeypatch.setattr(qnn, "comfy_samplers", FakeSamplersModule)
    monkeypatch.setattr(qnn, "comfy_sample", FakeComfySample)
    monkeypatch.setattr(qnn, "comfy_utils", FakeComfyUtils)
    monkeypatch.setattr(qnn, "folder_paths", FakeFolderPaths)
    monkeypatch.setattr(qnn, "comfy_hooks", FakeHooks)
    monkeypatch.setattr(qnn, "latent_preview", None)
    monkeypatch.setattr(
        qnn.KSamplerLoraSigmaInverse,
        "_build_bypass_lora_model",
        patched_build_bypass,
    )

    node = qnn.KSamplerLoraSigmaInverse()
    latent = {"samples": torch.zeros((1, 4, 8, 8), dtype=torch.float32)}
    positive = [[torch.zeros((1, 1, 1), dtype=torch.float32), {}]]
    negative = [[torch.zeros((1, 1, 1), dtype=torch.float32), {}]]

    (out,) = node.sample(
        model=FakeModel(),
        seed=11,
        steps=2,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive=positive,
        negative=negative,
        latent_image=latent,
        lora_name="fast.safetensors",
        min_lora_strength=0.0,
        max_lora_strength=2.0,
        min_lora_step=-1,
        max_lora_step=-1,
        denoise=1.0,
    )

    assert torch.allclose(out["samples"], torch.ones_like(latent["samples"]))
    assert captured["set_injections_called"] == 1
    assert captured["hook_fallback_called"] is False
    assert captured["multipliers"][0] == pytest.approx(0.0, abs=1e-6)
    assert captured["multipliers"][1] == pytest.approx(1.5, abs=1e-6)
