from __future__ import annotations

import numpy as np
import pytest

from pyradar.utils import (
    axis_aligned_bounds,
    compose_transforms,
    icp_register,
    inverse_transform,
    make_transform,
    max_unambiguous_range,
    max_unambiguous_velocity,
    random_sample,
    range_axis,
    range_resolution,
    transform_points,
    ula_angle_axis,
    ula_unambiguous_fov,
    velocity_axis,
    velocity_resolution,
    voxel_downsample,
)
from pyradar.utils.io import decode_dca1000, decode_tsw1400, read_dca1000


def _encode_dca1000(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(-1)
    output = np.empty(2 * flattened.size, dtype=np.int16)
    output[0::4] = flattened[0::2].real.astype(np.int16)
    output[1::4] = flattened[1::2].real.astype(np.int16)
    output[2::4] = flattened[0::2].imag.astype(np.int16)
    output[3::4] = flattened[1::2].imag.astype(np.int16)
    return output


def test_ti_dca1000_decoder_and_radar_frames(tmp_path, ula_radar) -> None:
    shape = (
        ula_radar.sampler.numLoops * ula_radar.mimo.numEmissions,
        ula_radar.mimo.numRx,
        ula_radar.sampler.numSamples,
    )
    count = int(np.prod(shape))
    values = np.arange(count, dtype=np.int16).reshape(shape) % 100 + 1j * (
        np.arange(count, dtype=np.int16).reshape(shape) % 37
    )
    raw = _encode_dca1000(values)
    decoded = decode_dca1000(
        raw,
        numChirps=shape[0],
        numRx=shape[1],
        numSamples=shape[2],
    )
    np.testing.assert_array_equal(decoded, values)
    path = tmp_path / "capture.bin"
    raw.tofile(path)
    frames = read_dca1000(path, ula_radar)
    assert len(frames) == 1
    np.testing.assert_array_equal(
        frames[0].data,
        values.reshape(
            ula_radar.sampler.numLoops,
            ula_radar.mimo.numEmissions,
            ula_radar.mimo.numRx,
            ula_radar.sampler.numSamples,
        ),
    )


def test_tsw1400_offset_binary_decoder() -> None:
    values = np.asarray([[[[-4 + 2j, 7 - 3j, 10 + 5j, -1 - 8j]]]])
    interleaved = np.empty((1, 1, 1, 8), dtype=np.int32)
    interleaved[..., 0::2] = values.real
    interleaved[..., 1::2] = values.imag
    raw = (interleaved + 2**15).astype(np.uint16)
    decoded = decode_tsw1400(
        raw,
        numFrames=1,
        numChirpsPerFrame=1,
        numRx=1,
        numSamples=4,
    )
    np.testing.assert_array_equal(decoded, values)
    with pytest.raises(ValueError, match="Expected"):
        decode_dca1000(np.zeros(3), numChirps=1, numRx=1, numSamples=2)


def test_pointcloud_sampling_bounds_and_icp() -> None:
    generator = np.random.default_rng(2)
    source = generator.normal(size=(100, 3))
    angle = 0.08
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    expected = make_transform(rotation, np.asarray([0.1, -0.05, 0.03]))
    target = transform_points(source, expected)
    result = icp_register(source, target, tolerance=1e-10, maxIterations=50)
    assert result.converged and result.rmse < 1e-8
    np.testing.assert_allclose(result.transform, expected, atol=1e-8)
    downsampled = voxel_downsample(np.vstack((source, source)), voxelSize=0.01)
    assert len(downsampled) <= len(source)
    assert random_sample(source, count=5, seed=4).shape == (5, 3)
    minimum, maximum = axis_aligned_bounds(source)
    assert np.all(minimum <= maximum)


def test_transform_composition_and_parameter_converters() -> None:
    first = make_transform(np.eye(3), np.asarray([1.0, 0.0, 0.0]))
    second = make_transform(np.eye(3), np.asarray([0.0, 2.0, 0.0]))
    combined = compose_transforms(first, second)
    point = np.asarray([[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(transform_points(point, combined), ((1.0, 2.0, 0.0),))
    np.testing.assert_allclose(
        transform_points(
            transform_points(point, combined), inverse_transform(combined)
        ),
        point,
    )
    assert range_resolution(bandwidth=1e9) == pytest.approx(299_792_458.0 / 2e9)
    assert max_unambiguous_range(sampleRate=2e6, slope=20e12) > 0.0
    assert velocity_resolution(
        wavelength=0.004, numSlowTimeSamples=16, slowTimeInterval=1e-4
    ) == pytest.approx(1.25)
    assert max_unambiguous_velocity(
        wavelength=0.004, slowTimeInterval=1e-4
    ) == pytest.approx(10.0)
    maximumRange = max_unambiguous_range(sampleRate=2e6, slope=20e12)
    np.testing.assert_allclose(
        range_axis(fftSize=4, sampleRate=2e6, slope=20e12),
        np.arange(4) * maximumRange / 4,
    )
    np.testing.assert_allclose(
        velocity_axis(fftSize=4, wavelength=0.004, slowTimeInterval=1e-4),
        [-10.0, -5.0, 0.0, 5.0],
    )
    assert ula_unambiguous_fov(wavelength=0.004, spacing=0.002) == (
        -np.pi / 2,
        np.pi / 2,
    )
    np.testing.assert_allclose(
        ula_angle_axis(fftSize=4, wavelength=0.004, spacing=0.002),
        [-np.pi / 2, -np.pi / 6, 0.0, np.pi / 6],
    )
    with pytest.raises(ValueError, match="finite"):
        make_transform(np.eye(3), np.asarray([np.nan, 0.0, 0.0]))
