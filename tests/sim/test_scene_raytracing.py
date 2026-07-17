from __future__ import annotations

import numpy as np
import pytest

from pyradar.sim import (
    RayPath,
    fresnel_schlick,
    linear_trajectory,
    reflect,
    refract,
    render_range_image,
    sample_cone_directions,
    sample_plane,
    sample_sphere,
)


def test_scene_generators_and_trajectory_are_deterministic() -> None:
    sphere = sample_sphere(radius=2.0, numPoints=32, center=(1.0, 2.0, 3.0), seed=4)
    np.testing.assert_allclose(np.linalg.norm(sphere - (1.0, 2.0, 3.0), axis=1), 2.0)
    np.testing.assert_array_equal(
        sphere,
        sample_sphere(radius=2.0, numPoints=32, center=(1.0, 2.0, 3.0), seed=4),
    )
    plane = sample_plane(
        width=2.0, height=3.0, numPoints=20, center=(5.0, 0.0, 0.0), seed=2
    )
    np.testing.assert_allclose(plane[:, 0], 5.0)
    poses = linear_trajectory((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), numFrames=3)
    np.testing.assert_allclose(poses[-1, :3, 3], (1.0, 2.0, 3.0))


def test_range_image_keeps_nearest_point_and_feature() -> None:
    points = np.asarray([[4.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 2.0, 0.0]])
    rendered = render_range_image(
        points,
        azimuthFov=(-0.5, 0.5),
        elevationFov=(-0.5, 0.5),
        angularResolution=(0.25, 0.25),
        features=np.asarray([[40], [20], [32]]),
    )
    row, column = np.argwhere(rendered.pointIndices == 1)[0]
    assert rendered.ranges[row, column] == 2.0
    assert rendered.features is not None and rendered.features[row, column, 0] == 20
    assert not np.any(rendered.pointIndices == 0)


def test_ray_optics_and_path_conversion() -> None:
    directions = sample_cone_directions(
        1000, np.deg2rad(20.0), forward=(1.0, 0.0, 0.0), seed=7
    )
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.min(directions[:, 0]) >= np.cos(np.deg2rad(20.0)) - 1e-12
    np.testing.assert_allclose(
        reflect((1.0, -1.0, 0.0), (0.0, 1.0, 0.0)), (2**-0.5, 2**-0.5, 0.0)
    )
    transmitted = refract((0.0, -1.0, 0.0), (0.0, 1.0, 0.0), 1.0, 1.5)
    assert transmitted is not None
    assert fresnel_schlick(1.0, 1.0, 1.5) == pytest.approx(0.04)
    path = RayPath(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]), True, 0.5
    )
    propagation = path.to_propagation_path(txId=0, rxId=1)
    assert propagation.pathLength == 3.0 and propagation.amplitude == 0.5


def test_invalid_ray_inputs_fail_loudly() -> None:
    with pytest.raises(ValueError, match="coneAngle"):
        sample_cone_directions(1, -1.0)
    with pytest.raises(ValueError, match="indices"):
        refract((1, 0, 0), (0, 1, 0), 0.0, 1.0)
    with pytest.raises(ValueError, match="receiver-terminated"):
        RayPath(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), False
        ).to_propagation_path(txId=0, rxId=0)
    missed = RayPath(np.asarray([[0.0, 0.0, 0.0]]), False)
    assert missed.length == 0.0
