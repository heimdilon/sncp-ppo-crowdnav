import numpy as np
import pytest

from sncp_ppo.real_world import Detection2D, ImageSpaceTracker, PlanarCalibration, VisionLocalizer
from sncp_ppo.real_world.vision_localizer import bbox_bottom_center


def test_homography_maps_image_points_to_robot_ground_plane():
    image_points = [(100, 400), (500, 400), (500, 200), (100, 200)]
    ground_points = [(0.5, 0.5), (0.5, -0.5), (2.0, -0.5), (2.0, 0.5)]

    calibration = PlanarCalibration.from_points(image_points, ground_points)

    projected = calibration.pixels_to_ground(image_points)
    np.testing.assert_allclose(projected, np.asarray(ground_points), atol=1e-6)
    assert calibration.reprojection_errors().max() < 1e-6


def test_calibration_round_trips_through_json(tmp_path):
    calibration = PlanarCalibration.from_points(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        frame_id="base_link",
    )
    path = tmp_path / "camera_plane.json"

    calibration.save(path)
    loaded = PlanarCalibration.load(path)

    np.testing.assert_allclose(loaded.pixel_to_ground((5, 5)), [0.5, 0.5], atol=1e-6)
    assert loaded.frame_id == "base_link"


def test_bbox_bottom_center_uses_feet_proxy():
    assert bbox_bottom_center((10, 20, 30, 80)) == (20, 80)


def test_vision_localizer_projects_bottom_center_and_estimates_velocity():
    calibration = PlanarCalibration.from_points(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(0, 0), (1, 0), (1, 1), (0, 1)],
    )
    localizer = VisionLocalizer(calibration, velocity_alpha=1.0)

    first = localizer.update([Detection2D((40, 20, 60, 80), 0.9, track_id=7)], timestamp_s=0.0)
    second = localizer.update([Detection2D((50, 20, 70, 80), 0.9, track_id=7)], timestamp_s=0.5)

    assert len(first) == len(second) == 1
    assert first[0].track_id == 7
    np.testing.assert_allclose([first[0].x, first[0].y], [0.5, 0.8], atol=1e-6)
    np.testing.assert_allclose([second[0].vx, second[0].vy], [0.2, 0.0], atol=1e-6)


def test_vision_localizer_filters_low_confidence_and_sorts_by_distance():
    calibration = PlanarCalibration.from_points(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(0, 0), (1, 0), (1, 1), (0, 1)],
    )
    localizer = VisionLocalizer(calibration, min_confidence=0.5)
    detections = [
        Detection2D((90, 90, 100, 100), 0.9, track_id=1),
        Detection2D((10, 10, 20, 20), 0.4, track_id=2),
        Detection2D((0, 0, 10, 10), 0.8, track_id=3),
    ]

    tracks = localizer.update(detections, timestamp_s=1.0)

    assert [t.track_id for t in tracks] == [3, 1]
    assert all(t.confidence >= 0.5 for t in tracks)


def test_invalid_detection_box_is_rejected():
    with pytest.raises(ValueError, match="x2 > x1"):
        bbox_bottom_center((10, 20, 10, 30))


def test_image_space_tracker_keeps_nearby_detection_id_stable():
    tracker = ImageSpaceTracker(max_distance_px=30)

    first = tracker.update([Detection2D((10, 10, 30, 50), 1.0)])
    second = tracker.update([Detection2D((12, 10, 32, 50), 1.0)])

    assert first[0].track_id == second[0].track_id


def test_image_space_tracker_assigns_new_id_for_far_detection():
    tracker = ImageSpaceTracker(max_distance_px=10)

    first = tracker.update([Detection2D((10, 10, 30, 50), 1.0)])
    second = tracker.update([Detection2D((200, 10, 220, 50), 1.0)])

    assert first[0].track_id != second[0].track_id
