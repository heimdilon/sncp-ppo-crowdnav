"""Create an image-to-ground homography calibration file.

Input JSON format:

{
  "frame_id": "base_link",
  "image_points": [[320, 460], [120, 360], [520, 360], [320, 260]],
  "ground_points_m": [[0.5, 0.0], [1.0, 0.5], [1.0, -0.5], [1.5, 0.0]]
}

Ground coordinates are robot-local meters: +x forward, +y left.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sncp_ppo.real_world import PlanarCalibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, help="JSON file with image/ground point pairs")
    parser.add_argument("--out", required=True, help="Output calibration JSON path")
    args = parser.parse_args()

    data = json.loads(Path(args.points).read_text(encoding="utf-8"))
    calibration = PlanarCalibration.from_points(
        data["image_points"],
        data["ground_points_m"],
        frame_id=data.get("frame_id", "base_link"),
    )
    calibration.save(args.out)
    errors = calibration.reprojection_errors()
    print(f"Wrote calibration: {args.out}")
    if len(errors):
        print(f"Mean reprojection error: {errors.mean():.4f} m")
        print(f"Max reprojection error: {errors.max():.4f} m")


if __name__ == "__main__":
    main()
