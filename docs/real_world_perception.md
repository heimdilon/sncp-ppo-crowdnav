# Real-World Perception Prototype

This is the first hardware path for moving the project from simulation to the
TurtleBot3 Waffle setup with Raspberry Pi 5 + Pi Camera Module 2.

## Sensor Contract

The policy needs each nearby person in the robot-local ground plane:

```text
id, x_forward_m, y_left_m, vx_mps, vy_mps, confidence
```

The monocular camera cannot measure metric depth directly. The first prototype
therefore assumes a flat floor and uses a homography from image pixels to robot
base coordinates.

## Calibration

Place at least four visible floor markers in front of the robot. Measure their
coordinates in meters in the robot frame:

- `+x`: forward from the robot
- `+y`: left of the robot
- origin: robot base frame projection on the floor

Create a point file:

```json
{
  "frame_id": "base_link",
  "image_points": [[320, 460], [160, 360], [480, 360], [320, 260]],
  "ground_points_m": [[0.5, 0.0], [1.0, 0.5], [1.0, -0.5], [1.5, 0.0]]
}
```

Then run:

```bash
python scripts/calibrate_camera_plane.py --points camera_points.json --out camera_plane.json
```

The script reports mean/max reprojection error in meters. Keep recalibrating
until the max error is small enough for safe testing; start with a target below
0.10 m on the calibration markers, then validate with a standing person.

## Live Pi Prototype

On the Raspberry Pi:

```bash
git clone https://github.com/heimdilon/sncp-ppo-crowdnav.git
cd sncp-ppo-crowdnav
sudo apt install python3-picamera2 python3-opencv python3-venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-realworld.txt
python scripts/run_picam2_human_localizer.py --calibration camera_plane.json --model yolo11n.pt
```

Use `--system-site-packages` deliberately: Raspberry Pi OS installs Picamera2
and OpenCV through apt, and a plain virtual environment would not see those
packages. Avoid `pip --break-system-packages`; it can damage the OS-managed
Python environment.

The script prints robot-local tracks. Use `--csv-out tracks.csv` during testing
so static and walking trials can be measured.

## Validation Gate

Before connecting this to `/cmd_vel`, run a tape-measure validation:

1. Put one person at known points such as `(1.0, 0.0)`, `(1.5, 0.5)`,
   `(2.0, -0.5)`.
2. Record 10 seconds per point.
3. Check median position error and jitter.
4. Walk left-right at roughly 0.3-0.8 m/s and inspect `vx, vy`.

Initial target:

- static median position error below 0.30-0.50 m
- stable updates above 10 Hz
- track IDs do not swap in one-person and two-person tests

If this fails, the camera-only path can still be used for person confirmation,
but metric localization should move to depth/stereo or a 2D tracking radar.
