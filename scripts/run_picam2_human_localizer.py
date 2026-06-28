"""Run PiCam2 + YOLO + supervision tracking and print robot-local human tracks.

This script is intended for the Raspberry Pi 5 deployment machine. Heavy camera
and model dependencies are imported only inside main so the repo test suite does
not require them.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from sncp_ppo.real_world import Detection2D, PlanarCalibration, VisionLocalizer


def _open_frame_source(args):
    if args.source == "picam2":
        from picamera2 import Picamera2

        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"format": "RGB888", "size": (args.width, args.height)}
        )
        camera.configure(config)
        camera.start()
        return camera, lambda: camera.capture_array()

    import cv2

    capture = cv2.VideoCapture(args.camera_index if args.source == "webcam" else args.video)
    if not capture.isOpened():
        raise RuntimeError(f"could not open frame source: {args.source}")
    return capture, lambda: capture.read()[1]


def _detections_from_ultralytics(result, tracker, min_confidence: float) -> list[Detection2D]:
    import supervision as sv

    detections = sv.Detections.from_ultralytics(result)
    if detections.class_id is not None:
        detections = detections[detections.class_id == 0]  # COCO person
    if detections.confidence is not None:
        detections = detections[detections.confidence >= min_confidence]
    detections = tracker.update_with_detections(detections)

    rows: list[Detection2D] = []
    tracker_ids = detections.tracker_id
    for i, xyxy in enumerate(detections.xyxy):
        confidence = 1.0 if detections.confidence is None else float(detections.confidence[i])
        track_id = None if tracker_ids is None else int(tracker_ids[i])
        rows.append(Detection2D(tuple(map(float, xyxy)), confidence, track_id))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, help="Calibration JSON from calibrate_camera_plane.py")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path/name")
    parser.add_argument("--source", choices=["picam2", "webcam", "video"], default="picam2")
    parser.add_argument("--video", help="Video path when --source video is used")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--csv-out", help="Optional CSV log path")
    args = parser.parse_args()

    from ultralytics import YOLO
    import supervision as sv

    calibration = PlanarCalibration.load(args.calibration)
    localizer = VisionLocalizer(calibration, min_confidence=args.confidence)
    model = YOLO(args.model)
    tracker = sv.ByteTrack()
    source, read_frame = _open_frame_source(args)

    csv_file = None
    writer = None
    if args.csv_out:
        csv_file = Path(args.csv_out).open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["time_s", "track_id", "x", "y", "vx", "vy", "confidence"],
        )
        writer.writeheader()

    try:
        while True:
            frame = read_frame()
            if frame is None:
                break
            timestamp = time.monotonic()
            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.confidence,
                classes=[0],
                device=args.device,
                verbose=False,
            )[0]
            detections = _detections_from_ultralytics(result, tracker, args.confidence)
            tracks = localizer.update(detections, timestamp)
            line = " | ".join(
                f"id={t.track_id} x={t.x:+.2f} y={t.y:+.2f} vx={t.vx:+.2f} vy={t.vy:+.2f}"
                for t in tracks
            )
            print(line or "no humans")
            if writer is not None:
                for t in tracks:
                    writer.writerow(
                        {
                            "time_s": f"{timestamp:.6f}",
                            "track_id": t.track_id,
                            "x": f"{t.x:.4f}",
                            "y": f"{t.y:.4f}",
                            "vx": f"{t.vx:.4f}",
                            "vy": f"{t.vy:.4f}",
                            "confidence": f"{t.confidence:.4f}",
                        }
                    )
                csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()
        if args.source == "picam2":
            source.stop()
        else:
            source.release()


if __name__ == "__main__":
    main()
