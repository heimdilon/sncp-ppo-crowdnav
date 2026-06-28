"""Run PiCam2 human detection and print robot-local human tracks.

This script is intended for the Raspberry Pi 5 deployment machine. Heavy camera
and model dependencies are imported only when their backend is selected, so the
repo test suite and the default Raspberry Pi path do not require PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sncp_ppo.real_world import Detection2D, ImageSpaceTracker, PlanarCalibration, VisionLocalizer


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


def _create_hog_detector():
    import cv2

    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return detector


def _detections_from_hog(frame, detector, hit_threshold: float) -> list[Detection2D]:
    rects, weights = detector.detectMultiScale(
        frame,
        hitThreshold=hit_threshold,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    detections: list[Detection2D] = []
    for i, rect in enumerate(rects):
        x, y, w, h = map(float, rect)
        weight = 1.0 if len(weights) <= i else float(weights[i])
        confidence = min(max(weight, 0.0), 1.0)
        detections.append(Detection2D((x, y, x + w, y + h), confidence=max(confidence, 0.01)))
    return detections


def _filter_detections_by_box(
    detections: list[Detection2D],
    *,
    min_box_height: float,
    min_box_width: float,
) -> list[Detection2D]:
    filtered: list[Detection2D] = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        if (y2 - y1) < min_box_height:
            continue
        if (x2 - x1) < min_box_width:
            continue
        filtered.append(det)
    return filtered


def _filter_tracks_by_ground(tracks, args):
    filtered = []
    for track in tracks:
        if track.x < args.min_ground_x:
            continue
        if track.x > args.max_ground_x:
            continue
        if abs(track.y) > args.max_abs_ground_y:
            continue
        filtered.append(track)
    return filtered


def _annotate_frame(frame, detections: list[Detection2D], tracks):
    import cv2

    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"id={det.track_id if det.track_id is not None else '?'} conf={det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    for track in tracks:
        text = f"id={track.track_id} x={track.x:+.2f} y={track.y:+.2f}"
        cv2.putText(annotated, text, (12, 24 + 18 * int(track.track_id % 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1)
    if not tracks:
        cv2.putText(annotated, "no humans", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
    return annotated


def _save_annotation(frame, detections: list[Detection2D], tracks, annotate_dir: Path, frame_index: int) -> None:
    import cv2

    annotated = _annotate_frame(frame, detections, tracks)
    annotate_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(annotate_dir / f"frame_{frame_index:06d}.jpg"), annotated)


class LiveMJPEGServer:
    """Tiny annotated-frame MJPEG server for SSH/headless Raspberry Pi use."""

    def __init__(self, *, host: str, port: int, jpeg_quality: int = 80) -> None:
        self.host = host
        self.port = int(port)
        self.jpeg_quality = int(jpeg_quality)
        self._lock = threading.Lock()
        self._frame: bytes | None = None

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
                if self.path not in ("/", "/stream.mjpg"):
                    self.send_error(404)
                    return
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(
                        b"<html><body style='margin:0;background:#111;'>"
                        b"<img src='/stream.mjpg' style='width:100%;height:auto;'/>"
                        b"</body></html>"
                    )
                    return

                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        frame = owner.latest_frame()
                        if frame is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def log_message(self, format, *args):  # noqa: A002 - inherited name
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def update(self, frame) -> None:
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        with self._lock:
            self._frame = encoded.tobytes()

    def latest_frame(self) -> bytes | None:
        with self._lock:
            return self._frame


def _load_ultralytics_backend(model_path: str):
    try:
        from ultralytics import YOLO
        import supervision as sv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The ultralytics backend needs torch/ultralytics/supervision. "
            "On Raspberry Pi OS where torch wheels are unavailable, run with "
            "--backend hog instead."
        ) from exc
    return YOLO(model_path), sv.ByteTrack()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, help="Calibration JSON from calibrate_camera_plane.py")
    parser.add_argument(
        "--backend",
        choices=["hog", "ultralytics"],
        default="hog",
        help="Detector backend. 'hog' uses only apt OpenCV; 'ultralytics' requires torch.",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path/name")
    parser.add_argument("--source", choices=["picam2", "webcam", "video"], default="picam2")
    parser.add_argument("--video", help="Video path when --source video is used")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hog-hit-threshold", type=float, default=0.0)
    parser.add_argument("--min-box-height", type=float, default=80.0)
    parser.add_argument("--min-box-width", type=float, default=30.0)
    parser.add_argument("--min-ground-x", type=float, default=0.0)
    parser.add_argument("--max-ground-x", type=float, default=4.0)
    parser.add_argument("--max-abs-ground-y", type=float, default=2.5)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means run forever")
    parser.add_argument("--annotate-dir", help="Optional directory for annotated debug frames")
    parser.add_argument("--annotate-every", type=int, default=20)
    parser.add_argument("--show", action="store_true", help="Show a live annotated OpenCV window")
    parser.add_argument("--stream-host", default="0.0.0.0")
    parser.add_argument("--stream-port", type=int, default=0, help="Serve live annotated MJPEG on this port; 0 disables")
    parser.add_argument("--print-pixels", action="store_true", help="Also print image-space boxes for detector debugging")
    parser.add_argument("--csv-out", help="Optional CSV log path")
    args = parser.parse_args()

    calibration = PlanarCalibration.load(args.calibration)
    localizer = VisionLocalizer(calibration, min_confidence=args.confidence)
    if args.backend == "ultralytics":
        model, tracker = _load_ultralytics_backend(args.model)
        image_tracker = None
    else:
        model = _create_hog_detector()
        tracker = None
        image_tracker = ImageSpaceTracker()
    source, read_frame = _open_frame_source(args)
    streamer = None
    if args.stream_port > 0:
        streamer = LiveMJPEGServer(host=args.stream_host, port=args.stream_port)
        streamer.start()
        print(f"Live stream: http://{args.stream_host}:{args.stream_port}/")

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
        frame_index = 0
        while True:
            frame = read_frame()
            if frame is None:
                break
            frame_index += 1
            timestamp = time.monotonic()
            if args.backend == "ultralytics":
                result = model.predict(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.confidence,
                    classes=[0],
                    device=args.device,
                    verbose=False,
                )[0]
                detections = _detections_from_ultralytics(result, tracker, args.confidence)
            else:
                detections = _detections_from_hog(frame, model, args.hog_hit_threshold)
                detections = _filter_detections_by_box(
                    detections,
                    min_box_height=args.min_box_height,
                    min_box_width=args.min_box_width,
                )
                detections = image_tracker.update(detections)
            tracks = localizer.update(detections, timestamp)
            tracks = _filter_tracks_by_ground(tracks, args)
            line = " | ".join(
                f"id={t.track_id} x={t.x:+.2f} y={t.y:+.2f} vx={t.vx:+.2f} vy={t.vy:+.2f}"
                for t in tracks
            )
            print(line or "no humans")
            if args.print_pixels and detections:
                print(
                    "pixels "
                    + " | ".join(
                        f"id={d.track_id} xyxy=({d.xyxy[0]:.0f},{d.xyxy[1]:.0f},{d.xyxy[2]:.0f},{d.xyxy[3]:.0f})"
                        for d in detections
                    )
                )
            needs_annotation = bool(args.show or streamer or args.annotate_dir)
            annotated = _annotate_frame(frame, detections, tracks) if needs_annotation else None
            if args.annotate_dir and args.annotate_every > 0 and frame_index % args.annotate_every == 0:
                _save_annotation(frame, detections, tracks, Path(args.annotate_dir), frame_index)
            if streamer is not None and annotated is not None:
                streamer.update(annotated)
            if args.show and annotated is not None:
                import cv2

                cv2.imshow("SNCP human localizer", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
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
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if csv_file is not None:
            csv_file.close()
        if args.source == "picam2":
            source.stop()
        else:
            source.release()
        if streamer is not None:
            streamer.stop()
        if args.show:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
