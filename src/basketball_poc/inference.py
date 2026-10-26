from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
import platform
import time

from .core import GoalTracker, Point, calibration, number
from .media import inspect_video, sha256, write_json


def analyze(video, model_path, raw_config, output, device="cpu", start=0, end=None):
    import cv2
    import torch
    from ultralytics import YOLO

    info = inspect_video(video)
    config = calibration(raw_config, info["width"], info["height"])
    start = number(start, "start")
    end = info["duration_seconds"] if end is None else number(end, "end")
    if not 0 <= start < end <= info["duration_seconds"]:
        raise ValueError("Analysis interval must lie inside source duration")
    if abs(info["fps"] - info["nominal_fps"]) > 0.01:
        raise ValueError("Variable frame rate is unsupported: normalize with FFmpeg first")
    model_path = Path(model_path).resolve(strict=True)
    source_hash, model_hash = sha256(video), sha256(model_path)
    actual_device = "0" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    interval = {"start_seconds": start, "end_seconds": end}
    manifest = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": info, "source_sha256": source_hash, "model_path": str(model_path), "model_sha256": model_hash,
                "config": config, "interval": interval, "requested_device": device, "device": actual_device,
                "python": platform.python_version(), "versions": {n: version(n) for n in ["ultralytics", "torch", "opencv-python"]}}
    implementation = hashlib.sha256()
    for module in sorted(Path(__file__).parent.glob("*.py")):
        implementation.update(module.name.encode("utf-8"))
        implementation.update(module.read_bytes())
    manifest["implementation_sha256"] = implementation.hexdigest()
    origin = model_path.with_suffix(".source.json")
    if origin.exists():
        manifest["model_provenance"] = json.loads(origin.read_text(encoding="utf-8"))
    capture = None
    began = time.perf_counter()
    processed = 0
    ambiguous = 0
    events = []
    try:
        model = YOLO(str(model_path))
        if config["ball_class"] not in model.names:
            raise ValueError(f"ball_class absent from model: {model.names}")
        manifest["model_classes"] = model.names
        capture = cv2.VideoCapture(info["path"])
        if not capture.isOpened():
            raise ValueError("OpenCV could not open the video")
        tracker = GoalTracker(config)
        rx1, ry1, rx2, ry2 = config["roi"]
        frame_number = 0
        previous_timestamp = None
        last_processed_timestamp = None
        with (output / "detections.jsonl").open("x", encoding="utf-8") as log:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                # Actual decoded media timestamp; never substitute wall-clock time.
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                expected = frame_number / info["fps"]
                if not math.isfinite(timestamp) or abs(timestamp - expected) > max(0.005, 0.1 / info["fps"]):
                    raise ValueError("Decoded timestamps are not constant-rate; normalize video with FFmpeg first")
                if previous_timestamp is not None and timestamp <= previous_timestamp:
                    raise ValueError("Decoder did not return increasing timestamps")
                previous_timestamp = timestamp
                current_frame = frame_number
                frame_number += 1
                if timestamp < start:
                    continue
                if timestamp >= end:
                    break
                if frame.shape[:2] != (info["height"], info["width"]):
                    raise ValueError("Decoded dimensions differ from calibration")
                crop = frame[ry1:ry2, rx1:rx2]
                prediction = model.predict(crop, conf=config["confidence"], imgsz=config["imgsz"],
                                           classes=[config["ball_class"]], device=actual_device, verbose=False)[0]
                balls, boxes = [], []
                for box in prediction.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = float(box.conf[0])
                    point = Point(timestamp, (x1+x2)/2 + rx1, (y1+y2)/2 + ry1, confidence)
                    balls.append(point)
                    boxes.append({"xyxy": [x1+rx1, y1+ry1, x2+rx1, y2+ry1], **asdict(point)})
                event = tracker.step(timestamp, balls)
                if event:
                    events.append({"id": f"goal-{len(events)+1:04d}", **event})
                log.write(json.dumps({"frame": current_frame, "time_seconds": timestamp, "balls": boxes}, allow_nan=False) + "\n")
                processed += 1
                ambiguous += len(balls) > 1
                last_processed_timestamp = timestamp
                if processed % 100 == 0:
                    print(f"{processed} frames, source {timestamp:.2f}s, {len(events)} candidates", flush=True)
        if not processed or last_processed_timestamp + 1.5 / info["fps"] < end:
            raise ValueError("Decoder ended before the requested interval was fully analyzed")
        write_json(output / "events.json", {"status": "candidates", "source_sha256": source_hash,
                                            "interval": interval, "events": events})
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        if capture is not None:
            capture.release()
        manifest["processing_seconds"] = time.perf_counter() - began
        manifest["processed_frames"] = processed
        manifest["ambiguous_frames"] = ambiguous
        manifest["candidate_count"] = len(events)
        write_json(output / "manifest.json", manifest)
    return manifest
