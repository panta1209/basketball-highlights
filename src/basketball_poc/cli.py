import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

from .core import evaluate
from .media import export_clips, inspect_video, read_json, reference_frame, sha256, write_json


def parser():
    root = argparse.ArgumentParser(description="Local basketball PoC: emitted goals are unverified candidates")
    sub = root.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor", help="Check runtime and actual CUDA support")
    p = sub.add_parser("inspect", help="Inspect local video and optionally write reference image")
    p.add_argument("video")
    p.add_argument("--frame")
    p.add_argument("--at", type=float, default=0)
    p = sub.add_parser("analyze", help="Detect candidate goals in a calibrated ROI")
    p.add_argument("video")
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--end", type=float)
    p = sub.add_parser("evaluate", help="Evaluate against complete manual goal annotations")
    p.add_argument("events")
    p.add_argument("annotations")
    p.add_argument("--tolerance", type=float, default=0.75)
    p.add_argument("--out", required=True)
    p = sub.add_parser("export", help="Export candidate clips and chronological montage")
    p.add_argument("video")
    p.add_argument("events")
    p.add_argument("--out", required=True)
    p.add_argument("--before", type=float, default=4)
    p.add_argument("--after", type=float, default=2)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.action == "doctor":
            import torch
            from importlib.metadata import version
            result = {"python": sys.version, "executable": sys.executable,
                      "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"),
                      "versions": {n: version(n) for n in ["torch", "ultralytics", "opencv-python"]},
                      "cuda_available": torch.cuda.is_available(), "torch_cuda_version": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        elif args.action == "inspect":
            result = reference_frame(args.video, args.at, args.frame) if args.frame else inspect_video(args.video)
            result["source_sha256"] = sha256(args.video)
        elif args.action == "analyze":
            from .inference import analyze
            result = analyze(args.video, args.model, read_json(args.config), args.out, args.device, args.start, args.end)
        elif args.action == "evaluate":
            if Path(args.out).exists():
                raise ValueError("Evaluation output already exists")
            result = evaluate(read_json(args.events), read_json(args.annotations), args.tolerance)
            result.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                           "events_path": str(Path(args.events).resolve()), "events_sha256": sha256(args.events),
                           "annotations_path": str(Path(args.annotations).resolve()), "annotations_sha256": sha256(args.annotations)})
            write_json(args.out, result)
        else:
            result = export_clips(args.video, read_json(args.events), args.out, args.before, args.after)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
