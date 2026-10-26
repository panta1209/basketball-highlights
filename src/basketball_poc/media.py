from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

from .core import number, timestamps


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise ValueError(f"{args[0]} failed: {result.stderr[-4000:]}")
    return result.stdout


def inspect_video(path):
    path = Path(path).resolve(strict=True)
    raw = json.loads(command(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)]))
    videos = [s for s in raw["streams"] if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")]
    if len(videos) != 1:
        raise ValueError("PoC requires exactly one video stream")
    video = videos[0]
    fps = float(Fraction(video.get("avg_frame_rate", "0/1")))
    rate = float(Fraction(video.get("r_frame_rate", "0/1")))
    duration = number(float(video.get("duration", raw["format"].get("duration", 0))), "duration")
    if fps <= 0 or duration <= 0:
        raise ValueError("Video duration and frame rate must be positive")
    rotations = [abs(float(s.get("rotation", 0))) % 360 for s in video.get("side_data_list", [])]
    if any(r != 0 for r in rotations) or float(video.get("tags", {}).get("rotate", 0)) % 360:
        raise ValueError("Video has rotation metadata; normalize its orientation with FFmpeg before calibration")
    return {"path": str(path), "duration_seconds": duration, "width": video["width"], "height": video["height"],
            "fps": fps, "nominal_fps": rate, "audio_streams": sum(s["codec_type"] == "audio" for s in raw["streams"]),
            "video_stream_index": video["index"], "size_bytes": path.stat().st_size}


def reference_frame(video, time, output):
    info = inspect_video(video)
    time = number(time, "frame time")
    if time >= info["duration_seconds"]:
        raise ValueError("Reference time must be inside the video")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Reference output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    command(["ffmpeg", "-v", "error", "-nostdin", "-n", "-ss", str(time), "-i", info["path"],
             "-frames:v", "1", "-update", "1", str(output)])
    return info


def export_clips(video, events, output, before=4, after=2):
    info = inspect_video(video)
    before, after = number(before, "before"), number(after, "after")
    if before + after <= 0:
        raise ValueError("before + after must be > 0")
    if events.get("source_sha256") != sha256(video):
        raise ValueError("Events do not refer to this source video")
    times = timestamps(events["events"], "events")
    if any(t >= info["duration_seconds"] for t in times):
        raise ValueError("Event outside source duration")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "source": info, "source_sha256": events["source_sha256"],
                "before_seconds": before, "after_seconds": after, "clips": [], "montage": None}
    try:
        for index, time in enumerate(times, 1):
            start, end = max(0, time - before), min(info["duration_seconds"], time + after)
            name = f"goal-{index:04d}.mp4"
            command(["ffmpeg", "-v", "error", "-nostdin", "-n", "-ss", str(start), "-i", info["path"],
                     "-t", str(end-start), "-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
                     "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                     "-c:a", "aac", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(output / name)])
            manifest["clips"].append({"file": name, "event_seconds": time, "start_seconds": start, "end_seconds": end})
        if times:
            # Only internally generated relative names appear in the concat file.
            listing = output / "concat.txt"
            listing.write_text("".join(f"file '{clip['file']}'\n" for clip in manifest["clips"]), encoding="utf-8")
            command(["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "concat", "-safe", "1", "-i", str(listing),
                     "-c", "copy", "-movflags", "+faststart", str(output / "montage.mp4")])
            manifest["montage"] = "montage.mp4"
        manifest["status"] = "completed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        write_json(output / "export.json", manifest)
    return manifest
