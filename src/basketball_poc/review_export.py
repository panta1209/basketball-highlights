"""Export human-reviewed ranges; the frozen automatic pipeline stays unchanged."""
from pathlib import Path
from .media import command, inspect_video, sha256, write_json
from .core import number, timestamps


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
        ordered = sorted(events['events'], key=lambda e: e['time_seconds'])
        for index, (time, event) in enumerate(zip(times, ordered, strict=True), 1):
            start = number(event.get('segment_start_seconds', max(0, time-before)), 'clip start')
            end = number(event.get('segment_end_seconds', min(info['duration_seconds'], time+after)), 'clip end')
            if not 0 <= start <= time < end <= info['duration_seconds']:
                raise ValueError('Clip range must contain the event and remain inside the video')
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
