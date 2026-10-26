"""Bounded observation of existing processing artifacts; no inference changes."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

from .media import read_json


def tail(path, limit=65536):
    try:
        with path.open('rb') as file:
            file.seek(0, 2)
            file.seek(max(0, file.tell()-limit))
            return file.read(limit).decode('utf-8', errors='replace')
    except OSError:
        return ''


def frame_progress(folder):
    for line in reversed(tail(folder/'candidates/detections.jsonl').splitlines()):
        try:
            return json.loads(line)['frame']+1
        except (ValueError, KeyError, TypeError):
            continue
    matches = re.findall(r'(\d+) frames,', tail(folder/'detection.log'))
    return int(matches[-1]) if matches else None


def observe(output, state, job):
    output = Path(output)
    now = time.time()
    start = datetime.fromisoformat(job['created_at_utc']).timestamp()
    done = job['status'] in ('ready_for_review', 'unsupported', 'failed', 'interrupted')
    elapsed = state.get('elapsed_seconds', max(0, now-start)) if done else max(0, now-start)
    stage = state.get('stage')
    fps = None; total = None
    try:
        source = read_json(output/'frozen.json')['source']
        fps = source['fps']; total = round(fps*source['duration_seconds'])
    except (OSError, ValueError, KeyError):
        pass
    label = {'preflight': '检查视频与模型', 'verification': '核验视频映射与结果完整性',
        'triage': '筛选待确认片段', 'rebound': '检查反弹候选', 'review': '准备进球预览'}.get(stage, '等待开始分析')
    completed = None
    paths = [output/'workflow.json', output/'logs'/f'{stage}.log']
    if stage == 'analysis':
        analysis = output/'analysis'
        active = analysis/('corrected' if (analysis/'corrected').exists() else 'original')
        paths += [analysis/'stabilization/stabilized.avi', analysis/'stabilization/stabilized.mkv',
                  output/'logs/analysis.log', active/'detection.log', active/'candidates/detections.jsonl']
        try:
            manifest = read_json(active/'candidates/manifest.json')
        except (OSError, ValueError):
            manifest = {}
        if manifest.get('status') == 'completed':
            label = '核验篮网变化与进球候选'
            paths += list(active.glob('*-native-crops.npz'))
        elif active.name == 'corrected' or (active/'candidates/detections.jsonl').exists():
            label = '识别篮球与投篮轨迹'; completed = frame_progress(active)
        elif (analysis/'stabilization').exists():
            label = '补偿镜头轻微晃动'
            matches = re.findall(r'stabilized (\d+) frames', tail(output/'logs/analysis.log'))
            completed = int(matches[-1]) if matches else 0
        else:
            label = '定位篮筐并检查机位'
    mtimes = []
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    last_activity = max(0., now-max(mtimes)) if mtimes else max(0., now-start)
    progress = {'label': label, 'completed_frames': completed, 'total_frames': total,
                'fps': fps, 'last_activity_seconds': round(last_activity, 1),
                'quiet': not done and last_activity >= 90}
    if completed is not None and total:
        progress['completed_frames'] = min(total, completed)
    return {'elapsed_seconds': round(elapsed, 1), 'progress': progress}
