"""Experimental shooter evidence and appearance suggestions, scoped to goal clips."""
from collections import OrderedDict
import math
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from .media import read_json, sha256, write_json
from .player_workflow import METHOD, fingerprint, replace_json

ROOT = Path(__file__).resolve().parents[2]
PERSON = ROOT/'models/ultralytics-assets-v8.3.0-yolo11n/yolo11n.pt'
BALL = ROOT/'third_party/josephattalla/e320817d0f87eceb4093c871de6d29d1adca0006/bball_model.pt'
HASHES = {PERSON: '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1',
          BALL: '40f3e596652a427ba290b3f72384e49aed12caf1a8ae41beaef4a8fffcf09fa3'}
SETTINGS = {'method': METHOD, 'person_imgsz': 640, 'ball_imgsz': 960,
            'person_conf': .30, 'ball_conf': .15, 'sample_fps': 10, 'batch': 4,
            'precision': 'float32', 'feature_version': 'torso-legs-hsv-gradient-v1'}


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, right-left)*max(0, bottom-top)
    return overlap/max(1, (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap)


def appearance(crop):
    image = cv2.resize(crop, (48, 96))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colors = []
    for start, end in ((18, 54), (54, 85)):
        values = hsv[start:end, 8:40]
        # Saturation/value retain dark/white clothes that hue alone cannot describe.
        for channel, bins, limit in ((0, 12, 180), (1, 8, 256), (2, 8, 256)):
            hist = cv2.calcHist([values], [channel], None, [bins], [0, limit]).reshape(-1)
            colors.extend((hist/max(float(hist.sum()), 1)).tolist())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture = []
    for start, end in ((18, 54), (54, 85)):
        region = gray[start:end, 8:40]
        gx = cv2.Sobel(region, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(region, cv2.CV_32F, 0, 1)
        magnitude, angle = cv2.cartToPolar(gx, gy)
        hist, _ = np.histogram(angle, bins=8, range=(0, 2*np.pi), weights=magnitude)
        texture.extend((hist/max(float(hist.sum()), 1)).tolist())
    return {'color': colors, 'texture': texture}


def feature_distance(a, b):
    # Hellinger distance on normalized histograms, with separate texture evidence.
    color = float(np.sqrt(np.mean((np.sqrt(a['color'])-np.sqrt(b['color']))**2)))
    texture = float(np.sqrt(np.mean((np.sqrt(a['texture'])-np.sqrt(b['texture']))**2)))
    return .7*color+.3*texture, color, texture


def _crop(image, box):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(x)) for x in box[:4]]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2-x1 < 10 or y2-y1 < 28:
        return None
    crop = image[y1:y2, x1:x2]
    scale = min(1., 256/max(crop.shape[:2]))
    return cv2.resize(crop, (max(1, round(crop.shape[1]*scale)), max(1, round(crop.shape[0]*scale)))) if scale < 1 else crop.copy()


def link_people(records, frames, tracks=None):
    """Short association only within this clip. No state survives this call."""
    tracks = [] if tracks is None else tracks
    for record, image in zip(records, frames):
        used = set()
        for box in record['people']:
            crop = _crop(image, box)
            if crop is None:
                continue
            feature = appearance(crop)
            options = []
            for index, track in enumerate(tracks):
                if index in used or record['time']-track['last']['time'] > .35:
                    continue
                previous = track['last']['box']
                overlap = iou(previous, box)
                distance = math.hypot((previous[0]+previous[2]-box[0]-box[2])/2,
                                      (previous[1]+previous[3]-box[1]-box[3])/2)/max(1, box[3]-box[1])
                color = feature_distance(feature, track['views'][0]['feature'])[0]
                if (overlap >= .15 or distance <= .35) and color < .16:
                    options.append((overlap-distance*.2-color, index))
            if options:
                index = max(options)[1]
            elif len(tracks) < 80:
                index = len(tracks)
                tracks.append({'id': index, 'observations': [], 'views': []})
            else:
                continue
            used.add(index)
            observation = {'time': record['time'], 'frame': record['frame'], 'box': box[:4]}
            record.setdefault('person_tracks', []).append({**observation, 'track': index})
            track = tracks[index]
            track['last'] = observation
            track['observations'].append(observation)
            quality = float(cv2.Laplacian(crop, cv2.CV_64F).var())*min(crop.shape[0], 150)/150
            if all(abs(v['time']-record['time']) >= .25 for v in track['views']):
                track['views'].append({**observation, 'crop': crop, 'feature': feature, 'quality': quality})
                track['views'].sort(key=lambda v: v['quality'], reverse=True)
                del track['views'][3:]
    return tracks


def shooter_evidence(records, event, width, seed_before=.55, max_gap=.36, min_points=6,
                     horizontal_margin=.5):
    """Follow a ball near this goal backwards, then look for visible departure."""
    time_goal = event['time_seconds']
    hoop = event.get('hoop', {})
    if not all(key in hoop for key in ('x', 'y', 'w', 'h')):
        return None, [], '缺少可对应的篮筐位置，需人工指定。'
    center = (hoop['x']+hoop['w']/2, hoop['y']+hoop['h']/2)
    # Prefer a hoop detected on the original frame; historical coordinates may be corrected.
    seeds = []
    for index, record in enumerate(records):
        if not time_goal-seed_before <= record['time'] <= time_goal+.15:
            continue
        hoops = record['hoops']
        local_center = center
        if hoops:
            target = min(hoops, key=lambda b: math.hypot((b[0]+b[2])/2-center[0], (b[1]+b[3])/2-center[1]))
            if math.hypot((target[0]+target[2])/2-center[0], (target[1]+target[3])/2-center[1]) < width*.12:
                local_center = ((target[0]+target[2])/2, (target[1]+target[3])/2)
        for ball in record['balls']:
            x, y = (ball[0]+ball[2])/2, (ball[1]+ball[3])/2
            distance = math.hypot(x-local_center[0], y-local_center[1])
            if distance < max(hoop['w']*3, width*.06):
                seeds.append((distance/max(hoop['w'], 1)+abs(record['time']-time_goal)*3,
                              index, {'x': x, 'y': y, 'time': record['time'], 'frame': record['frame']}))
    if not seeds:
        return None, [], '片段内未找到与本次进球连续对应的球路。'
    _, index, seed = min(seeds, key=lambda value: value[0])
    path = [seed]
    for record in reversed(records[:index]):
        last = path[-1]
        gap = last['time']-record['time']
        if gap > max_gap:
            break
        if record['time'] < time_goal-3.7:
            break
        vx = vy = 0.
        if len(path) > 1:
            dt = path[-2]['time']-last['time']
            vx, vy = (path[-2]['x']-last['x'])/dt, (path[-2]['y']-last['y'])/dt
        expected = (last['x']-vx*gap, last['y']-vy*gap)
        options = []
        for ball in record['balls']:
            x, y = (ball[0]+ball[2])/2, (ball[1]+ball[3])/2
            distance = math.hypot(x-expected[0], y-expected[1])
            if distance <= width*(.022+gap*.28):
                options.append((distance, {'x': x, 'y': y, 'time': record['time'], 'frame': record['frame']}))
        options.sort(key=lambda value: value[0])
        if len(options) > 1 and options[1][0]-options[0][0] < width*.008:
            break  # Multiple balls cannot be distinguished.
        if options:
            path.append(options[0][1])
    path.reverse()
    if len(path) < min_points or path[-1]['time']-path[0]['time'] < .5:
        return None, path, '进筐前球路过短或被遮挡，未确定离手人。'
    by_frame = {record['frame']: record for record in records}
    scores = {}
    evidence = {}
    for point in path:
        if not time_goal-3.5 <= point['time'] <= time_goal-.45:
            continue
        future = [p for p in path if .15 <= p['time']-point['time'] <= .45]
        if len(future) < 2:
            continue
        later = future[-1]
        for person in by_frame[point['frame']].get('person_tracks', []):
            x1, y1, x2, y2 = person['box']
            height, breadth = y2-y1, x2-x1
            if not (x1-breadth*horizontal_margin <= point['x'] <= x2+breadth*horizontal_margin
                    and y1-height*.25 <= point['y'] <= y1+height*.55):
                continue
            rise = (point['y']-later['y'])/height
            departure = math.hypot(later['x']-point['x'], later['y']-point['y'])/height
            if rise < .12 or departure < .25:
                continue
            # Upward departure must be connected to this goal, not a nearby loose ball.
            proximity = abs(point['x']-(x1+x2)/2)/max(breadth, 1)
            score = min(rise, 1)+min(departure, 1)-proximity*.5
            identifier = person['track']
            if score > scores.get(identifier, -math.inf):
                scores[identifier] = score
                evidence[identifier] = {'release_time': point['time'], 'release_frame': point['frame'],
                    'person_box': person['box'], 'ball': point, 'departure': later, 'score': score}
    ranked = sorted(scores, key=scores.get, reverse=True)
    if not ranked:
        return None, path, '未看到可区分的球离手过程，请回看后指定。'
    if len(ranked) > 1 and scores[ranked[0]]-scores[ranked[1]] < .35:
        return None, path, '出手位置有多个人物，无法可靠区分。'
    return {'track': ranked[0], **evidence[ranked[0]]}, path, '已找到离手到进筐的球路建议，请核对。'


def grouping(events, suggest_threshold=.12):
    players, assignments, suggestions = [], {}, []
    profiles = []
    for event in events:
        candidate = next((c for c in event['candidates'] if c['id'] == event['suggested_candidate_id']), None)
        if candidate is None:
            assignments[event['id']] = None
            continue
        player = {'id': 'p-'+uuid.uuid4().hex[:12], 'name': f'人物 {len(players)+1}', 'representative': candidate['id']}
        # Appearance-only matches stay explicit merge suggestions, not confirmed identities.
        for previous, other in profiles:
            if any(a['frame'] == b['frame'] and iou(a['box'], b['box']) < .1
                   for a in candidate['views'] for b in other['views']
                   if 'frame' in a and 'frame' in b):
                continue
            distances = [feature_distance(a['feature'], b['feature']) for a in candidate['views'] for b in other['views']]
            if distances:
                score, color, texture = sorted(distances)[len(distances)//2]
                if score < suggest_threshold and texture < .15:
                    suggestions.append({'left': previous['id'], 'right': player['id'],
                        'distance': score, 'reason': '外观相似，请核对是否为同一人'})
        profiles.append((player, candidate))
        players.append(player)
        assignments[event['id']] = player['id']
    return players, assignments, suggestions


def analyze_clips(snapshot_path, folder, device='auto', trial_settings=None):
    import torch
    from ultralytics import YOLO
    began = time.perf_counter()
    folder = Path(folder)
    settings = {**SETTINGS, **(trial_settings or {})}
    snapshot = read_json(snapshot_path)
    value = {key: value for key, value in snapshot.items() if key != 'snapshot_sha256'}
    if fingerprint(value) != snapshot['snapshot_sha256'] or sha256(snapshot['source']['path']) != snapshot['source_sha256']:
        raise ValueError('Confirmed selection or original media changed')
    for name, digest in snapshot.get('method_sha256', {}).items():
        path = (ROOT/name).resolve(strict=True)
        if not path.is_relative_to(ROOT) or sha256(path) != digest:
            raise ValueError('人物算法版本已变化，请重新匹配。')
    for path, digest in HASHES.items():
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'本地模型缺失或校验不符：{path.name}')
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda:0' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = [YOLO(str(PERSON)), YOLO(str(BALL))]
    cache = OrderedDict()
    cap = cv2.VideoCapture(snapshot['source']['path'])
    if not cap.isOpened():
        raise ValueError('无法读取原视频。')
    fps = snapshot['source']['fps']
    stride = max(1, round(fps/settings['sample_fps']))
    output, decoded, inferred, reused, ranges = [], 0, 0, 0, []
    images = folder/'images'
    images.mkdir(exist_ok=True)
    fallback = False
    def predict(batch):
        nonlocal device, fallback
        try:
            people = models[0].predict(batch, device=device, imgsz=settings['person_imgsz'],
                                       conf=settings['person_conf'], classes=[0],
                                       verbose=False, quantize=32)
            if settings.get('ball_view', 'full') == 'court':
                height, width = batch[0].shape[:2]
                left, top = 0, round(height*.10)
                right, bottom = round(width*.86), round(height*.85)
                ball_inputs = [frame[top:bottom, left:right] for frame in batch]
            elif settings.get('ball_view', 'full') == 'full':
                left = top = 0
                ball_inputs = batch
            else:
                raise ValueError('Unknown ball view')
            balls = models[1].predict(ball_inputs, device=device, imgsz=settings['ball_imgsz'],
                                      conf=settings['ball_conf'], classes=[0, 1],
                                      verbose=False, quantize=32)
        except torch.OutOfMemoryError:
            if device == 'cpu':
                raise
            device, fallback = 'cpu', True
            torch.cuda.empty_cache()
            for model in models:
                model.to('cpu')
            return predict(batch)
        result = []
        for person, ball in zip(people, balls):
            boxes = ball.boxes.data.cpu().tolist()
            for box in boxes:
                box[0] += left
                box[2] += left
                box[1] += top
                box[3] += top
            result.append((person.boxes.data.cpu().tolist(), boxes))
        return result
    try:
        with (folder/'observations.jsonl').open('x', encoding='utf-8') as log:
            for ordinal, event in enumerate(snapshot['selected_events']['events'], 1):
                start, end = event['segment_start_seconds'], event['segment_end_seconds']
                first, stop = math.ceil(start*fps), math.ceil(end*fps)
                ranges.append({'id': event['id'], 'start': start, 'end': end, 'first_frame': first, 'stop_frame': stop})
                replace_json(folder/'progress.json', {'done': ordinal-1, 'total': len(snapshot['selected_events']['events'])})
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
                records, batch_frames, batch_records, tracks = [], [], [], []
                def consume_batch():
                    nonlocal inferred, reused
                    missing = [i for i, record in enumerate(batch_records) if record['frame'] not in cache]
                    if missing:
                        results = predict([batch_frames[i] for i in missing])
                        inferred += len(missing)
                        for i, (people, balls) in zip(missing, results):
                            cache[batch_records[i]['frame']] = {'people': people,
                                'balls': [b for b in balls if int(b[5]) == 0],
                                'hoops': [b for b in balls if int(b[5]) == 1]}
                    reused += len(batch_records)-len(missing)
                    for record in batch_records:
                        record.update(cache[record['frame']])
                    link_people(batch_records, batch_frames, tracks)
                    records.extend(batch_records)
                    batch_frames.clear()
                    batch_records.clear()
                    while len(cache) > 128:
                        cache.popitem(last=False)
                # Retain at most four decoded frames, not a whole clip/movie.
                for index in range(first, stop):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError('进球片段解码不完整。')
                    decoded += 1
                    if index % stride != 0:
                        continue
                    actual = round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1
                    if actual != index:
                        raise ValueError('原视频帧索引发生变化。')
                    batch_frames.append(frame)
                    batch_records.append({'frame': index, 'time': index/fps})
                    if len(batch_frames) == settings['batch']:
                        consume_batch()
                if batch_records:
                    consume_batch()
                evidence, path, reason = shooter_evidence(records, event, snapshot['source']['width'],
                    seed_before=settings.get('seed_before', .55), max_gap=settings.get('max_gap', .36),
                    min_points=settings.get('min_points', 6),
                    horizontal_margin=settings.get('horizontal_margin', .5))
                valid = [t for t in tracks if len(t['observations']) >= 2 and t['views']]
                valid.sort(key=lambda t: (evidence is not None and t['id'] == evidence['track'],
                                           len(t['observations'])), reverse=True)
                candidates = []
                for track in valid[:24]:
                    candidate_id = f'e{ordinal:04d}-t{track["id"]:04d}'
                    selected_views = track['views']
                    if (settings.get('release_local_views') and evidence
                            and track['id'] == evidence['track']):
                        nearby = [observation for observation in track['observations']
                                  if evidence['release_time']-.1 <= observation['time']
                                  <= evidence['release_time']+.5]
                        chosen = []
                        for target_time in (evidence['release_time']+.3,
                                            evidence['release_time']+.1,
                                            evidence['release_time']+.5):
                            options = [observation for observation in nearby
                                       if all(abs(observation['time']-item['time']) >= .1
                                              for item in chosen)]
                            if options:
                                chosen.append(min(options,
                                    key=lambda item: abs(item['time']-target_time)))
                        local_views = []
                        for observation in chosen:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, observation['frame'])
                            ok, image = cap.read()
                            decoded += 1
                            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1 != observation['frame']:
                                raise ValueError('无法复核离手人物原帧。')
                            crop = _crop(image, observation['box'])
                            if crop is None:
                                continue
                            local_views.append({**observation, 'crop': crop,
                                'feature': appearance(crop),
                                'quality': float(cv2.Laplacian(crop, cv2.CV_64F).var()),
                                'source': 'release_near'})
                        if local_views:
                            selected_views = local_views
                    views = []
                    for v, view in enumerate(selected_views):
                        name = f'{candidate_id}-{v}.jpg'
                        if not cv2.imwrite(str(images/name), view['crop']):
                            raise ValueError('无法保存人物代表图。')
                        views.append({key: value for key, value in view.items() if key != 'crop'})
                        views[-1]['url'] = f'players/{folder.name}/images/{name}'
                    candidates.append({'id': candidate_id, 'track': track['id'],
                        'thumbnail_url': views[0]['url'], 'views': views,
                        'observations': len(track['observations'])})
                if settings.get('review_ball_origin') and evidence is None and path:
                    point = path[0]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, point['frame'])
                    ok, image = cap.read()
                    decoded += 1
                    if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES))-1 != point['frame']:
                        raise ValueError('无法复核球路起点原帧。')
                    height, width = image.shape[:2]
                    direction = 1 if len(path) < 2 or path[1]['x'] >= point['x'] else -1
                    left_extent, right_extent = ((.09, .15) if direction > 0 else (.15, .09))
                    left = max(0, round(point['x']-width*left_extent))
                    right = min(width, round(point['x']+width*right_extent))
                    top = max(0, round(point['y']-height*.11))
                    bottom = min(height, round(point['y']+height*.43))
                    region = image[top:bottom, left:right]
                    if region.size:
                        found = models[0].predict(region, device=device, imgsz=960, conf=.20,
                                                  classes=[0], verbose=False)[0]
                        nearby = []
                        for box in found.boxes.data.cpu().tolist():
                            box[0] += left
                            box[2] += left
                            box[1] += top
                            box[3] += top
                            x1, y1, x2, y2 = box[:4]
                            breadth, stature = x2-x1, y2-y1
                            if (x1-breadth*.1 <= point['x'] <= x2+breadth*.1
                                    and y1-stature*.3 <= point['y'] <= y1+stature*.55):
                                distance = (abs(point['x']-(x1+x2)/2)/max(breadth, 1)
                                    + abs(point['y']-(y1+stature*.15))/max(stature, 1))
                                nearby.append((distance, box))
                        for priority, (_, box) in enumerate(sorted(nearby, key=lambda item: item[0])[:2]):
                            crop = _crop(image, box)
                            if crop is None:
                                continue
                            candidate_id = f'e{ordinal:04d}-review-{priority}'
                            name = f'{candidate_id}-0.jpg'
                            if not cv2.imwrite(str(images/name), crop):
                                raise ValueError('无法保存待核对人物图。')
                            x1, y1, x2, y2 = box[:4]
                            margin_x, margin_y = width*.045, height*.075
                            scene_left = max(0, round(min(x1, point['x'])-margin_x))
                            scene_right = min(width, round(max(x2, point['x'])+margin_x))
                            scene_top = max(0, round(min(y1, point['y'])-margin_y))
                            scene_bottom = min(height, round(max(y2, point['y'])+margin_y))
                            scene = image[scene_top:scene_bottom, scene_left:scene_right].copy()
                            cv2.rectangle(scene, (round(x1)-scene_left, round(y1)-scene_top),
                                          (round(x2)-scene_left, round(y2)-scene_top), (40, 220, 40), 2)
                            cv2.circle(scene, (round(point['x'])-scene_left,
                                               round(point['y'])-scene_top), 6, (0, 220, 255), 2)
                            context_name = f'{candidate_id}-context.jpg'
                            if not cv2.imwrite(str(images/context_name), scene):
                                raise ValueError('无法保存待核对场景图。')
                            view = {'time': point['time'], 'frame': point['frame'],
                                'box': box[:4], 'feature': appearance(crop),
                                'source': 'ball_origin_zoom',
                                'url': f'players/{folder.name}/images/{name}'}
                            candidates.insert(priority, {'id': candidate_id,
                                'track': 80+priority, 'thumbnail_url': view['url'],
                                'review_context_url': f'players/{folder.name}/images/{context_name}',
                                'views': [view], 'observations': 1,
                                'review_priority': True})
                suggested = next((c['id'] for c in candidates if evidence and c['track'] == evidence['track']), None)
                output.append({'id': event['id'], 'time_seconds': event['time_seconds'],
                    'start': start, 'end': end, 'suggested_candidate_id': suggested,
                    'evidence': evidence, 'ball_path': path, 'reason': reason, 'candidates': candidates})
                for record in records:
                    log.write(__import__('json').dumps({'event_id': event['id'], **record})+'\n')
                del tracks
    finally:
        cap.release()
    players, assignments, suggestions = grouping(output,
        suggest_threshold=settings.get('merge_suggest_threshold', .12))
    result = {'version': METHOD, 'snapshot_sha256': snapshot['snapshot_sha256'],
        'method_sha256': snapshot.get('method_sha256', {}),
        'settings': settings, 'models_sha256': {str(path.relative_to(ROOT)): digest for path, digest in HASHES.items()},
        'device': device, 'cpu_fallback_after_oom': fallback, 'events': output, 'players': players,
        'assignments': assignments, 'merge_suggestions': suggestions, 'ranges': ranges,
        'decoded_frames': decoded, 'inferred_frames': inferred, 'reused_frames': reused,
        'elapsed_seconds': time.perf_counter()-began, 'generated_video_bytes': 0,
        'scope': 'Experimental suggestions; no human shooter or identity accuracy acceptance',
        'automatic_goal_metrics_modified': False}
    write_json(folder/'automatic.json', result)
    return result
