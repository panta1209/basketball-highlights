"""Automatic scene + shared profile + josephattalla events, without manual calibration."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import types

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from basketball_poc.media import inspect_video, read_json, sha256, write_json
from basketball_poc.scene import stable_hoops, roi_for_hoop, merge_boxes, seconds_to_frames
from basketball_poc.temporal import TemporalEventsMixin
from basketball_poc.upstream_adapter import adapt_source

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'e320817d0f87eceb4093c871de6d29d1adca0006'
UPSTREAM = ROOT/'third_party/josephattalla'/COMMIT
DEFAULT_MODEL = UPSTREAM/'bball_model.pt'


def discover_scene(video, info, model, profile, scene_mode='nearest-rank-v1'):
    capture = cv2.VideoCapture(str(video))
    samples = []
    frame_id, next_sample = 0, 0.0
    try:
        while frame_id/info['fps'] < min(profile['scout_seconds'],info['duration_seconds']):
            ok,frame = capture.read()
            if not ok:
                break
            seconds = frame_id/info['fps']
            if seconds + .5/info['fps'] >= next_sample:
                result = model.predict(frame,classes=[1],conf=profile['scout_confidence'],
                                       imgsz=profile['imgsz'],device='cpu',verbose=False)[0]
                samples.append({'frame':frame_id,'time_seconds':seconds,
                                'boxes':[{'xyxy':b.xyxy[0].tolist(),'confidence':float(b.conf[0])} for b in result.boxes]})
                next_sample += profile['scout_interval_seconds']
                if len(samples) % 5 == 0:
                    print(f'scout: {len(samples)} samples through {seconds:.1f}s',flush=True)
            frame_id += 1
    finally:
        capture.release()
    hoops = stable_hoops(samples,profile,scene_mode)
    for hoop in hoops:
        hoop['roi'] = roi_for_hoop(hoop['xyxy'],info['width'],info['height'],profile)
    return {'status':'ready' if hoops else 'unsupported','hoops':hoops,'samples':samples,
            'reason':None if hoops else 'No repeatedly observed stable hoop; automatic analysis unavailable'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video')
    parser.add_argument('--out',required=True)
    parser.add_argument('--ball-model',default=str(DEFAULT_MODEL))
    parser.add_argument('--scene-mode',choices=['nearest-rank-v1','linear-p95-v1'],default='nearest-rank-v1',
                        help='Experimental small-sample jitter quantile; default scene estimator unchanged')
    parser.add_argument('--scene-cache',help='Shared automatic scene evidence from another run of this exact video/profile')
    parser.add_argument('--detections-cache',help='Completed automatic run to replay without rerunning neural inference')
    parser.add_argument('--event-mode',choices=['native-v1','association-v2','rim-v2'],default='native-v1',
                        help='Controlled experiments; rim-v2 uses hoop-box top as an unverified rim proxy')
    args = parser.parse_args()
    profile_path = ROOT/'configs/automatic-ball-scale320.json'
    profile = read_json(profile_path)
    info = inspect_video(args.video)
    if abs(info['fps']-info['nominal_fps']) > .01:
        raise ValueError('Normalize variable frame rate before analysis')
    out = Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    # Preserve implementations, so later fixes cannot erase a run's provenance.
    (out/'geometry-implementation.py').write_bytes((ROOT/'src/basketball_poc/scene.py').read_bytes())
    (out/'benchmark-implementation.py').write_bytes(Path(__file__).read_bytes())
    for name in ('temporal','upstream_adapter'):
        (out/f'{name}-implementation.py').write_bytes((ROOT/f'src/basketball_poc/{name}.py').read_bytes())
    ball_path = Path(args.ball_model).resolve(strict=True)
    identity = {'source_sha256':sha256(args.video),'profile_sha256':sha256(profile_path),
                'scout_model_sha256':sha256(DEFAULT_MODEL),'dimensions':[info['width'],info['height']],
                'fps':info['fps'],'duration_seconds':info['duration_seconds']}
    # Keep historical native cache identities unchanged; experimental caches are distinct.
    if args.scene_mode != 'nearest-rank-v1':
        identity['scene_mode'] = args.scene_mode
    manifest = {'status':'preparing','created_at_utc':datetime.now(timezone.utc).isoformat(),
                'source':info,**identity,'profile':profile,'profile_path':str(profile_path),
                'ball_model_path':str(ball_path),'model_sha256':sha256(ball_path),
                'scout_model_path':str(DEFAULT_MODEL),'backend':'josephattalla-ball-scale320-experiment','commit':COMMIT,
                'upstream_provenance':read_json(UPSTREAM/'provenance.json'),
                'adapter_sha256':sha256(__file__),'geometry_implementation_sha256':sha256(ROOT/'src/basketball_poc/scene.py'),
                'temporal_implementation_sha256':sha256(ROOT/'src/basketball_poc/temporal.py'),
                'source_adapter_sha256':sha256(ROOT/'src/basketball_poc/upstream_adapter.py'),
                'event_mode':args.event_mode,'scene_mode':args.scene_mode,
                'interval':{'start_seconds':0,'end_seconds':info['duration_seconds']},
                'device':'cpu','versions':{n:version(n) for n in ('torch','ultralytics','opencv-python','numpy','dill')},
                'annotations_used_for_inference':False,'manual_calibration_used':False,
                'limitations':['One fixed camera; no later pan/cut handling','Hoop boxes describe net/hoop projection, not a precise 3D rim',
                               'Native line-crossing score remains a candidate; viewpoint generalization unverified']}
    began = time.perf_counter()
    processed,events,attempts = 0,[],[]
    detection_log = attempt_log = association_log = None
    try:
        cached_rows = None
        if args.detections_cache:
            cache = Path(args.detections_cache)
            cached_manifest = read_json(cache/'manifest.json')
            if (cached_manifest['status'] != 'completed'
                    or cached_manifest.get('scene_mode','nearest-rank-v1') != args.scene_mode
                    or any(cached_manifest.get(k) != v for k,v in identity.items())):
                raise ValueError('Detection cache must be completed with matching video/profile/scout identity')
            if cached_manifest['model_sha256'] != manifest['model_sha256']:
                raise ValueError('Detection cache ball-model mismatch')
            cached_rows = [json.loads(line) for line in (cache/'detections.jsonl').read_text().splitlines()]
            count = round(info['duration_seconds']*info['fps'])
            if len(cached_rows) != count or [r['frame'] for r in cached_rows] != list(range(count)):
                raise ValueError('Detection cache must cover all source frames exactly once')
            if any(abs(r['time_seconds']-r['frame']/info['fps']) > max(.005,.1/info['fps']) for r in cached_rows):
                raise ValueError('Detection cache timestamp mismatch')
            args.scene_cache = args.scene_cache or str(cache/'scene.json')
            manifest['detection_cache'] = {'path':str(cache.resolve()),'manifest_sha256':sha256(cache/'manifest.json'),
                                         'detections_sha256':sha256(cache/'detections.jsonl')}
        if args.scene_cache:
            scene = read_json(args.scene_cache)
            if scene.get('identity') != identity:
                raise ValueError('Automatic scene cache identity/profile/model mismatch')
            manifest['scene_cache_path'] = str(Path(args.scene_cache).resolve())
            manifest['scene_cache_sha256'] = sha256(args.scene_cache)
        else:
            scene = {**discover_scene(args.video,info,YOLO(str(DEFAULT_MODEL)),profile,args.scene_mode),'identity':identity}
        write_json(out/'scene.json',scene)
        manifest['scene_sha256'] = sha256(out/'scene.json')
        if scene['status'] != 'ready':
            manifest.update({'status':'unsupported','reason':scene['reason']})
            print(scene['reason'],flush=True)
            return
        print(f'automatic scene: {len(scene["hoops"])} stable hoops; {len(scene["hoops"])} relative regions',flush=True)
        if args.detections_cache and sha256(args.scene_cache) != cached_manifest['scene_sha256']:
            raise ValueError('Detection cache automatic-scene mismatch')
        model = YOLO(str(ball_path)) if cached_rows is None else None
        names = model.names if model else {int(k):v for k,v in cached_manifest['model_classes'].items()}
        if len(names) != 2 or names[0].lower() not in ('basketball','ball'):
            raise ValueError(f'Unexpected model classes: {names}')
        manifest['model_classes'] = names
        detection_log = (out/'detections.jsonl').open('x',encoding='utf-8')
        attempt_log = (out/'attempts.jsonl').open('x',encoding='utf-8')
        association_log = (out/'associations.jsonl').open('x',encoding='utf-8')

        class RoutedModel:
            def __init__(self,detector):
                self.detector = detector
                self.names = names

            def predict(self,frame,**kwargs):
                nonlocal processed
                frame_id = self.detector.frame_count-1
                actual_time = self.detector.source.get(cv2.CAP_PROP_POS_MSEC)/1000
                if abs(actual_time-frame_id/info['fps']) > max(.005,.1/info['fps']):
                    raise ValueError('Decoded source timestamps differ from declared constant frame rate')
                if frame.shape[:2] != (info['height'],info['width']):
                    raise ValueError('Source dimensions changed')
                if cached_rows is not None:
                    boxes = cached_rows[frame_id]['boxes']
                else:
                    balls = []
                    for hoop in scene['hoops']:
                        x1,y1,x2,y2 = hoop['roi']
                        result = model.predict(frame[y1:y2,x1:x2],classes=[0],conf=profile['inference_confidence'],
                                               imgsz=profile['ball_imgsz'],device='cpu',verbose=False)[0]
                        for b in result.boxes:
                            a,c,d,e = b.xyxy[0].tolist()
                            balls.append({'xyxy':[a+x1,c+y1,d+x1,e+y1],'class_id':0,'confidence':float(b.conf[0])})
                    hoops = [{'xyxy':h['xyxy'],'confidence':h['confidence'],'class_id':1,
                              'scene_hoop_id':h['id'],'source':'automatic_scout_consensus'} for h in scene['hoops']]
                    boxes = hoops + merge_boxes(balls,profile['merge_iou'])
                detection_log.write(json.dumps({'frame':frame_id,'time_seconds':actual_time,'boxes':boxes},allow_nan=False)+'\n')
                processed += 1
                if processed % 300 == 0:
                    detection_log.flush()
                    print(f'{processed} frames, {actual_time:.1f}s, {len(events)} candidates',flush=True)
                array = np.array([[*b['xyxy'],b['confidence'],b['class_id']] for b in boxes],dtype=np.float32).reshape(-1,6)
                yield Results(frame,str(args.video),names,boxes=array)

        source = adapt_source((UPSTREAM/'shot_detector.py').read_text(encoding='utf-8'))
        (out/'adapted-shot_detector.py').write_text(source,encoding='utf-8')
        manifest['adapted_source_sha256'] = sha256(out/'adapted-shot_detector.py')
        manifest['adapter_changes'] = ['Automatic full-frame hoop consensus and ball-only ROI inference with original coordinates',
                                       'Shared profile; native accept confidence and size-distance gates unchanged',
                                       'Track TTL and history measured in seconds; no fixed 20/30-frame assumptions',
                                       'Vertical-safe equivalent interpolation, verbose initialization and decision audit']
        if args.event_mode != 'native-v1':
            manifest['adapter_changes'].append('One observation per ball track per source frame; return actual new track ID')
        if args.event_mode == 'rim-v2':
            manifest['adapter_changes'].append('Adjacent chronological crossings at hoop-box top proxy; whole-ball below confirmation independent of net length')
            manifest['limitations'].append('Hoop-box top is not a precise rim detector; proxy geometry and projection ambiguity remain unverified')
        sys.path.insert(0,str(UPSTREAM))
        module = types.ModuleType('automatic_upstream')
        module.__dict__.update({'_route':RoutedModel,'_ttl_frames':seconds_to_frames(profile['track_ttl_seconds'],info['fps']),
                               '_history_frames':seconds_to_frames(profile['history_seconds'],info['fps'])})
        exec(compile(source,str(out/'adapted-shot_detector.py'),'exec'),module.__dict__)

        base = module.Shot_Detector if args.event_mode == 'native-v1' else type('TemporalDetector',(TemporalEventsMixin,module.Shot_Detector),{})

        class LoggedDetector(base):
            event_mode = args.event_mode
            source_fps = info['fps']
            track_ttl_seconds = profile['track_ttl_seconds']
            _ball_type = module.DetectedBall

            def _audit_association(self,ball_id,ball):
                record = {'frame':ball.frame-1,'time_seconds':(ball.frame-1)/info['fps'],
                          'ball_id':ball_id,**{key:float(getattr(ball,key)) for key in ('x','y','w','h','conf')}}
                association_log.write(json.dumps(record,allow_nan=False)+'\n')

            def _audit_temporal(self,ball_id,hoop_id,hoop,evidence):
                record = {'made':evidence['decision']=='candidate','time_seconds':(self.frame_count-1)/info['fps'],
                          'confirmed_at_seconds':(self.frame_count-1)/info['fps'],'ball_id':ball_id,'hoop_id':hoop_id,
                          'hoop':{key:float(getattr(hoop,key)) for key in ('x','y','w','h')},**evidence}
                for key in ('above','below','below_confirmation'):
                    if key in record:
                        record[key] = {attr:float(getattr(record[key],attr)) for attr in ('x','y','w','h','frame','conf')}
                self._save_record(record)

            def _save_record(self,record):
                attempts.append(record)
                attempt_log.write(json.dumps(record,allow_nan=False)+'\n')
                attempt_log.flush()
                if record['made']:
                    events.append({'id':f'goal-{len(events)+1:04d}','status':'candidate',**record})
                    print(f'make candidate at {record["time_seconds"]:.3f}s, hoop {record["hoop_id"]}',flush=True)

            def _audit_decision(self,ball_id,hoop_id,hoop,above,below,x_pred,made):
                ratio = (hoop.y-above.y)/(below.y-above.y)
                crossing = (above.frame-1+ratio*(below.frame-above.frame))/info['fps']
                box = {key:float(getattr(hoop,key)) for key in ('x','y','w','h')}
                def point(obj):
                    return {key:float(getattr(obj,key)) for key in ('x','y','w','h','frame','conf')}
                record = {'made':bool(made),'time_seconds':float(crossing),
                          'confirmed_at_seconds':(self.frame_count-1)/info['fps'],'ball_id':ball_id,'hoop_id':hoop_id,
                          'hoop':box,'above':point(above),'below':point(below),'predicted_x':float(x_pred)}
                self._save_record(record)

        detector = LoggedDetector(source=info['path'],display_object_info=False,verbose=False)
        detector.run()
        if processed != round(info['duration_seconds']*info['fps']):
            raise ValueError('Incomplete source interval')
        write_json(out/'events.json',{'status':'candidates','source_sha256':identity['source_sha256'],
                                    'interval':manifest['interval'],'events':events})
        detection_log.flush()
        manifest.update({'status':'completed','candidate_count':len(events),'decision_record_count':len(attempts),
                         'algorithm_attempt_count':detector.attempts,
                         'uncertain_count':sum(a.get('decision')=='uncertain' for a in attempts),
                         'detections_sha256':sha256(out/'detections.jsonl')})
    except Exception as error:
        manifest.update({'status':'failed','error':str(error)})
        raise
    finally:
        if detection_log:
            detection_log.close()
        if attempt_log:
            attempt_log.close()
        if association_log:
            association_log.close()
        manifest['processed_frames'] = processed
        manifest['processing_seconds'] = time.perf_counter()-began
        write_json(out/'manifest.json',manifest)
    print(json.dumps({key:manifest[key] for key in ('status','processed_frames','candidate_count','processing_seconds')},indent=2))


if __name__ == '__main__':
    main()
