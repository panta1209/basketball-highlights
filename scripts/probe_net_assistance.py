"""Net evidence after the eleven frozen first-three-minute static candidates."""
import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import cv2
import numpy as np
from basketball_poc.core import evaluate
from basketball_poc.media import read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/bilibili-3m-rim-02'
ANNOTATION = ROOT / 'data/annotations/bilibili-first-3m.automatic-evaluation.json'
PROFILE = {'version': 'candidate-net-assistance-v1', 'window_seconds': .8,
           'net_top_fraction': .15, 'ball_mask_scale': 1.2,
           'min_points': 6, 'max_forward_backward_pixels': 1.5,
           'noise_mad_multiplier': 3., 'minimum_displacement_hoop_widths': .03,
           'minimum_consecutive_frames': 3, 'pre_end_seconds': -.2,
           'post_start_seconds': -.1, 'lk_window': 11, 'lk_levels': 2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pixels_sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def lines(path):
    import json
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def exclusions(shape, boxes, origin, scale):
    mask = np.zeros(shape[:2], np.uint8)
    ox, oy = origin
    for box in boxes:
        if box['class_id'] != 0:
            continue
        x1, y1, x2, y2 = box['xyxy']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        rx, ry = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
        a, b = max(0, math.floor(cx-rx-ox)), max(0, math.floor(cy-ry-oy))
        c, d = min(shape[1], math.ceil(cx+rx-ox)), min(shape[0], math.ceil(cy+ry-oy))
        if a < c and b < d:
            mask[b:d, a:c] = 255
    return mask


def visible(points, mask):
    xy = np.rint(points.reshape(-1, 2)).astype(int)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
    good = np.zeros(len(xy), bool)
    good[inside] = mask[xy[inside, 1], xy[inside, 0]] == 0
    return good


def tracked(reference, current, points, excluded, profile):
    if points is None or not len(points):
        return np.empty((0, 2), np.float32)
    options = {'winSize': (profile['lk_window'],) * 2, 'maxLevel': profile['lk_levels'],
               'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}
    forward, valid, _ = cv2.calcOpticalFlowPyrLK(reference, current, points, None, **options)
    if forward is None:
        return np.empty((0, 2), np.float32)
    backward, reverse_valid, _ = cv2.calcOpticalFlowPyrLK(current, reference, forward, None, **options)
    if backward is None:
        return np.empty((0, 2), np.float32)
    error = np.linalg.norm(backward.reshape(-1, 2) - points.reshape(-1, 2), axis=1)
    good = valid.ravel().astype(bool) & reverse_valid.ravel().astype(bool)
    good &= error <= profile['max_forward_backward_pixels']
    good &= visible(forward, excluded)
    return (forward.reshape(-1, 2) - points.reshape(-1, 2))[good]


def decision(rows, profile):
    before = [row['displacement_p75_hoop_widths'] for row in rows
              if row['valid'] and row['relative_seconds'] <= profile['pre_end_seconds']]
    if len(before) < 6:
        return {'net_support': False, 'state': 'insufficient_evidence', 'reason': 'insufficient_pre_event_tracking',
                'threshold': None, 'maximum_consecutive_support_frames': 0}
    median = float(np.median(before))
    mad = float(np.median(np.abs(np.asarray(before) - median)))
    threshold = median + profile['noise_mad_multiplier'] * mad + profile['minimum_displacement_hoop_widths']
    consecutive = maximum = 0
    for row in rows:
        if row['relative_seconds'] < profile['post_start_seconds']:
            continue
        supported = row['valid'] and row['displacement_p75_hoop_widths'] >= threshold
        consecutive = consecutive + 1 if supported else 0
        maximum = max(maximum, consecutive)
    support = maximum >= profile['minimum_consecutive_frames']
    return {'net_support': support, 'state': 'net_interaction_support' if support else 'insufficient_evidence',
            'reason': 'sustained_local_displacement' if support else 'no_sustained_trackable_net_displacement',
            'pre_median': median, 'pre_mad': mad, 'threshold': threshold,
            'maximum_consecutive_support_frames': maximum}


def probe_case(case, frames, detections, fps, profile):
    x1, y1, x2, y2 = case['context_roi']
    hx1, hy1, hx2, hy2 = case['hoop_xyxy']
    width = hx2 - hx1
    reference = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    net = np.zeros(reference.shape, np.uint8)
    net[math.ceil(hy1 + (hy2-hy1)*profile['net_top_fraction'])-y1:math.ceil(hy2)-y1,
        math.floor(hx1)-x1:math.ceil(hx2)-x1] = 255
    control = np.full(reference.shape, 255, np.uint8)
    control[max(0, math.floor(hy1-.15*(hy2-hy1))-y1):math.ceil(hy2+.15*(hy2-hy1))-y1,
            max(0, math.floor(hx1-.15*width)-x1):math.ceil(hx2+.15*width)-x1] = 0
    excluded = exclusions(frames[0].shape, detections[case['frame_indices'][0]]['boxes'], (x1,y1), profile['ball_mask_scale'])
    net[excluded > 0] = control[excluded > 0] = 0
    points = [cv2.goodFeaturesToTrack(reference, maxCorners=100, qualityLevel=.01,
                                    minDistance=2, mask=mask, blockSize=3) for mask in (net, control)]
    rows = []
    for index, image in zip(case['frame_indices'], frames, strict=True):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        excluded = exclusions(image.shape, detections[index]['boxes'], (x1,y1), profile['ball_mask_scale'])
        net_flow, control_flow = [tracked(reference, grey, value, excluded, profile) for value in points]
        valid = min(len(net_flow), len(control_flow)) >= profile['min_points']
        shift = np.median(control_flow, axis=0) if len(control_flow) else np.array([0., 0.])
        magnitude = np.linalg.norm(net_flow - shift, axis=1) / width
        rows.append({'frame': index, 'relative_seconds': index/fps - case['time_seconds'],
                     'valid': bool(valid), 'visible_net_points': len(net_flow), 'control_points': len(control_flow),
                     'control_translation_pixels': shift.tolist(),
                     'displacement_p75_hoop_widths': float(np.percentile(magnitude, 75)) if valid else None})
    return {**case, 'initial_net_points': 0 if points[0] is None else len(points[0]),
            'initial_control_points': 0 if points[1] is None else len(points[1]),
            'rows': rows, **decision(rows, profile)}


def decode_cases(source, cases):
    wanted = {index for case in cases for index in case['frame_indices']}
    crops = {case['id']: [] for case in cases}
    capture = cv2.VideoCapture(str(source))
    require(capture.isOpened(), 'Cannot open frozen source')
    count = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if count in wanted:
            for case in cases:
                if count in case['frame_indices']:
                    x1,y1,x2,y2 = case['context_roi']
                    crops[case['id']].append(image[y1:y2, x1:x2].copy())
        count += 1
    capture.release()
    for case in cases:
        require(len(crops[case['id']]) == len(case['frame_indices']), 'Window decoding incomplete')
        crops[case['id']] = np.stack(crops[case['id']])
    return crops, count


def sheets(out, cases, crops, fps):
    pages = []
    for start in range(0, len(cases), 6):
        selected = cases[start:start+6]
        page = np.full((len(selected)*212, 6*184, 3), 245, np.uint8)
        for row, case in enumerate(selected):
            hx1,hy1,hx2,hy2 = case['hoop_xyxy']
            width, height = hx2-hx1, hy2-hy1
            cx1,cy1,_,_ = case['context_roi']
            for col, offset in enumerate((-.4,-.1,.1,.3,.5,.7)):
                index = min(range(len(case['frame_indices'])), key=lambda i: abs(case['frame_indices'][i]/fps-case['time_seconds']-offset))
                image = crops[case['id']][index]
                a,b = max(0,math.floor(hx1-.3*width)-cx1),max(0,math.floor(hy1-.3*height)-cy1)
                c,d = math.ceil(hx2+.3*width)-cx1,math.ceil(hy2+.3*height)-cy1
                tile = cv2.resize(image[b:d,a:c], (180,180), interpolation=cv2.INTER_NEAREST)
                y,x = row*212,col*184
                page[y+30:y+210,x:x+180] = tile
                label = f"{case['id']} {case['frame_indices'][index]/fps:.3f}s"
                cv2.putText(page,label,(x+2,y+18),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),1,cv2.LINE_AA)
        path = out/f'source-candidates-{start//6+1}.jpg'
        require(cv2.imwrite(str(path),page), 'Cannot save source page')
        pages.append({'path':str(path),'sha256':sha256(path),'candidate_ids':[case['id'] for case in selected],
                      'tiles':len(selected)*6,'native_pixels_resized_for_review':True})
    return pages


def experiment(out):
    out = Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    began = time.perf_counter()
    manifest, scene, events = [read_json(BASE/name) for name in ('manifest.json','scene.json','events.json')]
    require(manifest['status']=='completed' and manifest['candidate_count']==11 and len(events['events'])==11,
            'Expected exactly eleven frozen static candidates')
    source = Path(manifest['source']['path'])
    require(sha256(source)==manifest['source_sha256']==events['source_sha256'], 'Source identity mismatch')
    fps = manifest['source']['fps']
    cases=[]
    for event in events['events']:
        require(len(scene['hoops'])==1, 'This bounded experiment expects the single automatically discovered hoop')
        h = scene['hoops'][0]['xyxy']; x1,y1,x2,y2=h; w,height=x2-x1,y2-y1
        context=[max(0,math.floor(x1-w)),max(0,math.floor(y1-height)),
                 min(manifest['source']['width'],math.ceil(x2+w)),min(manifest['source']['height'],math.ceil(y2+height))]
        first=max(0,math.floor((event['time_seconds']-PROFILE['window_seconds'])*fps))
        last=min(manifest['processed_frames']-1,math.ceil((event['time_seconds']+PROFILE['window_seconds'])*fps))
        cases.append({'id':event['id'],'time_seconds':event['time_seconds'],'hoop_xyxy':h,
                      'context_roi':context,'frame_indices':list(range(first,last+1))})
    inputs={str(path.resolve()):sha256(path) for path in [source,BASE/'manifest.json',BASE/'scene.json',
            BASE/'events.json',BASE/'detections.jsonl',ANNOTATION]}
    snapshot=out/'probe-implementation.py';snapshot.write_bytes(Path(__file__).read_bytes())
    frozen={'status':'frozen','created_at_utc':datetime.now(timezone.utc).isoformat(),'inputs_sha256':inputs,
            'source':manifest['source'],'baseline':str(BASE),'cases':cases,'profile':PROFILE,
            'script_sha256':sha256(snapshot),'new_footage':False,'annotations_used_for_features':False,
            'candidate_detector_changed':False,'manual_coordinates':False,'training':False}
    write_json(out/'frozen.json',frozen)
    detections=lines(BASE/'detections.jsonl')
    crops,count=decode_cases(source,cases)
    require(count==manifest['processed_frames']==len(detections),'Full source frame count differs')
    records=[]
    for case in cases:
        images=crops[case['id']]
        np.savez_compressed(out/f"{case['id']}-source-crops.npz",frames=images)
        result=probe_case(case,images,detections,fps,PROFILE)
        result['decoded_crop_pixel_sha256']=[pixels_sha(image) for image in images]
        records.append(result)
        print(case['id'],round(case['time_seconds'],3),result['state'],result['maximum_consecutive_support_frames'],flush=True)
    # Labels are consumed only after every candidate's auxiliary features are finished.
    annotation=read_json(ANNOTATION)
    baseline=evaluate(events,annotation,.75)
    supported={**events,'events':[event for event,result in zip(events['events'],records,strict=True) if result['net_support']]}
    counterfactual=evaluate(supported,annotation,.75)
    require((baseline['true_positives'],baseline['false_positives'],baseline['false_negatives'])==(9,2,0),
            'Original static baseline changed')
    matches={row['prediction_seconds'] for row in baseline['matches']}
    for record in records:
        record['post_feature_evaluation_label']='made' if record['time_seconds'] in matches else 'miss'
    pages=sheets(out,records,crops,fps)
    require(all(sha256(path)==digest for path,digest in inputs.items()),'Input changed during experiment')
    write_json(out/'results.json',{'status':'completed','profile':PROFILE,'records':records,
               'baseline':baseline,'support_only_counterfactual':counterfactual,'source_pages':pages,
               'decoded_source_frames':count,'decoded_window_frames':sum(len(case['frame_indices']) for case in cases),
               'seconds':time.perf_counter()-began,'default_changed':False,'user_choices_changed':False,
               'semantic_limit':'Net displacement support is not validated physical goal confirmation; all unsupported cases remain uncertain.'})
    print('Baseline:',baseline['true_positives'],baseline['false_positives'],baseline['false_negatives'],flush=True)
    print('Net-support-only:',counterfactual['true_positives'],counterfactual['false_positives'],counterfactual['false_negatives'],flush=True)


def verify(out):
    out=Path(out).resolve(strict=True)
    require(not (out/'verification.json').exists(),'Verification already exists')
    frozen=read_json(out/'frozen.json');results=read_json(out/'results.json')
    require(sha256(__file__)==frozen['script_sha256'],'Implementation changed since freeze')
    require(all(sha256(path)==digest for path,digest in frozen['inputs_sha256'].items()),'Original evidence changed')
    crops,count=decode_cases(Path(frozen['source']['path']),frozen['cases'])
    require(count==results['decoded_source_frames']==5399,'Source decode did not match complete interval')
    detections=lines(BASE/'detections.jsonl')
    for record in results['records']:
        with np.load(out/f"{record['id']}-source-crops.npz",allow_pickle=False) as archive:
            require(np.array_equal(archive['frames'],crops[record['id']]),'Saved crop differs from native source pixels')
        require([pixels_sha(image) for image in crops[record['id']]]==record['decoded_crop_pixel_sha256'],'Crop pixels changed')
        actual=probe_case(record,crops[record['id']],detections,frozen['source']['fps'],frozen['profile'])
        require(actual['rows']==record['rows'] and actual['net_support']==record['net_support']
                and actual['threshold']==record['threshold'],'Auxiliary tracking replay differs')
        before=np.asarray([row['displacement_p75_hoop_widths'] for row in record['rows']
                           if row['valid'] and row['relative_seconds']<=-.2])
        if len(before)>=6:
            middle=float(np.median(before));threshold=middle+3*float(np.median(abs(before-middle)))+.03
            require(threshold==record['threshold'],'Independent threshold arithmetic differs')
    for page in results['source_pages']:
        require(sha256(page['path'])==page['sha256'],'Source page changed')
    events=read_json(BASE/'events.json');truth=read_json(ANNOTATION)
    for field,selected in [('baseline',events['events']),('support_only_counterfactual',
            [event for event,record in zip(events['events'],results['records'],strict=True) if record['net_support']])]:
        require(evaluate({**events,'events':selected},truth,.75)==results[field],'Evaluation differs')
    write_json(out/'verification.json',{'status':'passed','full_source_decoded_frames':count,
               'native_source_crop_pixels_equal':True,'auxiliary_tracking_replayed':True,
               'threshold_arithmetic_rechecked':True,'original_candidates_and_inputs_unchanged':True,
               'all_eleven_candidates_checked':True,'new_footage':False,'accuracy_outside_this_clip_claimed':False})
    print('Source/candidates, lossless crops, auxiliary replay and evaluation verified',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out');group.add_argument('--verify')
    args=parser.parse_args()
    cv2.setNumThreads(1);cv2.setRNGSeed(0)
    experiment(args.out) if args.out else verify(args.verify)
