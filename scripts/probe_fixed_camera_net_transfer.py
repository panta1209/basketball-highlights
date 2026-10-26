"""Apply the frozen v2 net helper to a fresh automatic run of a new source."""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import time
import probe_net_assistance as net

ROOT = Path(__file__).resolve().parents[1]

def check_inputs(base, freeze):
    frozen = net.read_json(freeze)
    for path,digest in frozen['inputs_sha256'].items():
        net.require(net.sha256(path)==digest,'Frozen transfer input changed: '+path)
    manifest,scene = [net.read_json(base/name) for name in ('manifest.json','scene.json')]
    net.require(manifest['status']=='completed','Automatic detection must complete before net calculation')
    net.require(manifest['event_mode']=='rim-v2' and manifest['scene_mode']=='nearest-rank-v1','Candidate rule changed')
    net.require('detection_cache' not in manifest and 'scene_cache_path' not in manifest,'New source requires fresh inference')
    net.require(manifest['source_sha256']==frozen['source_sha256'],'Source mismatch')
    net.require(manifest['processed_frames']==frozen['expected_frames'],'Detection coverage incomplete')
    net.require(manifest['profile']==frozen['candidate_profile'],'Candidate parameters differ')
    net.require(manifest['model_sha256']==net.sha256(manifest['ball_model_path']),'Model identity differs')
    events=net.read_json(base/'events.json')
    net.require(len(events['events'])==manifest['candidate_count'],'Candidate count differs')
    return frozen,manifest,scene,events

def experiment(base,freeze,out):
    declared,manifest,scene,events=check_inputs(base,freeze)
    profile=declared['net_profile']
    out.mkdir(parents=True,exist_ok=False)
    began=time.perf_counter()
    fps=manifest['source']['fps']; cases=[]
    for event in events['events']:
        hid=event['hoop_id']
        net.require(isinstance(hid,int) and 0<=hid<len(scene['hoops']),'Automatic hoop identity unavailable')
        h=scene['hoops'][hid]['xyxy'];x1,y1,x2,y2=h;w,height=x2-x1,y2-y1
        # Match the detector's rounded static hoop to this automatically discovered box.
        net.require(abs(event['hoop']['x']-(x1+x2)/2)<2 and abs(event['hoop']['y']-(y1+y2)/2)<2,'Automatic hoop mapping differs')
        context=[max(0,math.floor(x1-w)),max(0,math.floor(y1-height)),
                 min(manifest['source']['width'],math.ceil(x2+w)),min(manifest['source']['height'],math.ceil(y2+height))]
        first=max(0,math.floor((event['time_seconds']-profile['window_seconds'])*fps))
        last=min(manifest['processed_frames']-1,math.ceil((event['time_seconds']+profile['window_seconds'])*fps))
        cases.append({'id':event['id'],'time_seconds':event['time_seconds'],'hoop_id':hid,'hoop_xyxy':h,
                      'context_roi':context,'frame_indices':list(range(first,last+1))})
    paths=[base/name for name in ('manifest.json','scene.json','events.json','detections.jsonl')]
    paths += [Path(manifest['source']['path']),freeze,Path(__file__),Path(net.__file__)]
    inputs={str(path):net.sha256(path) for path in paths}
    frozen={'status':'frozen','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'source':manifest['source'],'baseline':str(base),'cases':cases,'profile':profile,
            'inputs_sha256':inputs,'wrapper_sha256':net.sha256(__file__),
            'net_implementation_sha256':net.sha256(net.__file__),'labels_or_goal_total_used':False,
            'manual_coordinates':False,'parameter_tuning':False}
    net.write_json(out/'frozen.json',frozen)
    (out/'wrapper-implementation.py').write_bytes(Path(__file__).read_bytes())
    (out/'net-implementation.py').write_bytes(Path(net.__file__).read_bytes())
    detections=net.lines(base/'detections.jsonl')
    crops,count=net.decode_cases(Path(manifest['source']['path']),cases)
    net.require(count==manifest['processed_frames']==len(detections),'Full decoding differs')
    records=[]
    for case in cases:
        images=crops[case['id']]
        net.np.savez_compressed(out/(case['id']+'-source-crops.npz'),frames=images)
        actual=net.probe_case(case,images,detections,fps,profile)
        actual['decoded_crop_pixel_sha256']=[net.pixels_sha(image) for image in images]
        records.append(actual)
        print(case['id'],round(case['time_seconds'],3),actual['state'],flush=True)
    selected=[event for event,record in zip(events['events'],records,strict=True) if record['net_support']]
    pages=net.sheets(out,records,crops,fps)
    net.write_json(out/'supported-events.json',{**events,'events':selected,'semantic_limit':'Candidate with net support; user item acceptance pending'})
    net.write_json(out/'results.json',{'status':'completed','profile':profile,'records':records,
        'candidate_count':len(records),'net_supported_count':len(selected),'unsupported_count':len(records)-len(selected),
        'source_pages':pages,'decoded_source_frames':count,'decoded_window_frames':sum(len(r['rows']) for r in records),
        'seconds':time.perf_counter()-began,'default_changed':False,'user_item_confirmation':'pending',
        'semantic_limit':'Unsupported candidates remain uncertain; net support alone does not establish a goal.'})
    for path,digest in inputs.items():net.require(net.sha256(path)==digest,'Input changed during calculation')
    print('Candidates',len(records),'Net support',len(selected),flush=True)

def verify(out):
    frozen,results=[net.read_json(out/name) for name in ('frozen.json','results.json')]
    net.require(not (out/'verification.json').exists(),'Verification already exists')
    for path,digest in frozen['inputs_sha256'].items():net.require(net.sha256(path)==digest,'Frozen input changed')
    net.require(net.sha256(__file__)==frozen['wrapper_sha256']==net.sha256(out/'wrapper-implementation.py'),'Wrapper differs')
    net.require(net.sha256(net.__file__)==frozen['net_implementation_sha256']==net.sha256(out/'net-implementation.py'),'Net helper differs')
    crops,count=net.decode_cases(Path(frozen['source']['path']),frozen['cases'])
    detections=net.lines(Path(frozen['baseline'])/'detections.jsonl')
    net.require(count==results['decoded_source_frames']==len(detections),'Decode coverage differs')
    for record in results['records']:
        with net.np.load(out/(record['id']+'-source-crops.npz'),allow_pickle=False) as saved:
            net.require(net.np.array_equal(saved['frames'],crops[record['id']]),'Native crop pixels differ')
        actual=net.probe_case(record,crops[record['id']],detections,frozen['source']['fps'],frozen['profile'])
        net.require(all(record[key]==value for key,value in actual.items()),'Net replay differs')
        before=net.np.asarray([r['displacement_p75_hoop_widths'] for r in record['rows'] if r['valid'] and r['relative_seconds']<=-.2])
        if len(before)>=6:
            med=float(net.np.median(before));threshold=med+2*float(net.np.median(abs(before-med)))+.02
            net.require(threshold==record['threshold'],'Independent threshold arithmetic differs')
    original=net.read_json(Path(frozen['baseline'])/'events.json')
    selected=[e for e,r in zip(original['events'],results['records'],strict=True) if r['net_support']]
    net.require(selected==net.read_json(out/'supported-events.json')['events'],'Supported list differs')
    net.write_json(out/'verification.json',{'status':'passed','decoded_source_frames':count,
        'candidate_count_checked':len(results['records']),'native_crop_pixels_equal':True,
        'same_net_implementation_replay_equal':True,'threshold_arithmetic_rechecked':True,
        'labels_used':False,'parameters_changed':False})
    print('Native pixels, net replay, threshold and output list verified',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates',type=Path)
    parser.add_argument('--freeze',type=Path)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out',type=Path);group.add_argument('--verify',type=Path)
    args=parser.parse_args();net.cv2.setNumThreads(1);net.cv2.setRNGSeed(0)
    if args.verify:verify(args.verify.resolve(strict=True))
    else:
        net.require(args.candidates and args.freeze,'Candidates and freeze required')
        experiment(args.candidates.resolve(strict=True),args.freeze.resolve(strict=True),args.out.resolve())
