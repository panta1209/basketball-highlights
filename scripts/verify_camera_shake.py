"""Replay events/net and verify corrected pixels against saved source transforms."""
import argparse
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import benchmark_fixed_camera_scale as scale
from basketball_poc.camera_shake import validate_motion
from basketball_poc.media import read_json,write_json,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);args=p.parse_args()
    out=args.run.resolve();cv2.setNumThreads(1);cv2.setRNGSeed(0)
    frozen=read_json(out/'frozen.json')['inputs_sha256'];scale.legacy.identities(frozen)
    prediction=read_json(out/'predictions-before-review.json')['files_sha256']
    scale.legacy.identities(prediction)
    result=read_json(out/'result.json')
    pixel_checks=[]
    if result['mode']=='stabilized_fallback':
        motion=read_json(out/'stabilization/stabilization.json')
        info=motion['source'];rows=motion['transforms']
        assert len(rows)==round(info['fps']*info['duration_seconds'])==motion['frames']
        assert sha256(motion['output'])==motion['output_sha256']
        for i,r in enumerate(rows):
            assert r['frame']==i and r['time_seconds']==i/info['fps']
            validate_motion(np.array(r['reference_to_current']),info['width'],info['height'],motion['profile'])
            assert r['inlier_points']>=motion['profile']['minimum_inliers']
            assert r['inlier_fraction']>=motion['profile']['minimum_inlier_fraction']
        source=cv2.VideoCapture(info['path']);corrected=cv2.VideoCapture(motion['output'])
        count=0
        while True:
            ok,a=source.read();ok2,b=corrected.read();assert ok==ok2
            if not ok:break
            if count%300==0 or count==len(rows)-1:
                matrix=np.array(rows[count]['reference_to_current'])
                expected=cv2.warpAffine(a,cv2.invertAffineTransform(matrix),
                    (info['width'],info['height']),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
                assert np.array_equal(expected,b),count
                pixel_checks.append(count)
            count+=1
        source.release();corrected.release();assert count==len(rows)
    if result['status']=='completed':
        pipe=Path(result['pipeline']);m=read_json(pipe/'candidates/manifest.json')
        with (out/'verification-replay.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'scripts/benchmark_automatic_scale.py'),
                m['source']['path'],'--out',str(out/'verification-replay'),'--event-mode','rim-v2',
                '--detections-cache',str(pipe/'candidates')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        for name in ['detections.jsonl','associations.jsonl','attempts.jsonl','events.json','scene.json']:
            assert (pipe/'candidates'/name).read_bytes()==(out/'verification-replay'/name).read_bytes(),name
        scale.compute(pipe,verify=True)
    scale.legacy.identities(prediction);scale.legacy.identities(frozen)
    write_json(out/'verification.json',{'status':'passed','recognition_status':result['status'],
        'source_transform_pixel_checks':pixel_checks,'frame_and_time_mapping_verified':True,
        'candidate_context_net_replay_equal':result['status']=='completed',
        'predictions_unchanged':True,'method_identities_unchanged':True})
    print('Verified',out, result['status'],flush=True)


if __name__=='__main__':main()
