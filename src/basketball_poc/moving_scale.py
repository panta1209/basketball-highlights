"""Full-frame and local source ball fusion with strict experiment cache identity."""
import math
from .moving_roi import restore_balls,crop_specs,validate_boxes
from .scene import merge_boxes

def fused_row(frame,seconds,full_boxes,crops,roi_profile,scale_profile):
    balls=[{**b,'observation_source':'full_frame_large','full_detection_index':i}
        for i,b in enumerate(full_boxes) if b['class_id']==0]
    balls+=restore_balls(crops,roi_profile)
    return {'frame':frame,'time_seconds':seconds,'full_frame_boxes':full_boxes,'crops':crops,
        'boxes':[b for b in full_boxes if b['class_id']==1]+merge_boxes(balls,scale_profile['merge_iou'])}

def validate_scale_cache(m,rows,geometry,identity,count,moving_profile,roi_profile,scale_profile,implementations):
    if m.get('status')!='completed' or m.get('backend')!='moving-scale-v1':raise ValueError('Incompatible source-scale backend')
    for key in ('source_sha256','model_sha256','ultralytics_version','inference'):
        if m.get(key)!=identity[key]:raise ValueError('Source-scale cache identity mismatch')
    for key in ('width','height','fps','duration_seconds'):
        if m.get('source',{}).get(key)!=identity['source'][key]:raise ValueError('Source-scale media geometry mismatch')
    if m.get('moving_profile')!=moving_profile or m.get('roi_profile')!=roi_profile or m.get('scale_profile')!=scale_profile or m.get('implementation_sha256')!=implementations:
        raise ValueError('Source-scale method/profile mismatch')
    if len(rows)!=count or len(geometry)!=count:raise ValueError('Incomplete source-scale cache')
    info=identity['source']
    for i,(r,g) in enumerate(zip(rows,geometry)):
        if type(r.get('frame')) is not int or r['frame']!=i or g.get('frame')!=i:raise ValueError('Source-scale frame order mismatch')
        t=r.get('time_seconds',float('nan'))
        if not math.isfinite(t) or abs(t-i/info['fps'])>1e-9 or g.get('time_seconds')!=t:raise ValueError('Source-scale timestamp mismatch')
        validate_boxes(r.get('full_frame_boxes'),info['width'],info['height'])
        expected=crop_specs(g,info['width'],info['height'],roi_profile)
        if not isinstance(r.get('crops'),list) or len(r['crops'])!=len(expected):raise ValueError('Source-scale crop coverage mismatch')
        for actual,spec in zip(r['crops'],expected):
            if {key:actual.get(key) for key in spec}!=spec:raise ValueError('Source-scale crop geometry mismatch')
        if r!=fused_row(i,t,r['full_frame_boxes'],r['crops'],roi_profile,scale_profile):raise ValueError('Source-scale restoration/fusion mismatch')
