"""Hoop-relative source crops and auditable local-to-source observations."""
import math

from .moving_cache import validate_cache
from .scene import merge_boxes


def dynamic_crop(hoop, width, height, profile):
    x1,y1,x2,y2=hoop['xyxy'];w=x2-x1;cx=(x1+x2)/2
    if w<=0 or y2<=y1 or not all(math.isfinite(x) for x in hoop['xyxy']):
        raise ValueError('Invalid hoop geometry')
    side=profile['horizontal_hoop_widths']*w;vertical=profile['vertical_hoop_widths']*w
    roi=[max(0,math.floor(cx-side)),max(0,math.floor(y1-vertical)),
         min(width,math.ceil(cx+side)),min(height,math.ceil(y1+vertical))]
    return roi if roi[0]<roi[2] and roi[1]<roi[3] else None


def crop_specs(geometry, width, height, profile):
    specs=[]
    for hoop in geometry['hoops']:
        roi=dynamic_crop(hoop,width,height,profile)
        if roi is not None:
            specs.append({'hoop_id':hoop['id'],'hoop_xyxy':hoop['xyxy'],
                          'geometry_source':hoop['geometry_source'],
                          'last_observed_seconds':hoop['last_observed_seconds'],'roi':roi})
    return specs


def validate_boxes(boxes, width, height):
    if not isinstance(boxes,list):raise ValueError('Missing observations')
    for b in boxes:
        a=b.get('xyxy',[]);conf=b.get('confidence',float('nan'))
        if (len(a)!=4 or not all(math.isfinite(c) for c in a)
            or not 0<=a[0]<a[2]<=width or not 0<=a[1]<a[3]<=height
            or b.get('class_id') not in (0,1) or not math.isfinite(conf) or not 0<=conf<=1):
            raise ValueError('Invalid local/source observations')


def restore_balls(crops, profile):
    balls=[]
    for ci,crop in enumerate(crops):
        x1,y1,x2,y2=crop['roi'];validate_boxes(crop['local_boxes'],x2-x1,y2-y1)
        for bi,b in enumerate(crop['local_boxes']):
            if b['class_id']!=0:raise ValueError('Ball-only ROI observation expected')
            a,c,d,e=b['xyxy']
            balls.append({**b,'xyxy':[a+x1,c+y1,d+x1,e+y1],
                          'observation_source':'dynamic_roi','crop_index':ci,'local_detection_index':bi})
    return merge_boxes(balls,profile['merge_iou'])


def detection_row(frame, seconds, full_boxes, crops, profile):
    return {'frame':frame,'time_seconds':seconds,'full_frame_boxes':full_boxes,'crops':crops,
            'boxes':[b for b in full_boxes if b['class_id']==1]+restore_balls(crops,profile)}


def validate_locator(manifest, rows, geometry, identity, count, moving_profile, moving_hash):
    validate_cache(manifest,rows,identity,count)
    if manifest.get('profile')!=moving_profile or manifest.get('implementation_sha256',{}).get('src/basketball_poc/moving.py')!=moving_hash:
        raise ValueError('Locator geometry method/profile mismatch')
    if len(geometry)!=count:raise ValueError('Incomplete locator geometry')
    for i,g in enumerate(geometry):
        if g.get('frame')!=i or g.get('time_seconds')!=rows[i]['time_seconds']:
            raise ValueError('Locator geometry timing mismatch')
        if not isinstance(g.get('hoops'),list) or not isinstance(g.get('motion'),dict):
            raise ValueError('Missing locator geometry')
        validate_boxes(rows[i]['boxes'],identity['source']['width'],identity['source']['height'])


def validate_roi_cache(manifest, rows, geometry, identity, count, moving_profile, roi_profile, implementations):
    for key in ('source_sha256','model_sha256','ultralytics_version','source'):
        # Absolute file location may change without changing source identity.
        if key=='source':
            if any(manifest.get(key,{}).get(k)!=identity[key][k] for k in ('width','height','fps','duration_seconds')):
                raise ValueError('ROI cache source geometry mismatch')
        elif manifest.get(key)!=identity[key]:raise ValueError('ROI cache identity mismatch')
    if (manifest.get('status')!='completed' or manifest.get('backend')!='moving-roi-v1'
        or manifest.get('inference')!=identity['inference'] or manifest.get('moving_profile')!=moving_profile
        or manifest.get('roi_profile')!=roi_profile or manifest.get('implementation_sha256')!=implementations):
        raise ValueError('Incompatible dynamic ROI cache')
    if len(rows)!=count or len(geometry)!=count:raise ValueError('Incomplete ROI cache coverage')
    width,height=identity['source']['width'],identity['source']['height']
    for i,(r,g) in enumerate(zip(rows,geometry)):
        if type(r.get('frame')) is not int or r['frame']!=i or g.get('frame')!=i:
            raise ValueError('ROI frame order mismatch')
        if not math.isfinite(r.get('time_seconds',float('nan'))) or abs(r['time_seconds']-i/identity['source']['fps'])>1e-9 or g.get('time_seconds')!=r['time_seconds']:
            raise ValueError('ROI time mismatch')
        validate_boxes(r.get('full_frame_boxes'),width,height)
        expected=crop_specs(g,width,height,roi_profile)
        if not isinstance(r.get('crops'),list) or len(expected)!=len(r['crops']):
            raise ValueError('ROI crop coverage mismatch')
        for actual,spec in zip(r['crops'],expected):
            if {k:actual.get(k) for k in spec}!=spec:raise ValueError('ROI crop geometry mismatch')
        if r!=detection_row(i,r['time_seconds'],r['full_frame_boxes'],r['crops'],roi_profile):
            raise ValueError('ROI restoration/merge mismatch')
