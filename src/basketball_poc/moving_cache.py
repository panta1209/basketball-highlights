"""Detection cache validation; the two legacy logging adapters were separately audited."""
import math


# These exact logging adapter versions produced full-frame native default 640/.2
# observations with the same pinned model and Ultralytics version. They are NOT
# video-specific algorithm parameters. Unrecognized legacy caches are rejected.
AUDITED_NATIVE_ADAPTERS = {
    'dad5925c5346a2af4f1b6b3e6a090a815a75bf96719f99a6f46ab985c3d1eaac',
    'c6cfc4bd82f49a574cfac6bb1a39f4364f096cdff9e1590d5e9eaef82f1c0166',
}
AUDITED_NATIVE_DETECTOR='4b2bd625beea6398b250c4fbcd38ebd1742793b7e720dcdbb8251b358b98886b'


def validate_cache(manifest, rows, identity, count, native=False):
    if manifest.get('status')!='completed':raise ValueError('Incomplete detection cache')
    if manifest.get('source_sha256')!=identity['source_sha256'] or manifest.get('model_sha256')!=identity['model_sha256']:
        raise ValueError('Source/model cache mismatch')
    info=manifest.get('source',{})
    if any(info.get(k)!=identity['source'][k] for k in ('width','height','fps','duration_seconds')):
        raise ValueError('Source geometry/timing cache mismatch')
    if native:
        if (manifest.get('backend')!='josephattalla' or manifest.get('adapter_sha256') not in AUDITED_NATIVE_ADAPTERS
            or manifest.get('adapted_source_sha256')!=AUDITED_NATIVE_DETECTOR
            or manifest.get('versions',{}).get('ultralytics')!=identity['ultralytics_version']
            or not manifest.get('inference','').startswith('full source frame; upstream default imgsz;')
            or identity['inference']!={'scope':'full_frame','imgsz':640,'confidence':.2}):
            raise ValueError('Unaudited native inference cache; static ROI caches cannot be imported')
    elif (manifest.get('inference')!=identity['inference'] or manifest.get('backend')!='moving-v1'
          or manifest.get('ultralytics_version')!=identity['ultralytics_version']):
        raise ValueError('Incompatible full-frame inference cache')
    if len(rows)!=count:raise ValueError('Incomplete frame coverage')
    for index,row in enumerate(rows):
        if type(row.get('frame')) is not int or row['frame']!=index:
            raise ValueError('Frame order mismatch')
        if not math.isfinite(row.get('time_seconds',float('nan'))) or row['time_seconds']<0 or abs(row['time_seconds']-index/info['fps'])>.005:
            raise ValueError('Cache timestamp mismatch')
        if not isinstance(row.get('boxes'),list):raise ValueError('Missing cached observations')
        for box in row['boxes']:
            coords=box.get('xyxy',[]);conf=box.get('confidence',float('nan'))
            if (len(coords)!=4 or not all(math.isfinite(c) for c in coords) or coords[0]>=coords[2] or coords[1]>=coords[3]
                or not math.isfinite(conf) or not 0<=conf<=1 or box.get('class_id') not in (0,1)):
                raise ValueError('Invalid cached observation')
    return rows
