"""Bounded ball context from raw detections; only net-assisted review events."""
from math import hypot

from .core import number
from .net_recovery import tracked_hoop

VERSION = 'shot-context-v2'


def motion_pair(points, fraction):
    """Repeated source frames and position jitter cannot invent a velocity."""
    last = points[-1]
    for previous in reversed(points[:-1]):
        if hypot(last['x']-previous['x'], last['y']-previous['y']) >= fraction*hypot(last['w'],last['h']):
            return previous,last
    return None


def shot_context(detections, hoops, existing_events, *, fps, duration_seconds,
                 candidate_profile, context_profile, net_window_seconds):
    fps=number(fps,'fps',1e-12);duration=number(duration_seconds,'duration',1e-12)
    ttl=number(candidate_profile['track_ttl_seconds'],'ttl',1e-12)
    history=number(candidate_profile['history_seconds'],'history',1e-12)
    window=number(net_window_seconds,'net window',1e-12)
    c=context_profile
    for key,value in c.items():
        if key!='version':number(value,key,1e-12)
    if c['weak_confidence']>c['strong_confidence'] or c['strong_confidence']>1 or c['recovery_max_delay_ttl_fraction']>1:
        raise ValueError('Invalid confidence or prediction limits')
    geometry=[tracked_hoop(h['xyxy']) for h in hoops]
    if not geometry:raise ValueError('Context requires automatic hoop geometry')
    expected=round(fps*duration)
    if len(detections)!=expected:raise ValueError('Context must cover the complete source')
    tracks=[];associations=[];rejections=[]
    for index,row in enumerate(detections):
        if row['frame']!=index or abs(row['time_seconds']-index/fps)>1e-6:
            raise ValueError('Context source frames or timing differ')
        used=set()
        for box in sorted((b for b in row['boxes'] if b['class_id']==0),key=lambda b:-b['confidence']):
            x1,y1,x2,y2=[number(v,'ball coordinate') for v in box['xyxy']]
            if x2<=x1 or y2<=y1:raise ValueError('Ball box must have positive area')
            p={'frame':index,'time_seconds':index/fps,'x':(x1+x2)/2,'y':(y1+y2)/2,
               'w':x2-x1,'h':y2-y1,'conf':number(box['confidence'],'confidence')}
            if p['conf']<c['weak_confidence']:continue
            options=[]
            for key,track in enumerate(tracks):
                if key in used:continue
                last=track[-1];dt=p['time_seconds']-last['time_seconds']
                if not 0<dt<=ttl+1e-12:continue
                if max(p['w']/last['w'],last['w']/p['w'],p['h']/last['h'],last['h']/p['h'])>c['maximum_size_ratio']:continue
                recent=[q for q in track if last['time_seconds']-q['time_seconds']<=history+1e-12]
                pair=motion_pair(recent,c['motion_min_ball_diagonals'])
                hoop=min(geometry,key=lambda h:hypot(h['x']-last['x'],h['y']-last['y'])/h['w'])
                allowance=hypot(last['w'],last['h'])*c['association_ball_diagonals']
                if pair:
                    a,b=pair;span=b['time_seconds']-a['time_seconds']
                    predicted=[b[k]+(b[k]-a[k])*dt/span for k in ('x','y')]
                    allowance+=.5*c['prediction_acceleration_hoop_widths_per_second_squared']*hoop['w']*dt*dt
                    method='velocity_prediction'
                else:
                    if dt>c['bootstrap_max_gap_seconds']+1e-12:continue
                    predicted=[last['x'],last['y']]
                    allowance=max(allowance,c['bootstrap_speed_hoop_widths_per_second']*hoop['w']*dt)
                    method='elapsed_time_bootstrap'
                residual=hypot(p['x']-predicted[0],p['y']-predicted[1])
                # A strongly observed nearby reversal must preserve rebound history.
                direct=hypot(p['x']-last['x'],p['y']-last['y'])
                nearby=hypot(last['w'],last['h'])*c['association_ball_diagonals']
                if residual>allowance and p['conf']>=c['strong_confidence'] and direct<=nearby:
                    predicted=[last['x'],last['y']];residual=direct;allowance=nearby
                    method='strong_nearby_direction_change'
                if residual<=allowance:
                    options.append((residual/allowance,key,{'method':method,'elapsed_seconds':dt,
                        'predicted_xy':predicted,'residual_pixels':residual,'allowed_pixels':allowance}))
            if options:
                _,key,evidence=min(options,key=lambda v:(v[0],v[1]))
                tracks[key].append(p)
            elif p['conf']>=c['strong_confidence']:
                key=len(tracks);tracks.append([p]);evidence={'method':'strong_seed'}
            else:
                rejections.append({'frame':index,'reason':'weak_detection_without_compatible_track'})
                continue
            used.add(key)
            associations.append({'track_id':key,**p,'weak_continuation':p['conf']<c['strong_confidence'],
                                 'association':evidence})

    proposals=[]
    for key,track in enumerate(tracks):
        for hid,hoop in enumerate(geometry):
            plane=hoop['y']-hoop['h']/2;left=hoop['x']-hoop['w']/2;right=hoop['x']+hoop['w']/2
            resolved=[]
            def fits(p):return p['w']<=hoop['w'] and p['h']<=hoop['h']
            def event(kind,index,t,anchor,a,b,px,confirmation=None):
                return {'id':f'context-{kind}-{key:04d}-{index:05d}-hoop-{hid:03d}',
                    'status':'candidate','evidence_origin':VERSION,'reason':kind,'hoop_id':hid,'hoop':hoop,
                    'context_track_id':key,'time_seconds':t,'net_anchor_seconds':anchor,
                    'timestamp_method':'observed_bracket' if kind=='crossing' else 'short_descending_continuation_estimate',
                    'rim_plane_y':plane,'rim_geometry':'hoop_box_top_proxy','predicted_x':px,
                    'previous_observation':a,'last_observation':b,'below_confirmation':confirmation,
                    'below_rim_observed':confirmation is not None,'net_required':True}
            # Prefer a bracket observed across a bounded gap, with a full-below observation.
            for i,(a,b) in enumerate(zip(track,track[1:])):
                if not a['y']<=plane<b['y'] or not fits(a) or not fits(b):continue
                confirmation=None
                for j in range(i+1,len(track)):
                    p=track[j]
                    if j>i+1 and p['time_seconds']-track[j-1]['time_seconds']>ttl+1e-12:break
                    if p['time_seconds']-a['time_seconds']>history+1e-12 or p['y']<=plane:break
                    if p['y']-p['h']/2>plane:
                        confirmation=(j,p);break
                if confirmation is None:continue
                j,below=confirmation;resolved.append((i,j))
                ratio=(plane-a['y'])/(b['y']-a['y']);px=a['x']+ratio*(b['x']-a['x'])
                t=a['time_seconds']+ratio*(b['time_seconds']-a['time_seconds'])
                if not left<px<right:
                    rejections.append({'track_id':key,'hoop_id':hid,'frame':a['frame'],'reason':'observed_crossing_outside_hoop'});continue
                if t+window>=duration:continue
                proposals.append(event('crossing',i,t,t,a,b,px,below))
            # A short falling prefix can trigger review when the predicted crossing is unobserved.
            for i,last in enumerate(track):
                if any(a<=i<=b for a,b in resolved):continue
                recent=[p for p in track[:i+1] if last['time_seconds']-p['time_seconds']<=history+1e-12]
                pair=motion_pair(recent,c['motion_min_ball_diagonals'])
                if not pair:continue
                previous,last=pair
                if not previous['y']<last['y']<=plane or not fits(previous) or not fits(last):continue
                if plane-last['y']>c['recovery_above_hoop_heights']*hoop['h']:continue
                if any(a['y']<=plane<b['y'] for a,b in zip(recent,recent[1:])):continue
                ratio=(plane-last['y'])/(last['y']-previous['y'])
                delay=ratio*(last['time_seconds']-previous['time_seconds'])
                px=last['x']+ratio*(last['x']-previous['x']);t=last['time_seconds']+delay
                if not 0<=delay<=ttl*c['recovery_max_delay_ttl_fraction'] or not left<px<right:continue
                if last['time_seconds']+max(ttl,window)>=duration:continue
                if i+1<len(track):
                    following=track[i+1]
                    if following['time_seconds']<=t+1/fps:continue
                    if following['time_seconds']-last['time_seconds']<=ttl and following['y']<=plane:continue
                proposals.append(event('lost',i,t,last['time_seconds'],previous,last,px))
    events=[]
    for proposal in sorted(proposals,key=lambda p:(p['reason']!='crossing',p['time_seconds'],p['id'])):
        duplicates=[e for e in [*existing_events,*events] if e['hoop_id']==proposal['hoop_id']
                    and abs(e['time_seconds']-proposal['time_seconds'])<=window]
        if duplicates:
            rejections.append({'candidate_id':proposal['id'],'reason':'same_hoop_event_in_net_window','duplicate_of':duplicates[0]['id']})
        else:events.append(proposal)
    return {'version':VERSION,'profile':c,'events':sorted(events,key=lambda e:(e['time_seconds'],e['id'])),
        'associations':associations,'track_count':len(tracks),'rejections':rejections,
        'processed_frames':len(detections),'weak_continuation_count':sum(a['weak_continuation'] for a in associations),
        'manual_timestamps_used':False,'labels_used':False,'net_motion_alone_creates_events':False}
