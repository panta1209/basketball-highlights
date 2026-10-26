"""Experimental simultaneous ball/hoop geometry with conservative motion breaks."""
from math import hypot

import cv2
import numpy as np
from basketball_poc.scene import iou


def box_geometry(box):
    x1, y1, x2, y2 = box
    return (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1


def relative_observation(ball, hoop, frame, seconds):
    x, y, w, h = box_geometry(ball['xyxy'])
    hx, _, hw, _ = box_geometry(hoop['xyxy'])
    if hw <= 0 or w <= 0 or h <= 0:
        raise ValueError('Invalid relative geometry')
    return {'frame': frame, 'time_seconds': seconds, 'x': (x-hx)/hw,
            'y': (y-hoop['xyxy'][1])/hw, 'w': w/hw, 'h': h/hw,
            'confidence': ball['confidence'], 'ball_xyxy': list(ball['xyxy']),
            'hoop_xyxy': list(hoop['xyxy']), 'geometry_source': hoop['geometry_source']}


def transform_box(box, matrix):
    x1, y1, x2, y2 = box
    points = np.array([[x1,y1,1],[x2,y1,1],[x2,y2,1],[x1,y2,1]]) @ np.asarray(matrix).T
    return [float(points[:,0].min()),float(points[:,1].min()),
            float(points[:,0].max()),float(points[:,1].max())]


class BackgroundMotion:
    """Partial affine registration is a reliability guard, not perfect cut detection."""

    def __init__(self, profile):
        self.profile = profile
        self.previous = None
        self.orb = cv2.ORB_create(nfeatures=profile['registration_features'])
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def update(self, frame):
        h, w = frame.shape[:2]
        scale = min(1, self.profile['registration_long_side']/max(h,w))
        gray = cv2.cvtColor(cv2.resize(frame, (round(w*scale),round(h*scale))), cv2.COLOR_BGR2GRAY)
        keys, desc = self.orb.detectAndCompute(gray, None)
        previous, self.previous = self.previous, (keys, desc)
        if previous is None:
            return {'reliable':False,'reason':'first_frame','matrix':None}
        old_keys, old_desc = previous
        if old_desc is None or desc is None:
            return {'reliable':False,'reason':'insufficient_features','matrix':None}
        ratio = self.profile['registration_ratio']
        def good(a,b):
            return {pair[0].queryIdx:pair[0].trainIdx for pair in self.matcher.knnMatch(a,b,k=2)
                    if len(pair)==2 and pair[0].distance < ratio*pair[1].distance}
        forward, reverse = good(old_desc,desc), good(desc,old_desc)
        pairs = [(a,b) for a,b in forward.items() if reverse.get(b)==a]
        if len(pairs)<self.profile['registration_min_matches']:
            return {'reliable':False,'reason':'insufficient_matches','matrix':None,'matches':len(pairs)}
        src = np.float32([old_keys[a].pt for a,b in pairs])
        dst = np.float32([keys[b].pt for a,b in pairs])
        matrix, mask = cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,
            ransacReprojThreshold=self.profile['registration_ransac_pixels'])
        inliers = int(mask.sum()) if mask is not None else 0
        reliable = (matrix is not None and np.isfinite(matrix).all()
                    and inliers>=self.profile['registration_min_inliers']
                    and inliers/len(pairs)>=self.profile['registration_min_inlier_fraction'])
        if reliable:
            matrix[:,2] /= scale  # Restore processing-image translation to source pixels.
        return {'reliable':bool(reliable), 'reason':'registered' if reliable else 'unreliable_registration',
                'matrix':matrix.tolist() if reliable else None, 'matches':len(pairs),
                'inliers':inliers,'inlier_fraction':inliers/len(pairs)}


class MovingCandidates:
    def __init__(self, profile):
        self.profile = profile
        self.hoops = {}
        self.tracks = {}
        self.next_hoop = self.next_ball = 0
        self.last_frame = -1
        self.last_time = -1
        self.cooldowns = {}

    def reset(self):
        self.hoops.clear()
        self.tracks.clear()
        self.cooldowns.clear()

    def update(self, frame, seconds, boxes, motion, image_size=None):
        if frame <= self.last_frame or seconds <= self.last_time:
            raise ValueError('Frames and timestamps must increase')
        self.last_frame, self.last_time = frame, seconds
        p = self.profile
        reset_reason = None
        if not motion['reliable']:
            self.reset()
            reset_reason = motion['reason']
        propagated = {}
        if motion['reliable']:
            for hid, hoop in self.hoops.items():
                if seconds-hoop['last_observed_seconds'] <= p['track_ttl_seconds']+1e-12:
                    transformed=transform_box(hoop['xyxy'],motion['matrix'])
                    if image_size and (transformed[2]<=0 or transformed[3]<=0
                                       or transformed[0]>=image_size[0] or transformed[1]>=image_size[1]):
                        continue
                    propagated[hid] = {**hoop,'xyxy':transformed,'geometry_source':'background_propagation'}
        detections = sorted((b for b in boxes if b['class_id']==1 and b['confidence']>=p['hoop_confidence']),
                            key=lambda b:b['confidence'],reverse=True)
        used = set()
        identity_resets=[]
        for detection in detections:
            box=detection['xyxy']; x,y,w,h=box_geometry(box)
            matches=[]
            incompatible=[]
            for hid, hoop in propagated.items():
                hx,hy,hw,hh=box_geometry(hoop['xyxy'])
                if min(w,h,hw,hh)<=0: continue
                distance=hypot(x-hx,y-hy)/hypot(hw,hh)
                scale_ratio=max(w/hw,hw/w,h/hh,hh/h)
                if distance<p['hoop_match_diagonals'] and scale_ratio<=p['max_hoop_scale_ratio']:
                    matches.append(hid)
                elif distance<p['hoop_match_diagonals']:
                    incompatible.append(hid)
            for old in incompatible:
                propagated.pop(old,None)
                self.tracks.pop(old,None)
                identity_resets.append({'hoop_id':old,'reason':'geometry_scale_jump'})
            if len(matches)==1 and matches[0] in used:
                if iou(box,propagated[matches[0]]['xyxy'])>.5:
                    continue  # Duplicate observation, not another supported hoop.
                matches=[]  # Nearby distinct hoops still require one-to-one identity.
            if len(matches)==1:
                hid=matches[0]
            else:
                # Ambiguous identity: never inherit tracks or let an unmatched old
                # estimate continue competing with the new fresh observation.
                for old in matches:
                    propagated.pop(old,None)
                    self.tracks.pop(old,None)
                    identity_resets.append({'hoop_id':old,'reason':'ambiguous_identity'})
                hid=self.next_hoop; self.next_hoop+=1
            propagated[hid]={'id':hid,'xyxy':list(box),'confidence':detection['confidence'],
                              'last_observed_seconds':seconds,'geometry_source':'detected'}
            used.add(hid)
        self.hoops=propagated
        self.tracks={hid:tracks for hid,tracks in self.tracks.items() if hid in self.hoops}
        decisions=[]
        for hid,hoop in self.hoops.items():
            tracks=self.tracks.setdefault(hid,{})
            for key in list(tracks):
                if seconds-tracks[key]['points'][-1]['time_seconds']>p['track_ttl_seconds']+1e-12:
                    del tracks[key]
            observations=[]
            for b in boxes:
                if b['class_id']!=0 or b['confidence']<p['ball_confidence']: continue
                observation=relative_observation(b,hoop,frame,seconds)
                # Same symmetric, dimensionless neighborhood for every hoop.
                if abs(observation['x'])<3.5 and abs(observation['y'])<6:
                    observations.append(observation)
            candidates=[]
            for i,observation in enumerate(observations):
                for key,track in tracks.items():
                    prev=track['points'][-1]
                    distance=hypot(observation['x']-prev['x'],observation['y']-prev['y'])
                    if distance < hypot(prev['w'],prev['h'])*p['ball_match_diagonals']:
                        candidates.append((distance,i,key))
            assignments={}; taken=set()
            for distance,i,key in sorted(candidates):
                if i not in assignments and key not in taken:
                    assignments[i]=key; taken.add(key)
            for i,observation in enumerate(observations):
                key=assignments.get(i)
                if key is None:
                    key=self.next_ball;self.next_ball+=1
                    tracks[key]={'points':[],'pending':None,'armed':False}
                track=tracks[key]
                if track['points']:
                    prev=track['points'][-1]
                    if observation['y']<=0:
                        track['pending']=None;track['armed']=True
                    elif prev['y']<=0<observation['y'] and track['armed']:
                        ratio=-prev['y']/(observation['y']-prev['y'])
                        track['pending']={'above':prev,'below':observation,
                            'predicted_x':prev['x']+ratio*(observation['x']-prev['x']),
                            'time_seconds':prev['time_seconds']+ratio*(seconds-prev['time_seconds'])}
                elif observation['y']<=0:
                    track['armed']=True
                track['points'].append(observation)
                track['points']=[o for o in track['points'] if seconds-o['time_seconds']<=p['history_seconds']+1e-12]
                pending=track['pending']
                if pending and observation['y']-observation['h']/2>0:
                    x=pending['predicted_x'];inside=-.5<x<.5
                    cooled=seconds-self.cooldowns.get(hid,float('-inf'))>=p['cooldown_seconds']
                    decision='candidate' if inside and cooled else ('cooldown' if inside else 'outside_projection')
                    record={**pending,'decision':decision,'made':decision=='candidate',
                        'confirmed_at_seconds':seconds,'hoop_id':hid,'ball_id':key,
                        'below_confirmation':observation,'hoop':{'x':(hoop['xyxy'][0]+hoop['xyxy'][2])/2,
                            'y':(hoop['xyxy'][1]+hoop['xyxy'][3])/2,
                            'w':hoop['xyxy'][2]-hoop['xyxy'][0],'h':hoop['xyxy'][3]-hoop['xyxy'][1]},
                        'rim_geometry':'dynamic_hoop_box_top_proxy','net_required':False}
                    decisions.append(record)
                    if decision=='candidate':self.cooldowns[hid]=seconds
                    track['pending']=None;track['armed']=False
        status=('uncertain_registration' if reset_reason and frame else
                ('uncertain_hoop_identity' if identity_resets else
                 ('available' if self.hoops else 'unavailable_hoop')))
        return {'frame':frame,'time_seconds':seconds,'status':status,'reset_reason':reset_reason,
                'identity_resets':identity_resets,'motion':motion,'hoops':list(self.hoops.values()),'decisions':decisions}
