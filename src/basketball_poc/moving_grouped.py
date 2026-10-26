"""Isolated grouped hoop observation experiment; original ball/event code preserved."""
from math import hypot
from basketball_poc.moving import transform_box, relative_observation
from basketball_poc.hoop_grouping import associate


class GroupedHoopCandidates:
    def __init__(self, profile, experiment):
        self.profile = profile
        self.experiment = experiment
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
        propagated,identity_resets,self.next_hoop,association=associate(
            propagated,detections,p,self.experiment,self.next_hoop)
        for h in propagated.values():
            if h['geometry_source']=='detected':h['last_observed_seconds']=seconds
        for record in identity_resets:
            self.tracks.pop(record['hoop_id'],None)
            self.cooldowns.pop(record['hoop_id'],None)
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
                'identity_resets':identity_resets,'identity_association':association,'motion':motion,'hoops':list(self.hoops.values()),'decisions':decisions}
