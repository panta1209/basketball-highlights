"""Event provenance and provisional readiness, separate from model accuracy."""
import math


def interval(value,duration):
    if value is None:return None
    if len(value)!=2 or not all(isinstance(x,(float,int)) and math.isfinite(x) for x in value):
        raise ValueError('Invalid interval')
    if not 0<=value[0]<=value[1]<duration:raise ValueError('Interval outside clip')
    return list(value)


def select_anchor(geometries,seconds,tolerance=.25):
    choices=[g for g in geometries if g['hoops'] and abs(g['time_seconds']-seconds)<=tolerance+1e-12]
    if not choices:return None
    row=min(choices,key=lambda g:(abs(g['time_seconds']-seconds),g['frame']))
    hoop=min(row['hoops'],key=lambda h:(-(h['xyxy'][2]-h['xyxy'][0]),h['id']))
    return {'frame':row['frame'],'time_seconds':row['time_seconds'],'hoop_id':hoop['id'],
        'hoop_xyxy':hoop['xyxy'],'selection':'nearest available frame within.25s then largest width, lowest ID tie'}


def readiness(annotation,duration,anchor,plan,target_correct):
    outcome=annotation['outcome'];evidence=interval(annotation['result_interval'],duration)
    interval(annotation.get('release_interval'),duration)
    if outcome not in ('made','miss','unknown'):raise ValueError('Invalid outcome')
    if outcome=='unknown' and evidence is not None:raise ValueError('Unknown must not fabricate a result time')
    for attempt in annotation.get('additional_attempts',[]):interval(attempt,duration)
    cuts=annotation.get('cuts_seconds',[])
    if any(not isinstance(t,(float,int)) or not math.isfinite(t) or not 0<t<duration for t in cuts):raise ValueError('Invalid cut')
    label_ready=outcome!='unknown' and evidence is not None
    center=sum(evidence)/2 if evidence else None
    full_context=center is not None and center-1>=0 and center+1<duration
    contamination=[]
    if center is not None:
        lo,hi=center-1,center+1
        if any(lo<=t<=hi for t in cuts):contamination.append('shot change inside result window')
        if any(a<=hi and b>=lo for a,b in annotation.get('additional_attempts',[])):
            contamination.append('other attempt inside result window')
    geometry=anchor is not None and bool(plan and plan['eligible'])
    reasons=[]
    if not label_ready:reasons.append('outcome/time unresolved')
    if not full_context:reasons.append('truncated result context')
    reasons.extend(contamination)
    if not geometry:reasons.append('sampled target identity missing')
    if target_correct is not True:reasons.append('automatic target unverified or incorrect')
    return {'label_ready':label_ready,'full_context':full_context,'geometry_ready':geometry,
        'target_correct':target_correct,'contamination':contamination,'provisional_training_ready':not reasons,
        'reasons':reasons,'center_seconds':center}


def game_folds(records):
    groups=sorted({r['game'] for r in records});folds=[]
    for group in groups:
        train=[r['id'] for r in records if r['game']!=group and r['readiness']['provisional_training_ready']]
        test=[r['id'] for r in records if r['game']==group and r['readiness']['provisional_training_ready']]
        by_id={r['id']:r for r in records}
        train_classes=sorted({by_id[i]['annotation']['outcome'] for i in train})
        test_classes=sorted({by_id[i]['annotation']['outcome'] for i in test})
        folds.append({'held_game':group,'train_ids':train,'test_ids':test,
            'train_classes':train_classes,'test_classes':test_classes,
            'both_classes_present':train_classes==test_classes==['made','miss']})
    return folds
