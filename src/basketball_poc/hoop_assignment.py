"""Experimental exclusive geometry assignment; no outcome/label inputs."""
from math import hypot, log

from basketball_poc.moving import box_geometry
from basketball_poc.scene import iou


def minimum_assignment(edges, row_count, prior_ids, excluded=None):
    """Rectangular Hungarian algorithm with cardinality-prioritized dummy columns."""
    ids=sorted(prior_ids); n=row_count; m=len(ids)+n
    if not n or not ids:return {}
    penalty=n+1; forbidden=penalty*(n+2)
    costs=[[edges.get((i,hid),forbidden) if (i,hid)!=excluded else forbidden
            for hid in ids]+[penalty]*n for i in range(n)]
    u=[0.]*(n+1);v=[0.]*(m+1);match=[0]*(m+1);way=[0]*(m+1)
    for i in range(1,n+1):
        match[0]=i;j0=0;mins=[float('inf')]*(m+1);used=[False]*(m+1)
        while True:
            used[j0]=True;i0=match[j0];delta=float('inf');j1=0
            for j in range(1,m+1):
                if used[j]:continue
                current=costs[i0-1][j-1]-u[i0]-v[j]
                if current<mins[j]:mins[j]=current;way[j]=j0
                if mins[j]<delta:delta=mins[j];j1=j
            for j in range(m+1):
                if used[j]:u[match[j]]+=delta;v[j]-=delta
                else:mins[j]-=delta
            j0=j1
            if match[j0]==0:break
        while True:
            j1=way[j0];match[j0]=match[j1];j0=j1
            if j0==0:break
    return {match[j]-1:ids[j-1] for j in range(1,len(ids)+1)
            if match[j] and (match[j]-1,ids[j-1]) in edges
            and (match[j]-1,ids[j-1])!=excluded}


def associate(propagated, detections, profile, experiment, next_id):
    kept=[];suppressed=[]
    for d in sorted(detections,key=lambda b:(-b['confidence'],tuple(b['xyxy']))):
        duplicate=next((i for i,k in enumerate(kept)
                        if iou(d['xyxy'],k['xyxy'])>experiment['duplicate_iou']),None)
        if duplicate is None:kept.append(d)
        else:suppressed.append({'observation':d,'kept_observation_index':duplicate})
    edges={};metrics=[];near_incompatible=set()
    for i,d in enumerate(kept):
        x,y,w,h=box_geometry(d['xyxy'])
        for hid,old in sorted(propagated.items()):
            hx,hy,hw,hh=box_geometry(old['xyxy'])
            if min(w,h,hw,hh)<=0:continue
            distance=hypot(x-hx,y-hy)/hypot(hw,hh)
            ratio=max(w/hw,hw/w,h/hh,hh/h);overlap=iou(d['xyxy'],old['xyxy'])
            compatible=distance<profile['hoop_match_diagonals'] and ratio<=profile['max_hoop_scale_ratio']
            cost=(distance/profile['hoop_match_diagonals']+
                  log(ratio)/log(profile['max_hoop_scale_ratio'])+1-overlap)/3
            metrics.append({'row':i,'prior_id':hid,'distance':distance,'scale_ratio':ratio,
                            'iou':overlap,'compatible':compatible,'cost':cost})
            if compatible:edges[i,hid]=cost
            elif distance<profile['hoop_match_diagonals']:near_incompatible.add(hid)
    proposed=minimum_assignment(edges,len(kept),propagated)
    total=sum(edges[i,h] for i,h in proposed.items());bad_rows=set();bad_ids=set();alternatives=[]
    for edge in sorted(proposed.items()):
        alt=minimum_assignment(edges,len(kept),propagated,excluded=edge)
        changed=[i for i in range(len(kept)) if alt.get(i)!=proposed.get(i)]
        cost=sum(edges[i,h] for i,h in alt.items())
        ambiguous=(len(alt)==len(proposed) and changed and
                   (cost-total)/len(changed)<=experiment['ambiguity_mean_cost_gap']+1e-12)
        alternatives.append({'excluded_edge':list(edge),'assignments':sorted(alt.items()),
          'cardinality':len(alt),'cost':cost,'changed_rows':changed,'ambiguous':bool(ambiguous)})
        if ambiguous:
            bad_rows.update(changed)
            bad_ids.update(h for i in changed for h in (alt.get(i),proposed.get(i)) if h is not None)
    bad_rows.update(i for i,h in proposed.items() if h in bad_ids)
    assignments={i:h for i,h in proposed.items() if i not in bad_rows and h not in bad_ids}
    resets={hid:'ambiguous_identity' for hid in bad_ids}
    for hid in propagated:
        if hid in assignments.values() or hid in resets:continue
        # Unsupported competitors are dropped, never merged into the winning history.
        if any((i,hid) in edges for i in assignments):resets[hid]='unsupported_competing_prior'
        elif hid in near_incompatible and not any(h==hid for i,h in edges):resets[hid]='geometry_scale_jump'
    result={hid:h for hid,h in propagated.items() if hid not in resets}
    applied=[]
    for i,d in enumerate(kept):
        hid=assignments.get(i)
        if hid is None:hid=next_id;next_id+=1
        applied.append([i,hid])
        result[hid]={'id':hid,'xyxy':list(d['xyxy']),'confidence':d['confidence'],
                    'last_observed_seconds':None,'geometry_source':'detected'}
    diagnostic={'prior_hoops':list(propagated.values()),'observations':kept,'suppressed':suppressed,
        'metrics':metrics,'proposed_assignments':sorted(proposed.items()),'proposed_cost':total,
        'alternatives':alternatives,'ambiguous_rows':sorted(bad_rows),'ambiguous_prior_ids':sorted(bad_ids),
        'retained_assignments':sorted(assignments.items()),'applied_assignments':applied}
    return result,[{'hoop_id':h,'reason':why} for h,why in sorted(resets.items())],next_id,diagnostic
