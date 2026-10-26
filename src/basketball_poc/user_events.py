"""Conservative links between source annotations and automatic candidates."""
def merge_events(annotations, candidates, tolerance=.75):
    possible={i:[j for j,c in enumerate(candidates) if c['run']==a['run'] and c['outcome']==a['outcome']
        and abs(c['time_seconds']-a['reference_seconds'])<=tolerance] for i,a in enumerate(annotations)}
    reverse={j:[i for i,js in possible.items() if j in js] for j in range(len(candidates))}
    linked=set();records=[]
    for i,a in enumerate(annotations):
        unique=possible[i][0] if len(possible[i])==1 and len(reverse[possible[i][0]])==1 else None
        if unique is not None:linked.add(unique)
        records.append({**a,'origin':'source_annotation','candidate_refs':[candidates[unique]] if unique is not None else [],
            'ambiguous_candidate_ids':[candidates[j]['id'] for j in possible[i]] if unique is None else []})
    for j,c in enumerate(candidates):
        if j not in linked:records.append({'run':c['run'],'reference_seconds':c['time_seconds'],'outcome':c['outcome'],
            'origin':'candidate_only','annotation_provenance':c['label_provenance'],'candidate_refs':[c],
            'ambiguous_candidate_ids':[c['id']] if reverse[j] else []})
    return records
