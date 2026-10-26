"""Deterministic author-label sampling and cross-batch original-game identity."""
from collections import Counter

from basketball_poc.outcome_data import game_folds


def select_batch(games, excluded, per_class=2):
    if not isinstance(per_class, int) or per_class < 1:
        raise ValueError('Positive per-class count required')
    selected=[];seen=set();game_ids=set()
    for data in sorted(games,key=lambda d:d['game_id']):
        game=data['game_id']
        if game in game_ids:raise ValueError('Duplicate game metadata')
        game_ids.add(game);counts=Counter()
        for row in sorted(data['clips'],key=lambda r:r['clip_file']):
            name=row['clip_file']
            if name in seen:raise ValueError('Duplicate source clip')
            seen.add(name)
            if not name.startswith(game+'_clip'):raise ValueError('Clip/game identity mismatch')
            shots=[e for e in row['events'] if e['event']=='shot']
            if len(shots)!=1:continue
            label=shots[0]['result']
            if label not in ('make','miss'):raise ValueError('Unknown author class')
            if name in excluded or counts[label]>=per_class:continue
            selected.append({'game':game,'author_label':label,'annotation':row,
                'remote':'clips/'+game+'/'+name})
            counts[label]+=1
        if counts!=Counter({'make':per_class,'miss':per_class}):
            raise ValueError('Insufficient source classes for '+game)
    return selected


def combine_batches(packages):
    records=[];clips=set();ids=set()
    for package in packages:
        for r in package['records']:
            if r['id'] in ids or r['clip_id'] in clips:raise ValueError('Duplicate cross-batch source')
            if not r['clip_id'].startswith(r['game']+'_clip'):raise ValueError('Original game mismatch')
            ids.add(r['id']);clips.add(r['clip_id']);records.append(r)
    ready=[r for r in records if r['readiness']['provisional_training_ready']]
    return {'records':records,'folds':game_folds(records),'summary':{
        'events':len(records),'provisional_training_ready':len(ready),
        'provisional_labels':dict(Counter(r['annotation']['outcome'] for r in ready)),
        'by_game':{g:dict(Counter(r['annotation']['outcome'] for r in ready if r['game']==g))
                   for g in sorted({r['game'] for r in records})},'training':False}}
