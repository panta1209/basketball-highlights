"""Fixed source-group linear probes; predictions stay diagnostic."""
import re

import numpy as np
import torch
import torch.nn.functional as F


def original_group(source_path):
    match=re.search(r'BV[0-9A-Za-z]+',source_path)
    if not match:
        raise ValueError('Original source ID must be explicit')
    return 'bilibili:'+match.group()


def sampling_plan(rows, event_seconds, fps):
    stride=max(1,round(fps/15))
    available={r['frame']:i for i,r in enumerate(rows)}
    planned=[]
    for offset in (-.4,0,.4):
        middle=round((event_seconds+offset)*fps)
        indices=[middle-8*stride+j*stride for j in range(16)]
        if any(fi not in available for fi in indices):
            return {'eligible':False,'reason':'sample outside available source window','stride':stride}
        planned.append(indices)
    mask=[[rows[available[fi]]['valid'] for fi in clip] for clip in planned]
    return {'eligible':bool(np.all(mask)),'reason':None if np.all(mask) else 'sampled target identity missing',
            'stride':stride,'source_indices':planned,'local_indices':[[available[fi] for fi in clip] for clip in planned],
            'valid_mask':mask,'effective_fps':fps/stride}


def source_folds(records):
    groups=sorted({r['source_group'] for r in records})
    if len(groups)<2:
        raise ValueError('Need at least two original sources')
    folds=[]
    for group in groups:
        train=[i for i,r in enumerate(records) if r['source_group']!=group and r['plan']['eligible']]
        test=[i for i,r in enumerate(records) if r['source_group']==group and r['plan']['eligible']]
        if not train or not test:
            raise ValueError('Empty train/test group')
        if {records[i]['label'] for i in train}!={0,1}:
            raise ValueError('Training fold lacks a result class')
        folds.append({'held_source':group,'train_indices':train,'test_indices':test})
    return folds


def fit_head(features,labels,penalty=.01):
    x=torch.as_tensor(features,dtype=torch.float64)
    if not torch.isfinite(x).all() or x.ndim!=2:
        raise ValueError('Invalid training features')
    x=F.normalize(x,p=2,dim=1);mean=x.mean(0);x=x-mean
    y=torch.as_tensor(labels,dtype=torch.float64)
    if set(y.tolist())!={0,1}:raise ValueError('Both outcome classes required')
    counts=torch.bincount(y.long(),minlength=2).double()
    sample_weights=len(y)/(2*counts[y.long()])
    w=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    b=torch.zeros((),dtype=torch.float64,requires_grad=True)
    optimizer=torch.optim.LBFGS([w,b],lr=1,max_iter=100,line_search_fn='strong_wolfe')
    evaluations=[]
    def closure():
        optimizer.zero_grad()
        loss=(F.binary_cross_entropy_with_logits(x@w+b,y,reduction='none')*sample_weights).mean()+penalty*(w*w).sum()/2
        loss.backward();evaluations.append(float(loss.detach()));return loss
    optimizer.step(closure)
    result={'weights':w.detach().tolist(),'bias':float(b.detach()),'train_mean':mean.tolist(),
            'penalty':penalty,'class_counts':counts.int().tolist(),'closure_loss':evaluations,
            'optimizer':'LBFGS lr1 max_iter100 strong_wolfe','feature_normalization':'per-row L2 then train-only mean'}
    if not np.isfinite(result['weights']).all():raise ValueError('Non-finite fitted head')
    return result


def score(features,head):
    x=torch.as_tensor(features,dtype=torch.float64)
    x=F.normalize(x,p=2,dim=1)-torch.tensor(head['train_mean'],dtype=torch.float64)
    return (x@torch.tensor(head['weights'],dtype=torch.float64)+head['bias']).numpy()


def confusion(labels,predictions):
    y=np.asarray(labels);p=np.asarray(predictions)
    tp=int(((y==1)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
    return {'samples':len(y),'TP':tp,'TN':tn,'FP':fp,'FN':fn,
            'accuracy':(tp+tn)/len(y) if len(y) else None,
            'balanced_accuracy':((tp/(tp+fn))+(tn/(tn+fp)))/2 if tp+fn and tn+fp else None,
            'made_precision':tp/(tp+fp) if tp+fp else None,
            'made_recall':tp/(tp+fn) if tp+fn else None}
