"""Fixed larger square video inputs for diagnostic outcome probes."""
import numpy as np
import torch
import torch.nn.functional as F

METHOD = {'encoder':'torchvision R3D18 KINETICS400_V1 frozen', 'resize':[256,256], 'crop':[224,224],
    'mean':[.43216,.394666,.37645], 'std':[.22803,.22145,.216989], 'frames_per_window':16,
    'windows':3,'offset_seconds':[-.4,0,.4], 'pool':'mean of three 512D vectors',
    'static':'index8 repeated16 per window', 'reverse':'reverse per window, temporal head without refit',
    'head':'balanced logistic LBFGS lr1 max_iter100 strong_wolfe float64',
    'l2':.01,'boundary_logit':0,'seed':0,'threads':4,'normalizer':'L2 then train-only mean',
    'official_preprocessing':False,'user_labels_used_for_training':False,'default_promotion':False}


def tensor_input(frames, plan):
    if not plan or not plan['eligible']:
        raise ValueError('Missing input must remain unknown')
    clips = np.stack([[frames[i] for i in window] for window in plan['local_indices']])
    if clips.shape != (3,16,384,384,3) or clips.dtype != np.uint8:
        raise ValueError('Expected three native RGB uint8 16-frame windows')
    x = torch.from_numpy(clips).permute(0,1,4,2,3).float()/255
    x = F.interpolate(x.reshape(48,3,384,384),size=(256,256),mode='bilinear',align_corners=False)
    x = x[:,:,16:240,16:240].reshape(3,16,3,224,224).permute(0,2,1,3,4).contiguous()
    return ((x-torch.tensor(METHOD['mean'])[None,:,None,None,None])/
        torch.tensor(METHOD['std'])[None,:,None,None,None]), clips


def controls(x):
    return {'temporal':x,'static':x[:,:,8:9].repeat(1,1,16,1,1),'reverse':x.flip(2)}


def eligible_indices(records, cohort):
    return [i for i,r in enumerate(records) if r['cohort']==cohort and r['eligible']]


def folds(records):
    public = eligible_indices(records,'public'); result=[]
    for group in sorted({records[i]['source_group'] for i in public}):
        train=[i for i in public if records[i]['source_group']!=group]
        test=[i for i in public if records[i]['source_group']==group]
        if {records[i]['label'] for i in train} != {0,1} or {records[i]['label'] for i in test} != {0,1}:
            raise ValueError('Both classes required in source-held train and test')
        result.append({'held_source':group,'train_indices':train,'test_indices':test})
    if len(result)<2: raise ValueError('Multiple original games required')
    return result
