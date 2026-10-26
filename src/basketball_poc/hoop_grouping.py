"""Experimental shared-top observation hypothesis, separate from physical truth."""
from itertools import combinations

from basketball_poc.hoop_assignment import associate as global_associate
from basketball_poc.scene import iou


def duplicate_metrics(a, b, experiment):
    a, b = a['xyxy'], b['xyxy']
    wa, wb = a[2]-a[0], b[2]-b[0]
    ha, hb = a[3]-a[1], b[3]-b[1]
    if min(wa, wb, ha, hb) <= 0:
        raise ValueError('Invalid hoop observation extent')
    width = min(wa, wb)
    top = abs(a[1]-b[1])/width
    center = abs((a[0]+a[2]-b[0]-b[2])/2)/width
    ratio = max(wa/wb, wb/wa)
    intersection = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0, min(a[3],b[3])-max(a[1],b[1]))
    containment = intersection/min(wa*ha, wb*hb)
    overlap = iou(a, b)
    shared = (ratio <= experiment['shared_top_max_width_ratio'] and
              top <= experiment['shared_top_max_offset_widths'] and
              center <= experiment['shared_center_max_offset_widths'] and
              containment >= experiment['shared_min_containment'])
    return {'iou':overlap, 'width_ratio':ratio, 'top_offset_widths':top,
            'center_offset_widths':center, 'containment':containment,
            'shared_top':shared, 'duplicate':overlap > experiment['duplicate_iou'] or shared}


def group_observations(detections, experiment):
    ordered = sorted(detections, key=lambda b:(-b['confidence'], tuple(b['xyxy'])))
    pairs = {(i,j):duplicate_metrics(ordered[i],ordered[j],experiment)
             for i,j in combinations(range(len(ordered)),2)}
    groups = []
    for i in range(len(ordered)):
        selected = next((g for g in groups if all(pairs[j,i]['duplicate'] for j in g)), None)
        if selected is None:
            groups.append([i])
        else:
            selected.append(i)
    representatives = [g[0] for g in groups]
    return [ordered[i] for i in representatives], {
        'raw_observations':ordered, 'groups':groups, 'representative_indices':representatives,
        'pair_metrics':[{'a':i,'b':j,**m} for (i,j),m in pairs.items()],
        'physical_sameness_verified':False}


def associate(propagated, detections, profile, experiment, next_id):
    representatives, grouping = group_observations(detections, experiment)
    # Complete-link grouping replaces NMS. A second NMS could collapse a chain
    # through its representative despite incompatible endpoints.
    association_experiment = {**experiment, 'duplicate_iou':1.0}
    hoops, resets, next_id, diagnostic = global_associate(
        propagated, representatives, profile, association_experiment, next_id)
    diagnostic['observation_grouping'] = grouping
    diagnostic['assignment_duplicate_iou'] = 1.0
    return hoops, resets, next_id, diagnostic
