"""Source-wide scale diagnostics; tile placement never depends on scene coordinates."""
import math
from .rim_probe import restore_rows
from .scene import merge_boxes

def tiles(width,height,fraction):
    if type(width) is not int or type(height) is not int or min(width,height)<2:
        raise ValueError('Positive source dimensions required')
    if not math.isfinite(fraction) or not .5<=fraction<1:
        raise ValueError('Overlapping tile fraction must be in [.5,1)')
    w,h=math.ceil(width*fraction),math.ceil(height*fraction)
    return [list(region) for region in dict.fromkeys((x,y,x+w,y+h) for y in (0,height-h) for x in (0,width-w))]

def restore_call(rows,region,call_id):
    boxes=restore_rows(rows,region,{0:'ball',1:'hoop'})
    return [{**b,'call_id':call_id,'local_index':index} for index,b in enumerate(boxes)]

def merge_calls(calls,threshold):
    return merge_boxes([b for c in calls for b in c['source_boxes']],threshold)
