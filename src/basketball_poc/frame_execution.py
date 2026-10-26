"""Scoped OpenCV scheduling and exact candidate-window net crops."""
from contextlib import contextmanager

import cv2
import numpy as np


def case_inventory(cases):
    return [{'id': c['id'], 'frame_indices': c['frame_indices'], 'context_roi': c['context_roi']}
            for c in cases]


@contextmanager
def frame_execution(net, threads=1, net_mode='full'):
    if threads not in (1, 2, 4, 8) or net_mode not in ('full', 'windows'):
        raise ValueError('Invalid frame execution options')
    previous_threads, previous_decode = cv2.getNumThreads(), net.decode_cases
    record = {'opencv_threads': threads, 'net_mode': net_mode, 'net_passes': [],
              'legacy_count_semantics': 'total source frame inventory, not actual net reads'}

    def decode(source, cases):
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise ValueError('Cannot open net source')
            count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            wanted = set()
            for case in cases:
                indices = case['frame_indices']
                if (not indices or any(type(i) is not int or not 0 <= i < count for i in indices)
                        or indices != sorted(set(indices))):
                    raise ValueError('Invalid net window indices')
                wanted.update(indices)
            wanted = sorted(wanted)
            crops = {c['id']: [] for c in cases}
            previous = -2
            for index in wanted:
                if index != previous+1 and not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise ValueError('Cannot seek to net window')
                if abs(capture.get(cv2.CAP_PROP_POS_FRAMES)-index) > .01:
                    raise ValueError('Net source seek index mismatch')
                ok, image = capture.read()
                if not ok:
                    raise ValueError('Net window decoding incomplete')
                for case in cases:
                    if index in case['frame_indices']:
                        x1,y1,x2,y2 = case['context_roi']
                        crops[case['id']].append(image[y1:y2,x1:x2].copy())
                previous = index
            for case in cases:
                if len(crops[case['id']]) != len(case['frame_indices']):
                    raise ValueError('Net crop inventory incomplete')
                crops[case['id']] = np.stack(crops[case['id']])
            record['net_passes'].append({'source_frame_inventory': count,
                'decoded_frame_indices': wanted, 'decoded_frames': len(wanted),
                'cases': case_inventory(cases), 'actual_opencv_threads': cv2.getNumThreads()})
            return crops, count
        finally:
            capture.release()

    cv2.setNumThreads(threads)
    if net_mode == 'windows':
        net.decode_cases = decode
    try:
        yield record
    finally:
        net.decode_cases = previous_decode
        cv2.setNumThreads(previous_threads)


def verify_frame_execution(record, cases, count):
    if record.get('opencv_threads') not in (1, 2, 4, 8) or record.get('net_mode') not in ('full', 'windows'):
        raise ValueError('Invalid frame execution evidence')
    if record['net_mode'] == 'windows':
        passes = record.get('net_passes', [])
        wanted = sorted({i for case in cases for i in case['frame_indices']})
        if (len(passes) != 1 or passes[0]['source_frame_inventory'] != count
                or passes[0]['decoded_frame_indices'] != wanted
                or passes[0]['decoded_frames'] != len(wanted)
                or passes[0]['cases'] != case_inventory(cases)
                or passes[0]['actual_opencv_threads'] != record['opencv_threads']):
            raise ValueError('Net window execution coverage mismatch')
