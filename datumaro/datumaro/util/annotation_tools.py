
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from itertools import groupby

import numpy as np

from datumaro.components.extractor import AnnotationType, RleMask
from datumaro.util.mask_tools import mask_to_rle


def find_instances(instance_anns):
    instance_anns = sorted(instance_anns, key=lambda a: a.group)
    ann_groups = []
    for g_id, group in groupby(instance_anns, lambda a: a.group):
        if not g_id:
            ann_groups.extend(([a] for a in group))
        else:
            ann_groups.append(list(group))

    return ann_groups

def find_group_leader(group):
    return max(group, key=lambda x: x.get_area())

def compute_bbox(annotations):
    boxes = [ann.get_bbox() for ann in annotations]
    x0 = min((b[0] for b in boxes), default=0)
    y0 = min((b[1] for b in boxes), default=0)
    x1 = max((b[0] + b[2] for b in boxes), default=0)
    y1 = max((b[1] + b[3] for b in boxes), default=0)
    return [x0, y0, x1 - x0, y1 - y0]

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def nms(boxes, iou_thresh=0.5):
    """
    Non-maxima suppression algorithm.
    """

    indices = np.argsort([b.attributes['score'] for b in boxes])
    ious = np.array([[a.iou(b) for b in boxes] for a in boxes])

    predictions = []
    while len(indices) != 0:
        i = len(indices) - 1
        pred_idx = indices[i]
        to_remove = [i]
        predictions.append(boxes[pred_idx])
        for i, box_idx in enumerate(indices[:i]):
            if iou_thresh < ious[pred_idx, box_idx]:
                to_remove.append(i)
        indices = np.delete(indices, to_remove)

    return predictions

def bbox_iou(bbox_a, bbox_b):
    """
    IoU computations for simple cases with bounding boxes
    """

    aX, aY, aW, aH = bbox_a
    bX, bY, bW, bH = bbox_b
    in_right = min(aX + aW, bX + bW)
    in_left = max(aX, bX)
    in_top = max(aY, bY)
    in_bottom = min(aY + aH, bY + bH)

    in_w = max(0, in_right - in_left)
    in_h = max(0, in_bottom - in_top)
    intersection = in_w * in_h

    a_area = aW * aH
    b_area = bW * bH
    union = a_area + b_area - intersection

    return intersection / max(1.0, union)

def iou(a, b):
    """
    Generic IoU computation with masks, polygons, and boxes.
    Returns -1 if no intersection, [0; 1] otherwise
    """
    try:
        from pycocotools.mask import iou as _iou

        is_bbox = AnnotationType.bbox in [a.type, b.type]
        if is_bbox:
            a = [a.get_bbox()]
            b = [b.get_bbox()]
        else:
            def _to_rle(ann):
                from pycocotools import mask as mask_utils
                if ann.type == AnnotationType.polygon:
                    x, y, w, h = ann.get_bbox()
                    w = x + w
                    h = y + h
                    return mask_utils.frPyObjects([ann.points], h, w)
                elif isinstance(ann, RleMask):
                    return [ann._rle]
                elif ann.type == AnnotationType.mask:
                    x, y, w, h = ann.get_bbox()
                    w = x + w
                    h = y + h
                    return mask_utils.frPyObjects(mask_to_rle(ann.image), h, w)
            a = _to_rle(a)
            b = _to_rle(b)
        return float(_iou(a, b, [not is_bbox]))
    except ImportError:
        return bbox_iou(a, b)
