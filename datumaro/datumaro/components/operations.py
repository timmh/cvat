
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from itertools import chain
import numpy as np

from datumaro.components.extractor import AnnotationType, Bbox
from datumaro.components.project import Dataset
from datumaro.util.annotation_tools import iou as segment_iou, \
    find_group_leader, compute_bbox


SEGMENT_TYPES = {
    AnnotationType.bbox,
    AnnotationType.polygon,
    AnnotationType.mask
}

def merge_annotations(a, b):
    merged = []
    for item in chain(a, b):
        found = False
        for elem in merged:
            if elem == item:
                found = True
                break
        if not found:
            merged.append(item)

    return merged

def merge_categories(sources):
    categories = {}
    for source in sources:
        categories.update(source)
    for source in sources:
        for cat_type, source_cat in source.items():
            if not categories[cat_type] == source_cat:
                raise NotImplementedError(
                    "Merging different categories is not implemented yet")
    return categories

def merge_datasets(a, b, iou_threshold=1.0, conf_threshold=1.0):
    # TODO: put this function to the right place
    merged_categories = merge_categories([a.categories(), b.categories()])
    merged = Dataset(categories=merged_categories)
    for item_a in a:
        item_b = b.get(item_a.id, subset=item_a.subset)

        # TODO: / note: must shift groups to avoid a and b intersection
        group_pad = max((a.group for a in item_a.annotations), default=0)
        for a in item_b.annotations:
            a.group += group_pad

        annotations = merge_segments(item_a.annotations + item_b.annotations,
            iou_threshold=iou_threshold, conf_threshold=conf_threshold)
        a_non_segments = filter(lambda a: a.type not in SEGMENT_TYPES,
            item_a.annotations)
        b_non_segments = filter(lambda a: a.type not in SEGMENT_TYPES,
            item_b.annotations)
        annotations += merge_annotations(a_non_segments, b_non_segments)
        merged.put(item_a.wrap(image=Dataset._merge_images(item_a, item_b),
            annotations=annotations))
    return merged

def merge_segments(anns, iou_threshold=1.0, conf_threshold=1.0):
    anns = get_segments(anns, conf_threshold=conf_threshold)
    clusters, _ = find_segment_clusters(anns, iou_threshold=iou_threshold)

    merged = []
    for cluster in clusters:
        if len(cluster) == 1: # leave singular untouched
            merged.extend(cluster)
            continue

        label, score = find_cluster_label(cluster)
        bbox = compute_bbox(cluster)
        attributes = {'score': score} if label is not None else None
        merged.append(Bbox(*bbox, label=label, attributes=attributes))
    return merged

def get_segments(anns, conf_threshold=1.0):
    return [ann for ann in anns \
        if conf_threshold <= ann.attributes.get('score', 1) and \
            ann.type in SEGMENT_TYPES
    ]

def find_cluster_label(cluster):
    label_votes = {}
    votes_count = 0
    visited_groups = set() # avoid group domination by population
    for s in cluster:
        if s.label is None:
            continue

        weight = 1.0
        if s.group is not None:
            if s.group in visited_groups:
                continue
            visited_groups.add(s.group)

            # in fact, do NMS (by area) in the group
            leader = find_group_leader(r for r in cluster
                if r.group == s.group)
            weight = leader.attributes.get('score', 1.0)

        label_votes[s.label] = weight + label_votes.get(s.label, 0)
        votes_count += 1

    label, score = max(label_votes.items(), key=lambda e: e[1], default=None)
    score = score / votes_count if votes_count else None
    return label, score

def find_segment_clusters(segments, iou_threshold=1.0):
    # build filtered IoU matrix
    ious = np.eye(len(segments))
    for i, a in enumerate(segments):
        for j, b in enumerate(segments[i+1:]):
            j = i + 1 + j
            iou = segment_iou(a, b)
            ious[i, j] = iou
            ious[j, i] = iou

    # find clusters
    clusters = []
    visited = set()
    for cluster_idx, _ in enumerate(segments):
        if cluster_idx in visited:
            continue

        cluster = set()
        to_visit = {cluster_idx}
        while to_visit:
            c = to_visit.pop()
            cluster.add(c)
            visited.add(c)

            for i in range(c+1, len(segments)):
                if i not in visited and iou_threshold <= ious[c, i]:
                    to_visit.add(i)

        clusters.append([segments[i] for i in cluster])

    return clusters, ious

