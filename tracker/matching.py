#pretty much igual mas ele acrescenta 2 funcowes chamadas new_distance e old_distance, not sure para h'a cpoisas diferentes nos imports basicamente porque o joao mudou files

import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from tracker.cython_bbox import bbox_overlaps as bbox_ious
from tracker import kalman_filter
import math as m
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def old_distance(atracks, btracks):
    """
    Compute cost based on the distance
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh for track in atracks]
        btlbrs = [track.tlwh for track in btracks]
        
    
    cost_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    boxes = np.ascontiguousarray(atlbrs, dtype=np.float)
    query_boxes = np.ascontiguousarray(btlbrs, dtype=np.float)
    
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    for k in range(K):
        box_area_k = query_boxes[k, 2] * query_boxes[k, 3]
        aspect_ratio_k = query_boxes[k, 2] / query_boxes[k, 3]
        for n in range(N):   
            box_area_n = boxes[n, 2] * boxes[n, 3] + 1
            aspect_ratio_n = boxes[n, 2] / boxes[n, 3]
            dist = boxes[n, 0:2]-query_boxes[k, 0:2]             
            cost_matrix[n, k] = 1 - (
                m.exp(-2*np.matmul(dist,dist)/820 ) *
                min(box_area_k,box_area_n)/max(box_area_k,box_area_n) *
                min(aspect_ratio_k, aspect_ratio_n)/max(aspect_ratio_k, aspect_ratio_n)
                )

    return cost_matrix

def new_distance(atracks, btracks):
    """
    Compute cost based on the distance
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks.copy()
        btlbrs = btracks.copy()
    else:
        atlbrs = [track.tlwh for track in atracks]
        btlbrs = [track.tlwh for track in btracks]
        
    
    cost_matrix = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    boxes = np.ascontiguousarray(atlbrs, dtype=np.float)
    query_boxes = np.ascontiguousarray(btlbrs, dtype=np.float)
    
    boxes[:,0:2] += boxes[:,2:4]*0.5 #change from top left to center
    query_boxes[:,0:2] += query_boxes[:,2:4]*0.5 #change from top left to center
    
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    for k in range(K):
        box_area_k = query_boxes[k, 2] * query_boxes[k, 3]
        #aspect_ratio_k = query_boxes[k, 2] / query_boxes[k, 3]
        for n in range(N):   
            box_area_n = boxes[n, 2] * boxes[n, 3] + 1
            #aspect_ratio_n = boxes[n, 2] / boxes[n, 3]
            dist_x = (boxes[n, 0] - query_boxes[k, 0]) / boxes[n, 2]
            dist_y = (boxes[n, 1] - query_boxes[k, 1]) / boxes[n, 3]
            cost_matrix[n, k] = 1 - (
                m.exp(- m.sqrt(dist_x**2 + dist_y**2)) *
                min(box_area_k,box_area_n)/max(box_area_k,box_area_n)
                #min(aspect_ratio_k, aspect_ratio_n)/max(aspect_ratio_k, aspect_ratio_n)
                )

    return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost