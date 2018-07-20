# --------------------------------------------------------
# WZN: THIS IS FOR CORNER BASED regression TARGET
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from rpn_msr.generate_anchors import generate_anchors_bv, generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_3d, bbox_transform_inv,bbox_transform_cnr_3d
from utils.transform import bv_anchor_to_lidar,bv_center_to_lidar
import pdb, time


_allowed_border =  0

def clip_anchors(all_anchors,im_info):
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]
    return inds_inside


def anchor_fv_target_layer(rpn_cls_score, gt_boxes, im_info,  _feat_stride = [16,], anchor_scales = [8, 16, 32],DEBUG = False, num_class=2):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels. NOTE: no box regression targets!
    """
    #t0 = time.time()

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap


    _anchors = generate_anchors(base_size=_feat_stride[0], ratios=np.array([1.54]),
                     scales=np.array([4,3]))
    # anchors by default are in bird view in bv pixels (lidar bv image coord)

    _num_anchors = _anchors.shape[0]
    
    #print 'time for anchors: ', time.time()-t0
    #t0 = time.time()

    im_info = im_info[0]
    #print 'arrive!!'

    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width, channels = rpn_cls_score.shape[1:4]

    num_scores = channels/num_class
    assert num_scores==_num_anchors, 'number of anchors does not match number of proposal scores'

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    
    # only keep anchors inside the image
    inds_inside = clip_anchors(all_anchors,im_info)

    #print 'time for clip: ', time.time()-t0
    #t0 = time.time()

    # WZN: only keep gt_boxes inside the image
    inds_inside_box = np.where(
        (gt_boxes[:, 0] >= -_allowed_border) &
        (gt_boxes[:, 1] >= -_allowed_border) &
        (gt_boxes[:, 2] < im_info[1] + _allowed_border) &  # width
        (gt_boxes[:, 3] < im_info[0] + _allowed_border) &   # height
        (gt_boxes[:, 4] > 0) #only keep object gt_boxes
    )[0]
    inds_ignore_box = np.where(gt_boxes[:, 4] == -1)[0]

    gt_boxes_pos = gt_boxes[inds_inside_box,:]
    gt_boxes_ignore = gt_boxes[inds_ignore_box,:]
    if gt_boxes_pos.shape[0]==0:
        no_gt=True
        gt_boxes_pos = np.reshape(np.array([-1000,-1000,-1000,-1000,-1]),[1,-1])
    else:
        no_gt=False

    if gt_boxes_ignore.shape[0]==0:
        no_ignore=True
    else:
        no_ignore=False

    if DEBUG:
        print ('total_anchors: ', total_anchors)
        print ('inds_inside: ', len(inds_inside))

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print ('anchors.shape: ', anchors.shape)


    #print 'time for clip2: ', time.time()-t0
    #t0 = time.time()

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes_pos, dtype=np.float))

    overlaps_ignore = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes_ignore, dtype=np.float))

    labels,_,_ = calc_label_and_reward(overlaps,len(inds_inside),cfg,no_gt,use_reward=False,multi_scale=True)


    #also for ignore labels, we need -1 for them
    if not(no_ignore):
        argmax_overlaps_ignore = overlaps_ignore.argmax(axis=1)
        max_overlaps_ignore = overlaps_ignore[np.arange(len(inds_inside)), argmax_overlaps_ignore]
        # WZN: disable those that captures the ignore windows
        labels[max_overlaps_ignore >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = -1


    # print 'labels = 0:, ', np.where(labels == 0)
    #all_inds = np.where(labels != -1)
    #labels_new = labels[all_inds]
    #zeros = np.zeros((labels_new.shape[0], 1), dtype=np.float32)
    anchors =  np.array([0]*5).astype(np.float32)# np.hstack((zeros, anchors[all_inds])).astype(np.float32)
    #anchors_3d =  np.hstack((zeros, anchors_3d[all_inds])).astype(np.float32)

    #print 'time for targets: ', time.time()-t0
    #t0 = time.time()

    if DEBUG:
        #_sums += bbox_targets[labels >= 1, :].sum(axis=0)
        #_squared_sums += (bbox_targets[labels >= 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)

        print ('labels shape before unmap: ', labels.shape)

    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)

    if DEBUG:
        print ('max_overlaps.shape: ', max_overlaps.shape)
        print ('rpn: max max_overlap', np.max(max_overlaps))
        max_100_ind = np.argsort(max_overlaps)[-50:]
        print ('rpn: max 50 overlaps: ', max_overlaps[max_100_ind])
        print ('rpn: max 50 indices:', max_100_ind)
        print ('rpn: num_positive', np.sum(labels >= 1))
        print ('rpn: num_negative', np.sum(labels == 0))
        _fg_sum += np.sum(labels >= 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print ('rpn: num_positive avg', _fg_sum / _count)
        print ('rpn: num_negative avg', _bg_sum / _count)
        #print 'fg inds: ', fg_inds
        print ('label shape', labels.shape)


    # labels
    rpn_labels = labels
    if DEBUG:
        print ('labels shape: ', labels.shape)

    #print 'time for unmap: ', time.time()-t0
    #t0 = time.time()
    #print '--------------------'

    return rpn_labels, anchors # origin: anchors_3d, WZN: changed to rewards