# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from rpn_msr.generate_anchors import generate_anchors_bv, generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_3d, bbox_transform_inv
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

def anchor_target_layer(rpn_cls_score, gt_boxes, gt_boxes_3d, im_info,  _feat_stride = [16,], anchor_scales = [8, 16, 32],rpn_bbox_pred=None,use_reward=False,DEBUG = False):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    #t0 = time.time()

    _anchors = generate_anchors_bv()
    # anchors by default are in bird view in bv pixels (lidar bv image coord)
    _num_anchors = _anchors.shape[0]

    if DEBUG:
        print ('anchors:')
        print (_anchors.shape)
        print ('anchor shapes:')
        print (np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        )))
        _counts = cfg.EPS
        _sums = np.zeros((1, 6))
        _squared_sums = np.zeros((1, 6))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0


    im_info = im_info[0]


    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    if DEBUG:
        print ('AnchorTargetLayer: height', height, 'width', width)
        print ('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print ('scale: {}'.format(im_info[2]))
        print ('height, width: ({}, {})'.format(height, width))
        print ('rpn: gt_boxes.shape', gt_boxes.shape)
        print ('rpn: gt_boxes', gt_boxes)
        gt_bboxes = gt_boxes.copy()
        gt_bboxes[:,0] = (gt_boxes[:,0]+gt_boxes[:,2])/2
        gt_bboxes[:,1] = (gt_boxes[:,1]+gt_boxes[:,3])/2
        gt_bboxes[:,2] = (gt_boxes[:,2]-gt_boxes[:,0])
        gt_bboxes[:,3] = (gt_boxes[:,3]-gt_boxes[:,1])
        print ('rpn: gt_bboxes', gt_bboxes)
        print ('feat_stride', _feat_stride)

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
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
    gt_boxes_pos = gt_boxes[inds_inside_box,:]
    gt_boxes_3d_pos = gt_boxes_3d[inds_inside_box,:]
    if gt_boxes_pos.shape[0]==0:
        no_gt=True
        gt_boxes_pos = np.reshape(np.array([-1000,-1000,-1000,-1000,-1]),[1,-1])
    else:
        no_gt=False

    if DEBUG:
        print ('total_anchors: ', total_anchors)
        print ('inds_inside: ', len(inds_inside))

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print ('anchors.shape: ', anchors.shape)

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes_pos, dtype=np.float))
    # WZN: additional overlap calculation using bbox prediction
    if not(rpn_bbox_pred is None):
        bbox_deltas = rpn_bbox_pred
        bbox_deltas = bbox_deltas.reshape((-1, 6))
        bbox_deltas = bbox_deltas[inds_inside,:]
        bbox_deltas = bbox_deltas[:,[0,1,3,4]]
        assert bbox_deltas.shape[0] == anchors.shape[0]
        assert bbox_deltas.shape[1] == 4
        proposals_bv = bbox_transform_inv(anchors, bbox_deltas)
        overlaps1 = bbox_overlaps(
                    np.ascontiguousarray(proposals_bv, dtype=np.float),
                    np.ascontiguousarray(gt_boxes_pos, dtype=np.float))
        overlaps = np.maximum(overlaps,overlaps1)

    #for each anchor, which object has biggest overlap?
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

    #for each object, which achor has biggest overlap?
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    #WZN: only keep those > 0
    gt_max_overlaps = gt_max_overlaps[gt_max_overlaps>0]

    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if DEBUG:
        print ('argmax_overlaps indices: ', gt_argmax_overlaps)
        print ('argmax_overlaps: ',gt_max_overlaps)

    reward_and_orig_labels = np.zeros((overlaps.shape[0],2))
    reward_and_orig_labels[:,0] = max_overlaps.copy()

    #print 'time for sorting: ', time.time()-t0
    #t0 = time.time()

    if use_reward:
        thres = cfg.TRAIN.THRES
        if no_gt:
            reward_and_orig_labels[:,0] *= 0
        else:
            pos_ind = max_overlaps > thres
            rewards_pos = reward_and_orig_labels[pos_ind,0]
            argmax_overlaps_pos = argmax_overlaps[pos_ind]
            for igt in range(overlaps.shape[1]):
                igt_index = argmax_overlaps_pos==igt
                #print np.sum(igt_index)
                temp_rewards = rewards_pos[igt_index]
                #normalize the reward
                if igt_index.size>1:
                    rewards_pos[igt_index] = (temp_rewards-np.mean(temp_rewards))/(np.std(temp_rewards)+1e-2)
                else:
                    rewards_pos[igt_index] = temp_rewards
            reward_and_orig_labels[pos_ind,0] = rewards_pos

        labels *= 0
        #labels[gt_argmax_overlaps] = 1
        if not no_gt:
            labels[pos_ind] = 1 # from 1 to N_gt ground truth objects
            reward_and_orig_labels[gt_argmax_overlaps,1] = 1
            reward_and_orig_labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP,1] = 1
        #print gt_boxes.shape[0],gt_boxes_pos.shape[0], np.sum(max_overlaps > thres)
        #rewards[max_overlaps> thres] = max_overlaps[max_overlaps> thres]

    elif cfg.TRAIN.RPN_HAS_BATCH:
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them

            # hard negative for proposal_target_layer
            hard_negative = np.logical_and(0 < max_overlaps, max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP)
            labels[hard_negative] = 0

        #WZN: add more negatives if there are not enough
        num_bg = np.sum(hard_negative) 
        if np.sum(hard_negative) < cfg.TRAIN.RPN_BATCHSIZE:
            bg_inds = np.where(np.logical_and(max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP,np.logical_not(hard_negative)))[0]
            enable_inds = npr.choice(
                bg_inds, size=(cfg.TRAIN.RPN_BATCHSIZE - num_bg), replace=False)
            labels[enable_inds] = 0


        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1
        if DEBUG:
            print ('number of hard negatives: ', num_bg)
            print ('resampled easy negatives: ', np.sum(labels == 0) - num_bg)
            print ('resampled indices: ', enable_inds, len(enable_inds))
            print ('bg_inds: ', bg_inds,len(bg_inds))


        # random sample 

        # fg label: above threshold IOU
        # print np.where(max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP)

        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
            
        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1
    else: 
        #use all anchors
        labels *= 0
        labels[gt_argmax_overlaps] = 1
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        #print "was %s inds, disabling %s, now %s inds" % (
            #len(bg_inds), len(disable_inds), np.sum(labels == 0))

    inds_pos = (labels>0)
    inds_inside_pos = inds_inside[inds_pos]
    anchors_3d_pos = bv_anchor_to_lidar(anchors[inds_pos,:])
    if no_gt:
        bbox_targets = np.reshape(np.array([0.0,0,0,0,0,0]),[1,6])
    else:
        bbox_targets = _compute_targets_3d(anchors_3d_pos, gt_boxes_3d_pos[argmax_overlaps[inds_pos], :])

    anchors =  np.array([0]*5).astype(np.float32)# np.hstack((zeros, anchors[all_inds])).astype(np.float32)


    if DEBUG:
        #_sums += bbox_targets[labels >= 1, :].sum(axis=0)
        #_squared_sums += (bbox_targets[labels >= 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print ('means:', means)
        print ('stdevs:', stds)

    if DEBUG:
        print ('gt_boxes_3d: ', gt_boxes_3d_pos[argmax_overlaps, :].shape)
        print ('labels shape before unmap: ', labels.shape)
        print ('targets shape before unmap: ', bbox_targets.shape)
    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    reward_and_orig_labels = _unmap(reward_and_orig_labels, total_anchors, inds_inside, fill=0)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside_pos, fill=0)

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
        print ('bbox_targets', bbox_targets.shape)
        print ('rewards pos: ', rewards_pos, np.mean(rewards_pos))


    # labels
    rpn_labels = labels
    rpn_bbox_targets = bbox_targets

    if DEBUG:
        print ('labels shape: ', labels.shape)
        print ('targets shape: ', bbox_targets.shape)

    #print 'time for unmap: ', time.time()-t0
    #t0 = time.time()
    #print '--------------------'

    return rpn_labels, rpn_bbox_targets, anchors, reward_and_orig_labels# origin: anchors_3d, WZN: changed to rewards



def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

def _compute_targets_3d(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 6
    assert gt_rois.shape[1] == 7

    return bbox_transform_3d(ex_rois, gt_rois[:, :6]).astype(np.float32, copy=False)

    
def cnr_target_layer(rpn_cls_score, gt_boxes_cnr, gt_boxes_cnr_3d, im_info,  _feat_stride = [16,],rpn_bbox_pred=None,use_reward=False,DEBUG = False):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    #t0 = time.time()

    # allow boxes to sit over the edge by a small amount
    
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]

    im_info = im_info[0]

    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    centers = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    _num_centers = centers.shape[0]
    inds_inside = np.ones(_num_centers,dtype=bool)
    #print 'time for clip: ', time.time()-t0
    #t0 = time.time()

    # WZN: only keep gt_boxes inside the image
    inds_inside_box = np.where(
        (np.amin(gt_boxes_cnr[:, 0:4],axis=1) >= -_allowed_border) &
        (np.amin(gt_boxes_cnr[:, 4:8],axis=1) >= -_allowed_border) &
        (np.amax(gt_boxes_cnr[:, 0:4],axis=1)< im_info[1] + _allowed_border) &  # width
        (np.amax(gt_boxes_cnr[:, 4:8],axis=1) < im_info[0] + _allowed_border) &   # height
        (gt_boxes_cnr[:, 9] > 0) #only keep object gt_boxes
    )[0]
    gt_boxes_cnr_pos = gt_boxes_cnr[inds_inside_box,:]
    gt_boxes_cnr_3d_pos = gt_boxes_cnr_3d[inds_inside_box,:]
    if gt_boxes_cnr_pos.shape[0]==0:
        no_gt=True
        gt_boxes_cnr_pos = np.reshape(np.array([-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1]),[1,-1])
        gt_boxes_pos = np.reshape(np.array([-1000,-1000,-1000,-1000,-1]),[1,-1])
    else:
        no_gt=False
        gt_boxes_pos = np.zeros((gt_boxes_cnr_pos.shape[0],5))
        gt_boxes_pos[:,0] = np.amin(gt_boxes_cnr_pos[:, 0:4],axis=1) 
        gt_boxes_pos[:,1] = np.amin(gt_boxes_cnr_pos[:, 4:8],axis=1) 
        gt_boxes_pos[:,2] = np.amax(gt_boxes_cnr_pos[:, 0:4],axis=1) 
        gt_boxes_pos[:,3] = np.amin(gt_boxes_cnr_pos[:, 0:4],axis=1) 
        gt_boxes_pos[:,4] = gt_boxes_cnr_pos[:,9]


    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((_num_centers, ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = normed_distance_to_box_bv(centers.astype(np.float),gt_boxes_cnr_pos.astype(np.float))

    #print 'time for overlap: ', time.time()-t0
    #t0 = time.time()

    labels,argmax_overlaps,reward_and_orig_labels = calc_label_and_reward(overlaps,_num_centers,use_reward=use_reward)

    