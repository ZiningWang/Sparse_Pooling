import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from rpn_msr.generate_anchors import generate_anchors_3d_angle
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_inv
from utils.transform import bv_anchor_to_lidar, bv_center_to_lidar, reassign_z
import pdb, time
from utils.util_voxels import bbox_transform_voxel
from utils.config_voxels import cfg as cfg_voxel

_allowed_border =  0
im_border = [cfg_voxel.X_MIN,cfg_voxel.Y_MIN,cfg_voxel.X_MAX,cfg_voxel.Y_MAX] #fwd_min,side_min,fwd_max,side_max
def clip_anchors_voxel(all_anchors,im_border=im_border):
    inds_inside = np.where(
        (all_anchors[:, 0] >= im_border[1]-_allowed_border) &
        (all_anchors[:, 1] >= im_border[0]-_allowed_border) &
        (all_anchors[:, 2] < im_border[3] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_border[2] + _allowed_border)    # height
    )[0]
    return inds_inside


def anchor_target_layer_voxel(rpn_cls_score, gt_boxes_3d, gt_ry, im_info,  _feat_stride = [16,], use_reward=False,DEBUG = False):
    t0 = time.time()
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    gt_boxes_3d = np.insert(gt_boxes_3d,6,0, axis=1)
    gt_boxes_3d[:,6:7] = gt_ry

    _anchors_3d_bbox,_anchors = generate_anchors_3d_angle()
    # anchors by default are in bird view in bv pixels (lidar bv image coord)
    _num_anchors = _anchors.shape[0]

    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]


    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.linspace(cfg_voxel.X_MIN, cfg_voxel.X_MAX, width).astype(np.float32)  #foward
    shift_y = np.linspace(cfg_voxel.Y_MIN, cfg_voxel.Y_MAX, height).astype(np.float32) #side
    
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel(), shift_x.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))


    shifts = np.vstack((shift_y.ravel(), 0*shift_y.ravel(),
                        shift_x.ravel(), 0*shift_y.ravel(), 0*shift_y.ravel(), 0*shift_y.ravel(), 0*shift_y.ravel())).transpose()
    all_anchors_3d_bbox = (_anchors_3d_bbox.reshape((1, A, 7)) +
                   shifts.reshape((1, K, 7)).transpose((1, 0, 2)))
    all_anchors_3d_bbox = all_anchors_3d_bbox.reshape((K * A, 7))
    all_anchors_3d_bbox[:,1] = reassign_z(all_anchors_3d_bbox[:,1])

    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = clip_anchors_voxel(all_anchors,im_border)

    if DEBUG:
        print ('rpn_cls_score shape:', rpn_cls_score.shape)
        print ('shiftx shape: ', shift_x.shape)
        print ('anchor bounds: ', np.amax(all_anchors,axis=0), np.amin(all_anchors,axis=0))
        print ('3D anchor bounds: ', np.amax(all_anchors_3d_bbox[:,0:3],axis=0), np.amin(all_anchors_3d_bbox[:,0:3],axis=0))
        print ('inds_inside :',inds_inside.shape)
        print ('total_anchors: ', total_anchors)

    #print 'time for clip: ', time.time()-t0
    #t0 = time.time()

    # WZN: only keep gt_boxes inside the image'
    gt_boxes_cnr = gt_boxes_3d[:,[0,2,3,4,7]]
    gt_boxes_cnr[:,[0,1]] -= gt_boxes_cnr[:,[2,3]]/2
    gt_boxes_cnr[:,[2,3]] += gt_boxes_cnr[:,[0,1]]

    inds_inside_box = np.where(
        (gt_boxes_cnr[:, 0] >= im_border[1]-_allowed_border) &
        (gt_boxes_cnr[:, 1] >= im_border[0]-_allowed_border) &
        (gt_boxes_cnr[:, 2] < im_border[3] + _allowed_border) &  # width
        (gt_boxes_cnr[:, 3] < im_border[2] + _allowed_border) &   # height
        (gt_boxes_cnr[:, 4] > 0) #only keep object gt_boxes
    )[0]
    gt_boxes_cnr_pos = gt_boxes_cnr[inds_inside_box,:]
    gt_boxes_3d_pos = gt_boxes_3d[inds_inside_box,:]

    if DEBUG:
        print ('inds_inside_box :',inds_inside_box.shape)
        print ('gt_boxes_3d_pos: ', gt_boxes_3d_pos)

    if gt_boxes_cnr_pos.shape[0]==0:
        no_gt=True
        gt_boxes_cnr_pos = np.reshape(np.array([-1000,-1000,-1000,-1000,-1]),[1,-1])
    else:
        no_gt=False

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    anchors_3d_bbox = all_anchors_3d_bbox[inds_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes_cnr_pos, dtype=np.float))


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
        labels[max_overlaps <= cfg_voxel.RPN_NEG_IOU] = 0
        labels[gt_argmax_overlaps] = 1
        labels[max_overlaps >= cfg_voxel.RPN_POS_IOU] = 1

        #print "was %s inds, disabling %s, now %s inds" % (
            #len(bg_inds), len(disable_inds), np.sum(labels == 0))

    inds_pos = (labels>0)
    inds_inside_pos = inds_inside[inds_pos]

    if DEBUG:
        print ('positive labels: ', inds_inside_pos.shape)
        print ('gt_max_overlaps: ', gt_max_overlaps)
        print ('max_overlaps: ', max_overlaps[inds_pos])
        print ('negative labels: ', inds_inside.shape[0]-inds_inside_pos.shape[0])
        print ('labels: ', labels.shape)
        print ('argmax_overlaps: ', argmax_overlaps.shape)
    
    if no_gt:
        bbox_targets = np.reshape(np.array([0.0,0,0,0,0,0,0]),[1,7])
    else:
        bbox_targets = _compute_targets_3d_angle(anchors_3d_bbox[inds_pos,:], gt_boxes_3d_pos[argmax_overlaps[inds_pos], :])
        # disable those angle difference too large to negative
        negative_angle_inds = np.absolute(bbox_targets[:,6]) > cfg_voxel.ANCHOR_ANGLE_THRESOLD
        if DEBUG:
            print ('disabled targets:',bbox_targets[negative_angle_inds,:])
        bbox_targets[negative_angle_inds,:] = 0


    # map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    reward_and_orig_labels = _unmap(reward_and_orig_labels, total_anchors, inds_inside, fill=0)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside_pos, fill=0)

    if not(no_gt):
        # disable those angle difference too large to negative
        labels[inds_inside_pos[negative_angle_inds]] = 0
        if DEBUG:
            targets_value = np.sum(np.absolute(bbox_targets),axis=1)
            assert np.all((labels>0)==(targets_value>0)), 'labeling and targeting mismatch'


    #if debug, reshaspe label to feature map size
    if DEBUG:
        reward_and_orig_labels = np.reshape(labels,[height, width,np.round(rpn_cls_score.shape[3]/2).astype(int)])

    # labels
    rpn_labels = labels
    rpn_bbox_targets = bbox_targets
    t1 = np.float32(time.time() - t0)
    #print rpn_labels.dtype,rpn_bbox_targets.dtype, all_anchors_3d_bbox.dtype, reward_and_orig_labels.dtype
    return rpn_labels, rpn_bbox_targets, all_anchors_3d_bbox, reward_and_orig_labels, t1 # origin: anchors_3d, WZN: changed to rewards

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



def _compute_targets_3d_angle(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    # output is Nx7
    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 7
    assert gt_rois.shape[1] == 8

    return bbox_transform_voxel(ex_rois, gt_rois[:, :7]).astype(np.float32, copy=False)