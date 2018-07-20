# WZN: Note here we unify all LIDAR points to camera frame!!!

import numpy as np
import yaml
from fast_rcnn.config import cfg
from rpn_msr.anchor_target_layer_voxel_tf import clip_anchors_voxel
from fast_rcnn.nms_wrapper import nms
from utils.transform import lidar_cnr_to_img
from utils.util_voxels import bbox_transform_inv_voxel
import pdb,time


#DEBUG = False
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""

def proposal_layer_voxel(rpn_cls_prob_reshape,rpn_bbox_pred,rpn_anchors_3d_bbox,im_info,cfg_key, _feat_stride = [8,], anchor_scales=[1.0, 1.0],DEBUG = False):
    # Algorithm:
    #
    if type(cfg_key) is bytes:
        cfg_key = cfg_key.decode('UTF-8','ignore')
        
    t0 = time.time()
    #_anchors = generate_anchors_bv()
    #  _anchors = generate_anchors(scales=np.array(anchor_scales))
    #_num_anchors = _anchors.shape[0]
    #print rpn_cls_prob_reshape.shape

    im_info = im_info[0]

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'
    # cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'

    pre_score_filt = cfg[cfg_key].RPN_SCORE_FILT
    pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
    min_size      = cfg[cfg_key].RPN_MIN_SIZE

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want

    height, width = rpn_cls_prob_reshape.shape[1:3]
    # scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :]
    scores = np.reshape(np.reshape(rpn_cls_prob_reshape, [1, height, width, -1, 2])[:,:,:,:,1],[1, height, width, -1])
    _num_anchors = scores.shape[-1]
    #print scores.shape
    assert _num_anchors == 2, 'wrong number of anchors: %d'%(_num_anchors)

    bbox_deltas = rpn_bbox_pred


    KA = rpn_anchors_3d_bbox.shape[0]
    K = int(width*height)

    
    bbox_deltas = bbox_deltas.reshape((-1, 7))

    scores = scores.reshape((-1, 1))
    score_filter = scores[:,0] > pre_score_filt
    #WZN: pre score filt
    scores = scores[score_filter,:]
    rpn_anchors_3d_bbox = rpn_anchors_3d_bbox[score_filter,:]
    bbox_deltas = bbox_deltas[score_filter,:]

    #print 'time for score pre_filt: ', time.time()-t0, scores.shape
    #t0 = time.time()

    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0 and pre_nms_topN<order.shape[0]:
        order = order[:pre_nms_topN]
    scores = scores[order,:]
    rpn_anchors_3d_bbox = rpn_anchors_3d_bbox[order,:]
    bbox_deltas = bbox_deltas[order,:]

    # print np.sort(scores.ravel())[-30:]

    # Convert anchors into proposals via bbox transformations
    proposals_3d = bbox_transform_inv_voxel(rpn_anchors_3d_bbox, bbox_deltas)
    # convert back to lidar_bv (x1,z1,x2,z2)
    proposal_bv_cam = np.zeros([proposals_3d.shape[0],4],dtype=np.float32)
    proposal_bv_cam[:,0] = proposals_3d[:,0]-proposals_3d[:,3]/2
    proposal_bv_cam[:,2] = proposals_3d[:,0]+proposals_3d[:,3]/2
    proposal_bv_cam[:,1] = proposals_3d[:,2]-proposals_3d[:,4]/2
    proposal_bv_cam[:,3] = proposals_3d[:,2]+proposals_3d[:,4]/2



    #WZN: delete those not in image
    all_anchors = np.zeros([rpn_anchors_3d_bbox.shape[0],4],dtype=np.float32)
    all_anchors[:,0] = rpn_anchors_3d_bbox[:,0]-rpn_anchors_3d_bbox[:,3]/2
    all_anchors[:,1] = rpn_anchors_3d_bbox[:,2]-rpn_anchors_3d_bbox[:,4]/2
    all_anchors[:,2] = all_anchors[:,0]+rpn_anchors_3d_bbox[:,3]
    all_anchors[:,3] = all_anchors[:,1]+rpn_anchors_3d_bbox[:,4]

    ind_inside = clip_anchors_voxel(all_anchors)
    #ind_inside = np.logical_and(ind_inside,clip_anchors(proposals_bv, im_info[:2]))
    proposal_bv_cam = proposal_bv_cam[ind_inside,:]
    proposals_3d = proposals_3d[ind_inside,:]
    #proposals_img = proposals_img[ind_inside,:]
    scores = scores[ind_inside,:]


    # 2. clip predicted boxes to image

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])

    #keep = _filter_boxes(proposal_bv_cam, min_size * im_info[2])
    #proposal_bv_cam = proposal_bv_cam[keep, :]
    #proposals_3d = proposals_3d[keep, :]
    #proposals_img = proposals_img[keep, :]
    #scores = scores[keep]

    #WZN: discard
    '''
    # TODO: pass real image_info
    keep = _filter_img_boxes(proposals_img, [375, 1242])
    proposals_bv = proposals_bv[keep, :]
    proposals_3d = proposals_3d[keep, :]
    proposals_img = proposals_img[keep, :]
    scores = scores[keep]
    '''
    # 4. sort all (proposal, score) pairs by score from highest to lowest
    ''' WZN: moved to upper to save time
    # 5. take top pre_nms_topN (e.g. 6000) 
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals_bv = proposals_bv[order, :]
    proposals_3d = proposals_3d[order, :]
    proposals_img = proposals_img[order, :]
    scores = scores[order]
    '''

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposal_bv_cam, scores)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposal_bv_cam = proposal_bv_cam[keep, :]
    proposals_3d = proposals_3d[keep, :]
    #proposals_img = proposals_img[keep, :]
    scores = scores[keep]

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposal_bv_cam.shape[0], 1), dtype=np.float32)
    blob_bv_cam = np.hstack((batch_inds, proposal_bv_cam.astype(np.float32, copy=False)))
    #blob_img = np.hstack((batch_inds, proposals_img.astype(np.float32, copy=False)))
    blob_3d = np.hstack((batch_inds, proposals_3d.astype(np.float32, copy=False)))


    t1 = np.float32(time.time() - t0)
    return blob_bv_cam, blob_3d, scores,t1




def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    #WZN: filter boxes too far away
    ds = (boxes[:, 3] + boxes[:, 1])/2
    keep = np.where((ws >= min_size) & (hs >= min_size) )[0] #& (ds<460)
    return keep

def _filter_img_boxes(boxes, im_info):
    """Remove all boxes with any side smaller than min_size."""
    padding = 50
    w_min = -padding
    w_max = im_info[1] + padding
    h_min = -padding
    h_max = im_info[0] + padding
    keep = np.where((w_min <= boxes[:,0]) & (boxes[:,2] <= w_max) & (h_min <= boxes[:,1]) &
                    (boxes[:,3] <= h_max))[0]
    return keep
