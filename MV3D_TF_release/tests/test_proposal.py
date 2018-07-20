import os,sys
#add library to the system path
lib_path = os.path.abspath(os.path.join('lib'))
sys.path.append(lib_path)

from rpn_msr.anchor_target_layer_voxel_tf import anchor_target_layer_voxel as anchor_target_layer_voxel_py
from rpn_msr.proposal_layer_voxel_tf import proposal_layer_voxel as proposal_layer_py_voxel
import numpy as np
from datasets.factory import get_imdb
imdb = get_imdb('kitti_trainval')
from fast_rcnn.train_mv_voxel import get_training_roidb
roidb = get_training_roidb(imdb)

from roi_data_layer.layer import RoIDataLayer
data_layer = RoIDataLayer(roidb, imdb.num_classes)
blobs = data_layer.forward()
while blobs['gt_rys'].shape[0]==0:
	blobs = data_layer.forward()

im_info = blobs['im_info']
print (im_info)
H = int(im_info[0][0]/2)
W = int(im_info[0][1]/2)
rpn_cls_score = np.random.randn(1,H,W,4)
gt_boxes_3d = blobs['gt_boxes_3d']
gt_ry = blobs['gt_rys']

print (gt_boxes_3d.shape)

rpn_labels, rpn_bbox_targets, all_anchors_3d_bbox, reward_and_orig_labels, t1 = anchor_target_layer_voxel_py(rpn_cls_score, gt_boxes_3d, gt_ry, im_info, _feat_stride = 2, use_reward=False,DEBUG = True)

rpn_cls_prob_reshape = rpn_cls_score.astype(np.float32)
rpn_bbox_pred = rpn_bbox_targets.astype(np.float32)
rpn_anchors_3d_bbox = all_anchors_3d_bbox.astype(np.float32)
cfg_key='TRAIN'
blob_bv_cam, blob_3d, scores, t2 = \
	proposal_layer_py_voxel(rpn_cls_prob_reshape,rpn_bbox_pred,rpn_anchors_3d_bbox,im_info,cfg_key, _feat_stride = 2, anchor_scales=[1.0, 1.0],DEBUG = False)

print (blob_bv_cam.shape, blob_3d.shape, scores.shape)
print (blob_bv_cam[0:3,:])
print (blob_3d[0:3,:])
print (scores[0:3,:])