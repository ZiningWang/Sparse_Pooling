import os,sys
#add library to the system path
lib_path = os.path.abspath(os.path.join('lib'))
sys.path.append(lib_path)
from fast_rcnn.config import cfg
from utils.transform import calib_to_P,computeCorners3D,projectToImage,lidar_to_camera
from utils.util_voxels import lidar_in_cam_to_voxel
import matplotlib.pyplot as plt
from utils.draw import drawBox3D, show_image_boxes, scale_to_255
from rpn_msr.anchor_target_layer_voxel_tf import anchor_target_layer_voxel as anchor_target_layer_voxel_py
import numpy as np
from datasets.factory import get_imdb
from networks.factory import get_network
import tensorflow as tf
imdb = get_imdb('kitti_train_small')
imdb.set_object('peds')
from fast_rcnn.train_mv_voxel import get_training_roidb

roidb = get_training_roidb(imdb)

from roi_data_layer.layer import RoIDataLayer
data_layer = RoIDataLayer(roidb, imdb.num_classes)
blobs = data_layer.forward()
while not blobs['gt_rys'].shape[0]>0:
	blobs = data_layer.forward()


im_info = blobs['im_info']
print (im_info)
H = int(im_info[0][1]/2)
W = int(im_info[0][0]/2)
rpn_cls_score = np.zeros([1,H,W,4])
gt_boxes_3d = blobs['gt_boxes_3d']
gt_ry = blobs['gt_rys']

print (' gt_boxes_3d.shape: ',gt_boxes_3d.shape)

rpn_labels, rpn_bbox_targets, all_anchors_3d_bbox, reward_and_orig_labels, t1 = anchor_target_layer_voxel_py(rpn_cls_score, gt_boxes_3d, gt_ry, im_info,  _feat_stride = 2, use_reward=False,DEBUG = True)

print (rpn_labels.shape, rpn_bbox_targets.shape, all_anchors_3d_bbox.shape, reward_and_orig_labels.shape)


#show ground truth
gt_boxes_3d = blobs['gt_boxes_3d']
gt_ry = blobs['gt_rys']
gt_corners = np.zeros((gt_ry.shape[0],24))
print (gt_ry)
for i in range(gt_ry.shape[0]):
	gt_corners[i,:] = computeCorners3D(gt_boxes_3d[i,:], gt_ry[i,:]).flatten()
gt_voxel_cnr = lidar_in_cam_to_voxel(gt_corners.reshape((-1, 3, 8)))
voxel_coord = blobs['voxel_data']['coordinate_buffer']
voxel_size = blobs['im_info'][0]
print ('im_info: ',blobs['im_info'][0])
#show voxels in transposed version so it suits my prefered POV
voxel_full = np.zeros((int(voxel_size[0]),int(voxel_size[1])))

for i in range(gt_voxel_cnr.shape[0]):
	voxel_full = drawBox3D(voxel_full,gt_voxel_cnr[i,:,:])
print ('gt_boxes_3d: ',gt_boxes_3d)
print ('gt_rys: ', gt_ry)
print ('anchors_3d: ', all_anchors_3d_bbox[rpn_labels>0,:])

print ('targets: ',rpn_bbox_targets[rpn_labels>0,:])
print (' voxel_full.shape: ',voxel_full.shape)

voxel_full[voxel_coord[:,3],voxel_coord[:,2]] = 100
#print (np.amax(voxel_full,axis=0))
'''
plt.figure()
img_bv = np.flipud(voxel_full)#scale_to_255(voxel_full, min=0, max=2)
plt.imshow(img_bv)
'''

testing_scores=True
if testing_scores:
	#Testing the scores output by the network
	network = get_network('kitti_voxeltrain',use_bn=True,use_focal=False)
	rpn_data = network.get_output('rpn_data')
	rpn_label = tf.reshape(rpn_data[0],[-1])
	t_anchor = rpn_data[4]
	#front view
	rpn_cls_score = network.get_output('rpn_cls_score')
	VFE_feature = network.get_output('feature_output')

	rpn_keep = tf.reshape(tf.where(tf.not_equal(rpn_label,-1)),[-1])
	# only regression positive anchors
	rpn_bbox_keep = tf.reshape(tf.where(tf.greater(rpn_label, 0)),[-1])

	config = tf.ConfigProto(allow_soft_placement=False)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config) 
	sess.run(tf.global_variables_initializer())
	Mij_pool,M_val,M_size,img_index_flip_pool = network.produce_sparse_pooling_input(blobs['img_index'],blobs['img_size'],
	                                                    blobs['bv_index'],blobs['bv_size'],M_val=None)

	# Make one SGD update
	feed_dict={network.image_data: blobs['image_data'],
	           network.vox_feature: blobs['voxel_data']['feature_buffer'],
	           network.vox_coordinate: blobs['voxel_data']['coordinate_buffer'],
	           network.vox_number: blobs['voxel_data']['number_buffer'],
	           network.im_info: blobs['im_info'],
	           network.im_info_fv: blobs['im_info_fv'],
	           network.keep_prob: 0.5,
	           network.gt_boxes: blobs['gt_boxes'],
	           network.gt_boxes_3d: blobs['gt_boxes_3d'],
	           network.gt_rys: blobs['gt_rys'],        
	           network.Mij_tf: Mij_pool,
	           network.M_val_tf: M_val,
	           network.M_size_tf: M_size,
	           network.img_index_flip_tf: img_index_flip_pool}
	rpn_cls_score_debug,VFE_feature_debug = sess.run([rpn_cls_score,VFE_feature],
	                feed_dict=feed_dict) 
	print ('network score shape: ', rpn_cls_score_debug.shape, 'all_anchors_3d_bbox shape: ', all_anchors_3d_bbox.shape)
	rpn_labels, rpn_bbox_targets, all_anchors_3d_bbox, rpn_labels_reshaped, t1 = anchor_target_layer_voxel_py(rpn_cls_score_debug, gt_boxes_3d, gt_ry, im_info,  _feat_stride = 2, use_reward=False,DEBUG = True)

	active_anchors_3d = all_anchors_3d_bbox[rpn_labels>0,:]
	print (rpn_bbox_targets[rpn_labels>0,:])

	rpn_labels_reshaped = np.sum(rpn_labels_reshaped,axis=2)
	print ('label reshaped shape: ', rpn_labels_reshaped.shape, 'voxel shape: ', voxel_full.shape, 'anchors shape: ',all_anchors_3d_bbox.shape)
	print ('non-zero labels: ',np.where(rpn_labels_reshaped>0), 'not reshaped: ',np.where(rpn_labels>0))
	print ('gt_boxes: \n', gt_voxel_cnr)


	#VFE features
	VFE_feature_debug = np.amax(np.squeeze(VFE_feature_debug),axis=0)
	VFE_feature_debug = np.amax(VFE_feature_debug,axis=2)
	print ('feature map after VFE: ', VFE_feature_debug.shape)
	plt.figure()
	non_zero_voxel_coord = [np.where(rpn_labels_reshaped>0)[0],np.where(rpn_labels_reshaped>0)[1]]
	non_zero_voxel_coord[0]*=2
	non_zero_voxel_coord[1]*=2
	VFE_feature_debug[non_zero_voxel_coord]=100
	plt.imshow(VFE_feature_debug)
	#plt.show()


testing_overlap = True
if testing_overlap:
	#Testing the anchor overlaps
	active_anchors_3d = all_anchors_3d_bbox[rpn_labels>0,:]
	print (rpn_bbox_targets[rpn_labels>0,:])
	#print 'active_anchors_3d: \n', active_anchors_3d
	active_cnrs = np.zeros((active_anchors_3d.shape[0],24))
	for i in range(active_anchors_3d.shape[0]):
		active_cnrs[i,:] = computeCorners3D(active_anchors_3d[i,:], active_anchors_3d[i,6]).flatten()
	active_voxel_cnr = lidar_in_cam_to_voxel(active_cnrs.reshape((-1, 3, 8)))
	print (active_voxel_cnr)
	for i in range(active_voxel_cnr.shape[0]):
		pass#voxel_full = drawBox3D(voxel_full,active_voxel_cnr[i,:,:])

	
	plt.figure()
	img_bv = np.flipud(voxel_full)#scale_to_255(voxel_full, min=0, max=2)
	plt.imshow(img_bv)
	#plt.show()
	
	img_plot = np.squeeze(blobs['image_data'])#+cfg.PIXEL_MEANS
	print (img_plot.shape)
	img_mean = np.mean(np.mean(img_plot,axis=0),axis=0)
	print (img_mean)
	img_plot+=cfg.PIXEL_MEANS
	img_mean = np.mean(np.mean(img_plot,axis=0),axis=0)
	print (img_mean)
	plt.figure()
	plt.imshow(img_plot/255)

if testing_overlap or testing_scores:
	plt.show()
	

print ('image index: ', blobs['im_id'])