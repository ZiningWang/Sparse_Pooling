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
pretrained_model = '/data/RPN/fromBDD/MV3D_TF/output/default/train/voxelnet_train_peds_at0110_0047/VGGnet_fast_rcnn_iter_70000.ckpt.meta'

roidb = get_training_roidb(imdb)

from roi_data_layer.layer import RoIDataLayer
data_layer = RoIDataLayer(roidb, imdb.num_classes)
blobs = data_layer.forward()
while blobs['gt_rys'].shape[0]>0:
	blobs = data_layer.forward()

#find a specific sample
while blobs['im_id']!='000000':
	print (blobs['im_id'])
	blobs = data_layer.forward()

network = get_network('kitti_voxeltrain',use_bn=True,use_focal=False,use_fusion=True)
rpn_data = network.get_output('rpn_data')
rpn_label = tf.reshape(rpn_data[0],[-1])
t_anchor = rpn_data[4]
#front view
rpn_cls_score = network.get_output('rpn_cls_score')
VFE_feature = network.get_output('feature_output')
img_features_pooled = network.get_output('img_features_pooled')
lidar_features = network.get_output('lidar_features')
img_features = network.get_output('img_features')

rpn_keep = tf.reshape(tf.where(tf.not_equal(rpn_label,-1)),[-1])
# only regression positive anchors
rpn_bbox_keep = tf.reshape(tf.where(tf.greater(rpn_label, 0)),[-1])

config = tf.ConfigProto(allow_soft_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
sess.run(tf.global_variables_initializer())


#use input to exam sparse pooling
Mij_pool0,M_val0,M_size0,img_index_flip_pool0 = network.produce_sparse_pooling_input(blobs['img_index'],blobs['img_size'],
                                                    blobs['bv_index'],blobs['bv_size'],M_val=blobs['M_val'],stride=[1,1]) #blobs['M_val']
print ('M_val ', M_val0)
Mij_tf0 = tf.placeholder(tf.int64,shape=(None,2))
M_val_tf0 = tf.placeholder(tf.float32,shape=(None))
M_size_tf0= tf.placeholder(tf.int64,shape=(2))
img_index_flip_tf0 = tf.placeholder(tf.int32,shape=(None,3))
image_data0 = tf.placeholder(tf.float32, shape=[None, None, None, 3])
M_tf0 = tf.SparseTensor(indices=Mij_tf0,values=M_val_tf0,
                          dense_shape=M_size_tf0)
pooled_size = [1,200,240,3]

img_pooled_tf = tf.gather_nd(image_data0,img_index_flip_tf0)
bv_flat_tf = tf.sparse_tensor_dense_matmul(M_tf0,img_pooled_tf)
img_features_pooled0 = tf.reshape(bv_flat_tf,pooled_size)
#img_features_pooled0 = sparse_pool([M_tf0,image_data0,img_index_flip_tf0],pooled_size)
feed_dict= {Mij_tf0:Mij_pool0,
			M_val_tf0:M_val0,
			M_size_tf0:M_size0,
			img_index_flip_tf0:img_index_flip_pool0,
			image_data0:blobs['image_data']}
img_features_pooled0_debug,bv_flat_tf_debug = sess.run([img_features_pooled0,bv_flat_tf],feed_dict=feed_dict)
img_features_pooled0_debug = np.sum(np.absolute(np.squeeze(img_features_pooled0_debug)),axis=-1)
bv_flat_tf_debug = np.absolute(bv_flat_tf_debug)
print ('M_size: ',M_size0)
print ('pooled feature: ', img_features_pooled0_debug.shape)
print ('bv_index ',blobs['bv_index'][:2,:])
print ('bv_size: ', blobs['bv_size'])
print ('img_index ',blobs['img_index'][:,:2])
plt.figure()
plt.subplot(2,1,1)
plt.imshow(img_features_pooled0_debug)
voxel_full = np.zeros((200,240))
voxel_full[blobs['bv_index'][:,1],blobs['bv_index'][:,0]]=200
plt.subplot(2,1,2)
plt.imshow(voxel_full)
plt.show()
#whole network
'''
saver = tf.train.Saver(max_to_keep=10)
network.load(pretrained_model, sess, saver, True)
Mij_pool,M_val,M_size,img_index_flip_pool = network.produce_sparse_pooling_input(blobs['img_index'],blobs['img_size'],
                                                    blobs['bv_index'],blobs['bv_size'],M_val=None) #blobs['M_val']
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
rpn_cls_score_debug,VFE_feature_debug,img_features_pooled_debug,lidar_features_debug,img_features_debug = sess.run([rpn_cls_score,
				VFE_feature,img_features_pooled,lidar_features,img_features],
                feed_dict=feed_dict) 
img_features_pooled_debug = np.sum(np.absolute(np.squeeze(img_features_pooled_debug)),axis=-1)
lidar_features_debug = np.sum(np.absolute(np.squeeze(lidar_features_debug)),axis=-1)
img_features_debug = np.sum(np.absolute(np.squeeze(img_features_debug)),axis=-1)
VFE_feature_debug = np.squeeze(np.sum(VFE_feature_debug,axis=1))
print ('VFE plot shape: ', VFE_feature_debug.shape)
print ('pooled feature: ', img_features_pooled_debug.shape)
print ('orig img feature: ', img_features_debug.shape)
print ('Lidar conv feature: ', lidar_features_debug.shape)
img_plot = img_features_pooled_debug
lidar_plot = lidar_features_debug
img_orig_plot = img_features_debug
VFE_plot = VFE_feature_debug[:,:,0:3]
plt.figure()
plt.subplot(2,2,1)
plt.imshow(scale_to_255(img_plot,np.amin(img_plot),np.amax(img_plot)))
plt.subplot(2,2,2)
plt.imshow(scale_to_255(lidar_plot,np.amin(lidar_plot),np.amax(lidar_plot)))
plt.subplot(2,2,3)
plt.imshow(scale_to_255(img_orig_plot,np.amin(img_orig_plot),np.amax(img_orig_plot)))
plt.subplot(2,2,4)
plt.imshow(scale_to_255(VFE_plot,np.amin(VFE_plot),np.amax(VFE_plot)))
plt.show()
'''