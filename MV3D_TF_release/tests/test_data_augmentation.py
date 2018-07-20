import os,sys
#add library to the system path
lib_path = os.path.abspath(os.path.join('lib'))
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('tools'))
sys.path.append(lib_path)
import numpy as np
import cv2
from fast_rcnn.config import cfg
from utils.transform import calib_to_P,computeCorners3D,projectToImage,lidar_to_camera
from utils.util_voxels import lidar_in_cam_to_voxel
import time
import matplotlib.pyplot as plt
from utils.draw import drawBox3D, show_image_boxes, scale_to_255

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('object_name', type=str,default='cars')
parser.add_argument('--exp_name', type=str,default='N')
args_in = parser.parse_args()
object_name = args_in.object_name

from datasets.factory import get_imdb
imdb = get_imdb('kitti_train_small')
imdb.set_object(object_name)
from fast_rcnn.train_mv_voxel import get_training_roidb
roidb = get_training_roidb(imdb)

pos_roidb_num = 0
for i in range(len(roidb)):
	if roidb[i]['ry'].shape[0]>0:
		pos_roidb_num+=1
		print (roidb[i]['image_path'][-10:-4])
print ('positive roidbs: ', pos_roidb_num)

from roi_data_layer.layer import RoIDataLayer
data_layer = RoIDataLayer(roidb, imdb.num_classes)
blobs = data_layer.forward()

while blobs['gt_rys'].shape[0]==0:
	blobs = data_layer.forward()


while blobs['im_id']!='000103':
	print (blobs['im_id'])
	blobs = data_layer.forward()	

voxel_coord = blobs['voxel_data']['coordinate_buffer']
voxel_size = blobs['im_info'][0]
print (blobs['im_info'][0])
voxel_full = np.zeros((int(voxel_size[0]),int(voxel_size[1])))



#show ground truth
gt_boxes_3d = blobs['gt_boxes_3d']
gt_ry = blobs['gt_rys']
gt_corners = np.zeros((gt_ry.shape[0],24))
for i in range(gt_ry.shape[0]):
	gt_corners[i,:] = computeCorners3D(gt_boxes_3d[i,:], gt_ry[i,:]).flatten()
gt_voxel_cnr = lidar_in_cam_to_voxel(gt_corners.reshape((-1, 3, 8)))
print (gt_voxel_cnr)
for i in range(gt_voxel_cnr.shape[0]):
	voxel_full = drawBox3D(voxel_full,gt_voxel_cnr[i,:,:])
print (voxel_full.shape)


voxel_full[voxel_coord[:,3],voxel_coord[:,2]] = 200

plt.figure()
img_bv = np.flipud(voxel_full)#scale_to_255(voxel_full, min=0, max=2)
plt.imshow(img_bv)


# show data for fusion
print ('========================data for fusion=======================')
bv_size = blobs['bv_size']
bv_index = blobs['bv_index']
img_size = blobs['img_size']
img_index = blobs['img_index']
print ('bv_size: ', bv_size)
print ('bv_index: ',bv_index.shape)
print ('image_size: ',img_size)
print ('img_index: ', img_index.shape)
voxel_full2 = np.zeros((int(voxel_size[0]),int(voxel_size[1])))
voxel_full2[bv_index[:,0],bv_index[:,1]] = 200
plt.figure()
img_bv2 = np.flipud(voxel_full2)#scale_to_255(voxel_full, min=0, max=2)
plt.imshow(img_bv2)

img_full = np.zeros((img_size[0],img_size[1]))
print ('violation img_size[0]: ',np.sum(img_index[0,:]>img_size[0]) + np.sum(img_index[0,:]<0))
print ('violation img_size[1]: ',np.sum(img_index[1,:]>img_size[1]) + np.sum(img_index[1:]<0))
img_full[img_index[0,:],img_index[1,:]] = (240-bv_index[:,0])/255.0
plt.figure()
plt.subplot(2,1,1)
img_fv = img_full.transpose()#scale_to_255(voxel_full, min=0, max=2)
plt.imshow(img_fv)
plt.subplot(2,1,2)
img_plot = np.squeeze(blobs['image_data'])
img_plot+=cfg.PIXEL_MEANS
plt.imshow(img_plot/255)
print ('camera scaling: ',blobs['im_info_fv'])


plt.show()