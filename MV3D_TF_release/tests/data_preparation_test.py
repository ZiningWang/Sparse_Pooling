import os,sys
#add library to the system path
lib_path = os.path.abspath(os.path.join('lib'))
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('tools'))
sys.path.append(lib_path)
import numpy as np
import cv2
from utils.transform import calib_to_P,clip3DwithinImage,projectToImage,lidar_to_camera
import time

from datasets.factory import get_imdb
imdb = get_imdb('kitti_test')
from fast_rcnn.train_mv_voxel import get_training_roidb
roidb = get_training_roidb(imdb)

from roi_data_layer.layer import RoIDataLayer
data_layer = RoIDataLayer(roidb, imdb.num_classes)
blobs = data_layer.forward()
for key in blobs:
    print key
print blobs['bv_size']
print np.amax(blobs['bv_index'],axis=0)
print blobs['im_info']
for key in blobs['voxel_data']:
	print blobs['voxel_data'][key].shape

t0 = time.time()
for ind in xrange(imdb.num_images):
	blobs = data_layer.forward()
print 'overall %d seconds, average: %d ms.'%(int(time.time()-t0),int(1000*(time.time()-t0)/imdb.num_images) )
