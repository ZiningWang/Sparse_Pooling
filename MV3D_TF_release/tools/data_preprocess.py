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
imdb = get_imdb('kitti_trainval')

import read_lidar_VOXEL_test as rl
root_dir = '/data/RPN/mscnn-master/data/testing'#'/data/RPN/mscnn-master/data/training'
velodyne = os.path.join(root_dir, "velodyne/")
bird = os.path.join(root_dir, "lidar_pc/")
if not os.path.exists(bird)::
    os.mkdir(bird)
    
t0 = time.time()

for ind in xrange(imdb.num_images):
    filename = velodyne + str(ind).zfill(6) + ".bin"
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    im = cv2.imread(imdb.image_path_at(ind))
    img_size = np.array([im.shape[1],im.shape[0]])
    calib = imdb.calib_at(ind)

    P = calib_to_P(calib)
    indices = clip3DwithinImage(scan[:,0:3].transpose(),P,img_size)
    scan = scan[indices,:]
    scan[:,0:3] = lidar_to_camera(scan[:,0:3].transpose(),calib)
    np.save(bird+str(ind).zfill(6)+".npy",scan.astype(np.float32))
    
    #lidar_append = rl.lidar_bv_append(scan,calib,img_size,in_camera_frame=True)
    #np.save(bird+str(ind).zfill(6)+"_append.npy",lidar_append)
    '''DEBUG only
    scan = np.load(bird+str(ind).zfill(6)+".npy")
    P = calib_to_P(calib,from_camera=True)
    img_points = projectToImage(scan[:,0:3].transpose(),P)
    '''

print ('overall %d seconds'%(int(time.time()-t0)))