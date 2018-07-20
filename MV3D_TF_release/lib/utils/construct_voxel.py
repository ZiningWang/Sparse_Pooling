import numpy as np
import matplotlib.pyplot as plt
import time
import os,sys
#add library to the system path
lib_path = os.path.abspath(os.path.join('lib'))
sys.path.append(lib_path)
from utils.transform import calib_to_P,clip3DwithinImage,projectToImage,lidar_to_camera
from utils.config_voxels import cfg

side_range = (cfg.Y_MIN, cfg.Y_MAX-0.01)
fwd_range = (cfg.X_MIN, cfg.X_MAX-0.01)
height_range = (cfg.Z_MIN, cfg.Z_MAX-0.01) #
res = cfg.VOXEL_X_SIZE
zres = cfg.VOXEL_Z_SIZE
NUM_VOXEL_FEATURES = 7
MAX_NUM_POINTS=cfg.VOXEL_POINT_COUNT

'''
def lidar_bv_append(scan,calib,img_size,in_camera_frame=False):
    #return the additional data for MV3D_img
    if not(in_camera_frame):
        P = calib_to_P(calib)
        indices = clip3DwithinImage(scan[:,0:3].transpose(),P,img_size)
        scan = scan[indices,:]
    else:
        P = calib_to_P(calib,from_camera=True)
    bv_index,scan_filtered,bv_size = point_in_bv_indexes(scan)
    N = bv_index.shape[0]
    img_points = projectToImage(scan_filtered,P)

    img_index = np.round(img_points).astype(int)
    img_index = np.vstack((img_index,np.zeros((1,N))))
    return {'bv_index':bv_index,'img_index':img_index,'bv_size':bv_size,'img_size':img_size}
'''

def point_cloud_2_top_sparse(points,
                      res=res,
                      zres=zres,
                      side_range=side_range,  # left-most to right-most
                      fwd_range=fwd_range,  # back-most to forward-most
                      height_range=height_range,  # bottom-most to upper-most
                      top_count = None,
                      to_camera_frame = False,
                      points_in_cam = False,
                      calib=None,
                      img_size = [0,0],
                      augmentation=False,
                      img_index2=None
                      ):
    """ Creates an birds eye view representation of the point cloud data for MV3D.
    WZN: NOTE to get maximum speed, should feed all LIDARs to the function because we wisely initialize the grid
    """
    #t0 = time.time()
    if to_camera_frame:
        indices = clip3DwithinImage(points[:,0:3].transpose(),P,img_size)
        points = points[indices,:]
        img_index2 = img_index2[:,indices]
        points[:,0:3] = lidar_to_camera(points[:,0:3].transpose(),calib)
        points_in_cam = True

    if points_in_cam:
        points = points[:,[2,0,1,3]] #forward, side, height
        #x_points = points[:, 1]
        #y_points = points[:, 2]
        #z_points = points[:, 0]
    else:
        assert False, 'Wrong, cannot process LIDAR coordinate points'
        points[:,1] = -points[:1]
        #x_points = points[:, 0]
        #y_points = -points[:, 1]
        #z_points = points[:, 2]
    

    # INITIALIZE EMPTY ARRAY - of the dimensions we want

    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    z_max = int((height_range[1] - height_range[0]) / zres)
    voxel_full_size = np.array([ z_max+1, x_max+1, y_max+1])
    
    '''
    if top_count is None:
        top_count = np.zeros([y_max+1, x_max+1, z_max+1],dtype=int)-1
    else:
        assert x_max==(top_count.shape[1]-1) and y_max==(top_count.shape[0]-1), 'shape mismatch of top_count, %d vs. %d and %d vs. %d'%(x_max,top_count.shape[1]-1,y_max,top_count.shape[0]-1)
    '''

    f_filt = np.logical_and(
        (points[:, 0] > fwd_range[0]), (points[:, 0] < fwd_range[1]))
    s_filt = np.logical_and(
        (points[:, 1] > side_range[0]), (points[:, 1] < side_range[1]))
    z_filt = np.logical_and(
        (points[:, 2] > height_range[0]), (points[:, 2] < height_range[1]))
    filter = np.logical_and(np.logical_and(f_filt, s_filt),z_filt)

    #print np.sum(f_filt),np.sum(s_filt),np.sum(z_filt)

    points_filt = points[filter,:] #fwd,side,height
    img_index2 = img_index2[:,filter]
    xyz_points = points_filt[:, 0:3]
    xyz_img = np.zeros_like(xyz_points,dtype=int)
    #points_filt = points_filt.tolist()
    #reflectance = points_filt[:,3]

    #print 'init time: ', time.time()-t0
    #t0 = time.time()

    counter_all = 0
    counter_voxels = 0
    

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after
    xyz_img[:,0] = (((xyz_points[:,1]-side_range[0]) / res).astype(np.int32))  # x axis is -y in LIDAR
    xyz_img[:,1] = (((xyz_points[:,0]-fwd_range[0]) / res).astype(np.int32))   # y axis is -x in LIDAR
    xyz_img[:,2] = (((xyz_points[:,2]-height_range[0]) / zres).astype(np.int32)) 

    #print xyz_img.shape

    unique_xyz,indices_inv = np.unique(xyz_img,axis=0,return_inverse=True,return_counts=False)
    counter_voxels = unique_xyz.shape[0]


    top_sparse = np.zeros([counter_voxels,MAX_NUM_POINTS,NUM_VOXEL_FEATURES])
    #WZN: the first colum is always 0 which indicates the batch number!!! IMPORTANT
    indices_and_count_sparse = np.zeros([counter_voxels,5],dtype=int)#.tolist()
    indices_and_count_sparse[:,1:4] = unique_xyz[:,[2,0,1]]  # voxel shape is 1x10(updpw)x200(side)x240(fwd) for network 
    #indices_and_count_sparse = np.array([[0]*4]*counter_voxels)
    #print indices_and_count_sparse.shape

    filt_indices = []
    for j in range(xyz_img.shape[0]):
        sparse_index = indices_inv[j]
        num_points = indices_and_count_sparse[sparse_index,-1]
        if num_points<MAX_NUM_POINTS:
            top_sparse[sparse_index,num_points,0:4] = points_filt[j]
            indices_and_count_sparse[sparse_index,-1] += 1
            filt_indices.append(j)


    top_sparse[:,:,4:7] = top_sparse[:,:,0:3]-np.expand_dims(np.sum(top_sparse[:,:,0:3],axis=1)/indices_and_count_sparse[:,4:5],1)
    # so for corrdinates, it is [y_img(from z_cam), x_img(from x_cam), z_img(from y_cam)], but for feature it is [z_cam(x_lidar),x_cam(-y_lidar),y_cam(z_lidar)]

    voxel_dict = {'feature_buffer': top_sparse,
                  'coordinate_buffer': indices_and_count_sparse[:,0:4],
                  'number_buffer': indices_and_count_sparse[:,-1]}

    #construct image indexes
    if points_in_cam:
        P = calib_to_P(calib,from_camera=True)
    else:
        assert False, 'Wrong, cannot process LIDAR coordinate points'

    img_index2 = img_index2[:,filt_indices]
    N = img_index2.shape[1]
    img_index = np.vstack((img_index2,np.zeros((1,N)).astype(int)))
    bv_index = xyz_img[filt_indices,:][:,[1,0]]
    M_val = 1.0/(indices_and_count_sparse[indices_inv[filt_indices],-1])

    return voxel_dict,voxel_full_size,img_index,bv_index,M_val