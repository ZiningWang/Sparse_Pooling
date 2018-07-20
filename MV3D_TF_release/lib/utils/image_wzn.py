#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image
import numpy as np
import time
from misc_util import warn
import time

def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out

def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)
    
    def change_hue(x):
        x += hue*255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x
    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    #constrain_image(im)
    return im

def rand_scale(s):
    scale = random.uniform(1, s)
    if(random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res

def random_distort_point(points, l_disturbance):
    location = points[:, 0:3]

    points[:, 0:3] = points[:, 0:3] + np.random.normal(loc = 0.0, scale = l_disturbance, size = np.shape(points[:, 0:3]))

    return points

def data_augmentation(img, points, shape, jitter, hue, saturation, exposure, l_disturbance):

    width, height = img.size
    oh = height
    ow = width
    dw =int(ow*jitter)
    dh =int(oh*jitter)

    pleft  = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop   = random.randint(-dh, dh)
    pbot   = random.randint(-dh, dh)

    swidth =  ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth)  / ow
    sy = float(sheight) / oh
    
    flip = random.randint(1,10000)%2
    cropped = img.crop( (pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft)/ow)/sx
    dy = (float(ptop) /oh)/sy

    sized = cropped.resize(shape)

    if flip: 
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
        points[:, 1] = np.negative(points[:, 1]) # For 3D point itself, we need to flip y axis of points location. For label, we need to change along its x axis.

    img = random_distort_image(sized, hue, saturation, exposure)
    points = random_distort_point(points, l_disturbance)
   
    return img, points, flip, dx,dy,sx,sy 

def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    max_boxes = 50
    label = np.zeros((max_boxes,5))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 5))
        cc = 0
        for i in range(bs.shape[0]):
            x1 = bs[i][1] - bs[i][3]/2
            y1 = bs[i][2] - bs[i][4]/2
            x2 = bs[i][1] + bs[i][3]/2
            y2 = bs[i][2] + bs[i][4]/2
            
            x1 = min(0.999, max(0, x1 * sx - dx)) 
            y1 = min(0.999, max(0, y1 * sy - dy)) 
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))
            
            bs[i][1] = (x1 + x2)/2
            bs[i][2] = (y1 + y2)/2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] =  0.999 - bs[i][1] 
            
            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break
    label = np.reshape(label, (-1))
    return label

def get_points_detection(labpath, flip):
    max_boxes = 50
    label = np.zeros((max_boxes,4))
    if os.path.getsize(labpath):
        bs = np.loadtxt(labpath)
        if bs is None:
            return label
        bs = np.reshape(bs, (-1, 4))
        cc = 0
        for i in range(bs.shape[0]):
            if bs[i][1] > 0.999 or bs[i][1] < 0.001:
                continue
            if bs[i][2] > 0.999 or bs[i][2] < 0.001:
                continue
            if bs[i][3] > 0.999 or bs[i][3] < 0.001:
                continue            

            if flip:
                bs[i][1] = 0.999-bs[i][1] # <- 1 is x in camera coordinate, 0 class, 1 x, 2 y, 3 z
            
            label[cc] = bs[i]
            cc += 1
            if cc >= 50:
                break    
    label = np.reshape(label, (-1))
    return label
def calib_to_P(calib):
    #WZN: get the actual overall calibration matrix from Lidar coord to image
    #calib is 4*12 read by imdb
    #P is 3*4 s.t. uvw=P*XYZ1
    C2V = np.vstack((np.reshape(calib[3,:],(3,4)),np.array([0,0,0,1])))
    R0 = np.hstack((np.reshape(calib[2,:],(4,3)),np.array([[0],[0],[0],[1]])))
    P2 = np.reshape(calib[0,:],(3,4))
    P = np.matmul(np.matmul(P2,R0),C2V)
    return P

def projectToImage(pts_3D, P):
    """
    PROJECTTOIMAGE projects 3D points in given coordinate system in the image
    plane using the given projection matrix P.

    Usage: pts_2D = projectToImage(pts_3D, P)
    input: pts_3D: 3xn matrix
          P:      3x4 projection matrix
    output: pts_2D: 2xn matrix

    last edited on: 2012-02-27
    Philip Lenz - lenz@kit.edu
    """
    # project in image
    mat = np.vstack((pts_3D, np.ones((pts_3D.shape[1]))))

    pts_2D = np.dot(P, mat)

    # scale projected points
    pts_2D[0, :] = pts_2D[0, :] / pts_2D[2, :]
    pts_2D[1, :] = pts_2D[1, :] / pts_2D[2, :]
    pts_2D = np.delete(pts_2D, 2, 0)

    return pts_2D

def clip3DwithinImage(pts_3D,pts_intensity, P,image_size):
    """
    WZN: first project 3D points to image than return index 
    that keep only the points visible to the image
    input see projectToImage(), image_size should be [side,height]
    return the indices of pts_3D
    """
    pts_2D = projectToImage(pts_3D,P)
    # print "check1"
    # print np.shape(pts_2D)
    # print np.shape(pts_intensity)
    pts_2D = np.vstack((pts_2D, pts_intensity))

    # print "========="
    # print pts_2D[:, 0:10]
    # print np.shape(pts_2D)
    indices = np.logical_and(pts_2D[0,:]<image_size[0]-1,pts_2D[0,:]>=0)
    indices = np.logical_and(indices,pts_2D[1,:]>=0)
    indices = np.logical_and(indices,pts_2D[1,:]<image_size[1]-1)
    return pts_2D, indices
def load_kitti_calib(velo_calib_path):


    with open(velo_calib_path) as fi:
        lines = fi.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        
    return {'P2' : P2.reshape(3,4),
            'P3' : P3.reshape(3,4),
            'R0' : R0.reshape(3,3),
            'Tr_velo2cam' : Tr_velo_to_cam.reshape(3, 4)}

def calib_gathered(raw_calib):
    calib = np.zeros((4, 12))
    calib[0, :] = raw_calib['P2'].reshape(12)
    calib[1, :] = raw_calib['P3'].reshape(12)
    calib[2, :9] = raw_calib['R0'].reshape(9)
    calib[3, :] = raw_calib['Tr_velo2cam'].reshape(12)

    return calib

def lidar_points_to_2D_by_depth(lidar_points,
                                lidar_range,
                                calib_matrix,
                                img_size):

                                # res=res,
                                # zres=zres,
                                # side_range=side_range,  # left-most to right-most
                                # fwd_range=fwd_range,  # back-most to forward-most
                                # height_range=height_range,  # bottom-most to upper-most
    P = calib_to_P(calib_matrix)
    proj_img, indices = clip3DwithinImage(lidar_points[:,0:3].transpose(), lidar_points[:, 3], P,img_size)
    proj_img = np.vstack((proj_img, lidar_range))
    proj_points_clipped = proj_img[:, indices]

    return proj_points_clipped

def project(width, height, velo_bin_path, velo_calib_path):
    proj_w = 256 
    proj_h = 64
    proj_c = 2 # [x, y, z, intensity, dist, density]  
    dist_range      = [0, 70] 
    lidar_points    = np.fromfile(velo_bin_path, dtype=np.float32)
    lidar_points = lidar_points.reshape((-1, 4))
    front_proj      = np.zeros([proj_h, proj_w, proj_c], dtype=np.float32) 
    points_range    = np.sqrt(np.square(lidar_points[:, 0]) + np.square(lidar_points[:, 1]) + np.square(lidar_points[:, 2]))

    pts_xyzi    = lidar_points[(lidar_points[:, 0] > 0) & (points_range < dist_range[1]) & (points_range > dist_range[0])]
    pts_range   = points_range[(lidar_points[:, 0] > 0) & (points_range < dist_range[1]) & (points_range > dist_range[0])]

    raw_calib       = load_kitti_calib(velo_calib_path)
    calib_matrix    = calib_gathered(raw_calib)

    img_size        = [width, height]
    projected_2D    = lidar_points_to_2D_by_depth(pts_xyzi, pts_range, calib_matrix, img_size)
    projected_2D[0:2, :] = np.floor(projected_2D[0:2, :]).astype(np.int32)

    xi_point    = projected_2D[1]*proj_h/img_size[1]
    yi_point    = projected_2D[0]*proj_w/img_size[0]
    intensity   = projected_2D[2]
    dist        = projected_2D[3]

    for idx in range(len(xi_point)):
        xx = int(xi_point[idx])
        yy = int(yi_point[idx])
        front_proj[xx, yy, 0] = intensity[idx]
        front_proj[xx, yy, 1] = dist[idx]

    front_proj[:, :, 0] = (255.0 / front_proj[:, :, 0].max() * (front_proj[:, :, 0] - front_proj[:, :, 0].min())).astype(np.uint8)
    front_proj[:, :, 1] = (255.0 / front_proj[:, :, 1].max() * (front_proj[:, :, 1] - front_proj[:, :, 1].min())).astype(np.uint8)

    intense_img = Image.fromarray(front_proj[:, :, 0])
    dist_img = Image.fromarray(front_proj[:, :, 1])
    # print np.shape(front_proj)
    return intense_img, dist_img

def load_points(velo_bin_path, velo_calib_path, x_limit, y_limit, z_limit, width, height):
    # dist_range      = [0, 60]
    lidar_points    = np.fromfile(velo_bin_path, dtype=np.float32)
    lidar_points    = lidar_points.reshape((-1, 4))
    lidar_points    = lidar_points[:, 0:3]

    points_x = lidar_points[:, 0]
    points_y = lidar_points[:, 1]
    points_z = lidar_points[:, 2]
    idx = (lidar_points[:, 0] > x_limit[0]) & (lidar_points[:, 0] < x_limit[1]) & (lidar_points[:, 1] < y_limit[0]) & (lidar_points[:, 1] > y_limit[1]) & (lidar_points[:, 2] > z_limit[0]) & (lidar_points[:, 2] < z_limit[1]) 

    raw_calib       = load_kitti_calib(velo_calib_path)
    calib_matrix    = calib_gathered(raw_calib)
    P               = calib_to_P(calib_matrix)

    pts_xyzi = lidar_points[idx]
    pts_2D = projectToImage(pts_xyzi[:,0:3].transpose(), P)
    pts_2D = pts_2D.transpose()

    clipped_idx = (pts_2D[:, 0] < width-1) & (pts_2D[:, 0] >= 0) & (pts_2D[:, 1] < height-1) & (pts_2D[:, 1] >= 0)

    clipped_pts_xyzi = pts_xyzi[clipped_idx]

    # warn("clipped idx: {}".format(len(clipped_pts_xyzi)))

    # sample_idx = np.random.choice(range(len(clipped_pts_xyzi)), n_points)

    return clipped_pts_xyzi, P

    # warn("max x: {} min x: {}, max y: {} min y: {}, max z: {}, min z:{}".format(np.amax(clipped_pts_xyzi[:,0]), np.amin(clipped_pts_xyzi[:,0]), np.amax(clipped_pts_xyzi[:,1]), np.amin(clipped_pts_xyzi[:,1]), np.amax(clipped_pts_xyzi[:,2]), np.amin(clipped_pts_xyzi[:,2])))
    # return clipped_pts_xyzi[sample_idx]

def voxelize(points, max_point_voxel, x_limit, y_limit, nx, ny):
    # divide point into nx by ny by 1 voxel, 1 is nz
    # output shape is (nx * ny) by n_points in each voxel 
    # warn("Voxelize test")

    # x_limit = [1, 71]
    # y_limit = [40, -40]
    # z_limit = [-3, 3]

    t0 = time.time()

    dx = (x_limit[1] - x_limit[0]) / nx # => 7
    dy = (y_limit[1] - y_limit[0]) / ny # => -8
    dz = 6.0
    idx_x = range(nx)
    idx_y = range(ny)
    voxel_count = []
    count = 0
    criteria = 30

    # num_points = 100
    # t0 = time.time()
    # warn("point shape:{}".format(np.shape(points)))
    idx_p = np.random.permutation(range(len(points)))
    points = points[idx_p]
    
    # t1 = time.time()
    # warn("permut: {}".format(t1-t0))
    voxels = np.zeros((nx, ny, max_point_voxel, 6), dtype=float) # local x, local y,  local z, x, y, z 
    npoints_voxel = np.zeros((nx, ny)) # <= indicate number of points  in each voxel

    # count = np.zeros((nx, ny), dtype = np.int)
    # voxels = np.zeros((nx, ny, max_point, 3))
    # for i in range(len(points)):
    #     point = points[i]
    #     x = int((point[0] - x_limit[0]) / dx)
    #     y = int((point[1] - y_limit[0]) / dy)
    #     c = count[x, y]
    #     if c >= max_point:
    #         continue
    #     voxels[x, y, c, :] = point
    #     count[x, y] = count[x, y]+1

    # x and y are in velodyne coordinate.
    for x in idx_x:
        x_front = x_limit[0] + dx * x
        x_back = x_limit[0] + dx * (x+1)
        for y in idx_y:
            y_left = y_limit[0] + dy * y
            y_right = y_limit[0] + dy * (y+1)
            idx = (points[:, 0] >= x_front) & (points[:, 0] < x_back) & (points[:, 1] >= y_right) & (points[:, 1] < y_left)
            voxel_points = points[idx]
            if len(voxel_points) > max_point_voxel:
                voxel_points = voxel_points[0:max_point_voxel, :]

            voxels[x, y, 0:len(voxel_points), 0:3] = voxel_points # original x, y, z
            voxel_points[:, 0] = (voxel_points[:, 0] - x_front)/float(dx)
            voxel_points[:, 1] = (voxel_points[:, 1] - y_right)/float(dy)
            voxel_points[:, 2] = voxel_points[:, 2]/ dz
            # warn("voxel shape: {}".format(np.shape(voxels[x, y, :, :])))
            voxels[x, y, 0:len(voxel_points), 3:] = voxel_points # locally normalized x, y, z

            npoints_voxel[x, y] = len(voxel_points)
            voxel_count.append(len(voxel_points))
            if len(voxel_points) > criteria:
                count += 1
            # warn("voxel {} {}: {}".format(x, y, len(voxel_points)))
    # t1 = time.time()
    # warn("Voxelize: {}".format(t1-t0))

    # warn("Voxelize: {}, over {}: {}/{}, mean: {}, total point: {}".format(t1-t0, criteria, count, nx * ny, np.mean(voxel_count), len(points)))
    return voxels, npoints_voxel

def associated_voxel(width, height, x_limit, y_limit, nx, ny, n2d, P):
    # x_limit = [1, 71]
    # y_limit = [40, -40]
    # z_limit = [-3, 3]
    # n2d = 13

    vortex = np.zeros((nx * ny * 4, 3)) # total voxel nx, ny, 4 vortex for each voxel, 3 for x, y, z
    dx = (x_limit[1] - x_limit[0])/nx # => 7
    dy = (y_limit[1] - y_limit[0])/ny # => -8

    # warn("dx, dy: {} {}".format(dx, dy))

    idx = 0
    for x_idx in range(nx):
        x_s = x_limit[0] + x_idx * dx
        x_e = x_limit[0] + (x_idx + 1) * dx
        for y_idx in range(ny):
            y_s = y_limit[0] + y_idx * dy
            y_e = y_limit[0] + (y_idx + 1) * dy            
            vortex[idx, :] = [x_s, y_s, 0] # <= 3 is temporal z value
            vortex[idx+1, :] = [x_s, y_e, 0] # <= 3 is temporal z value
            vortex[idx+2, :] = [x_e, y_s, 0] # <= 3 is temporal z value
            vortex[idx+3, :] = [x_e, y_e, 0] # <= 3 is temporal z value
            idx = idx + 4

    # warn("voxel shape: {}".format(vortex))

    test = np.zeros((1, 3))
    # test[0, :] = [30,-5,0]
    # proj = projectToImage(test.transpose(), P)
    # warn("proj test: {}".format(proj))

    vortex_proj = projectToImage(vortex[:,0:3].transpose(), P)
    vortex_proj = vortex_proj.transpose()


    # for i in range(len(vortex_proj)):
    #     if i >= 40 and i <= 80 and i%4 ==0:
    #         warn("{} : {}, from {}".format(i, vortex_proj[i], vortex[i, :]))

    idx = 0
    max_asso_voxel = 30
    asso_voxel_mask = np.zeros((n2d, nx*ny), np.int)
    # num_asso_voxel = np.zeros(n2d)
    div_x = width/float(n2d)


    # warn("div_x : {}".format(div_x))
    for k in range(nx*ny):
        x = vortex_proj[k*4:k*4+4, 0]
        x_min = np.amin(x)
        x_max = np.amax(x)

        x_ind_min = int(np.floor(x_min/float(div_x)))
        x_ind_max = int(np.ceil(x_max/float(div_x)))

        # warn("k: {}, x_min: {}, x_max:{}, x_ind_min:{}, x_ind_max: {}".format(k, x_min, x_max, x_ind_min, x_ind_max))

        if x_ind_max <= 0 or x_ind_min >= n2d:
            continue
        x_ind_min = np.maximum(0, x_ind_min)
        x_ind_max = np.minimum(n2d, x_ind_max)

        asso_2d = np.arange(x_ind_min, x_ind_max)
        # warn("asso_2d: {}".format(asso_2d))
        for a2d in asso_2d:
            asso_voxel_mask[a2d, k] = 1

        # warn("idx: {} : associated with {}".format(k, div_x[asso_2d_idx])) 

        # idx = idx + 4
    # used_voxel = np.zeros(nx*ny, np.int)
    # for i in range(len(asso_voxel_mask)):
    #     warn("{}: {}".format(i, asso_voxel_mask[i]))
    #     for j in range(int(num_asso_voxel[i])):
    #         used = asso_voxel[i, j]
    #         used_voxel[used] = 1
    # warn("sum: {}".format(np.sum(asso_voxel_mask)))

    return asso_voxel_mask

    # warn("total voxel:{}".format(np.sum(used_voxel)))
 
        # warn("{}: {}".format(i, asso_voxel[i, 0:int(num_asso_voxel[i])]))







def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure, max_point_voxel, l_disturbance):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
    lab_locpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels_loc').replace('.jpg', '.txt').replace('.png','.txt')
    velo_bin_path = imgpath.replace('JPEGImages', 'Velodyne/binary').replace('.jpg', '.bin').replace('.png','.bin')
    velo_calib_path = imgpath.replace('JPEGImages', 'Velodyne/calib').replace('.jpg', '.txt').replace('.png','.txt')
    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    width, height = img.size

    # t0 = time.time()
    x_limit = [1, 71]
    y_limit = [40, -40]
    z_limit = [-3, 3]



    points, proj_matrix = load_points(velo_bin_path, velo_calib_path, x_limit, y_limit, z_limit, width, height)
    # t1 = time.time()
    # print(t1-t0)
    img, points, flip,dx,dy,sx,sy = data_augmentation(img,
                                                    points, 
                                                    shape, 
                                                    jitter, 
                                                    hue, 
                                                    saturation, 
                                                    exposure,
                                                    l_disturbance
                                                    )
    w, h = img.size
    label = fill_truth_detection(labpath, w, h, flip, dx, dy, 1./sx, 1./sy)
    label_loc = get_points_detection(lab_locpath, flip)
    nx = 10
    ny = 10
    n2d = 13
    voxel, _ = voxelize(points, max_point_voxel, x_limit, y_limit, nx, ny)

    # warn("width {}, height {}".format(width, height))

    asso_voxel = associated_voxel(width, height, x_limit, y_limit, nx, ny, n2d, proj_matrix)

    # used_voxel = np.zeros(nx*ny, np.int)
    # for i in range(len(asso_voxel)):
    #     for j in range(int(num_asso_voxel[i])):
    #         used = asso_voxel[i, j]
    #         used_voxel[used] = 1
    # check_idx = (used_voxel == 1) & (np.reshape(npoints_voxel, -1) > 30)
    # check = range(len(check_idx))
    # check = np.reshape(check, (-1, 1))
    # check = check[check_idx]
    # print check



    return img, label, voxel, asso_voxel, label_loc

def load_data(imgpath, shape, n_points):
    labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')
    lab_locpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels_loc').replace('.jpg', '.txt').replace('.png','.txt')
    velo_bin_path = imgpath.replace('JPEGImages', 'Velodyne/binary').replace('.jpg', '.bin').replace('.png','.bin')
    velo_calib_path = imgpath.replace('JPEGImages', 'Velodyne/calib').replace('.jpg', '.txt').replace('.png','.txt')


    img = Image.open(imgpath).convert('RGB')
    width, height = img.size

    x_limit = [1, 71]
    y_limit = [40, -40]
    z_limit = [-3, 3]

    points = load_points(velo_bin_path, velo_calib_path, n_points, x_limit, y_limit, z_limit, width, height)

    # warn("width {} height {}".format(shape[0], shape[1]))
    sized = img.resize((shape[0], shape[1]))
    voxel, _ = voxelize(points, max_point_voxel, x_limit, y_limit, nx, ny)
    asso_voxel = associated_voxel(width, height, x_limit, y_limit, nx, ny, n2d, proj_matrix)

    # img, points, flip,dx,dy,sx,sy = data_augmentation(img,
    #                                                 points, 
    #                                                 shape, 
    #                                                 jitter, 
    #                                                 hue, 
    #                                                 saturation, 
    #                                                 exposure,
    #                                                 l_disturbance
    #                                                 )
    return img, sized, voxel, asso_voxel
