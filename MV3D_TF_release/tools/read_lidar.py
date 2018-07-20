import numpy as np
import os
import matplotlib.pyplot as plt

# the lidar shows from 10-60m the density is approx porpotional to. sqrt(1/(x-5.5))
# so density should be reweight by max(((x-5.5)/(60-54.5))^2,(4.5/54.5)^2)
den_thre = (4.5/54.5)**2
side_range = (-30., 30.)
fwd_range = (0., 60)
height_range = (-2, 0.4) #
res = 0.1
zres = 0.3

#WZN
def density_reweight(dist):
    den_weight = np.square((dist-5.5)/54.5)
    den_weight[den_weight<den_thre] = den_thre
    return den_weight

# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================

def point_in_bv_indexes(points,
                      res=res,
                      zres=zres,
                      side_range=side_range,  # left-most to right-most
                      fwd_range=fwd_range,  # back-most to forward-most
                      height_range=height_range,  # bottom-most to upper-most
                      ):
#WZN: same as point_cloud_2_top but 
#return: indexes as N*2 in the bv image
#        filtered lidar points
#        bv image size
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    reflectance = points[:,3]

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    z_max = int((height_range[1] - height_range[0]) / zres)
    # z_max =
    #top = np.zeros([y_max+1, x_max+1, z_max+1], dtype=np.float32)

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and(
        (x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and(
        (y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)

    z_filt = np.logical_and((z_points >= height_range[0]),
                            (z_points < height_range[1]))
    zfilter = np.logical_and(filter, z_filt)
    indices = np.argwhere(zfilter).flatten()

    xi_points = x_points[indices]
    yi_points = y_points[indices]
    zi_points = z_points[indices]
    ref_i = reflectance[indices]
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-yi_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-xi_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after
    # shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.floor(fwd_range[1] / res))

    return np.vstack((x_img,y_img)).transpose(), np.vstack((xi_points,yi_points,zi_points)), [x_max+1,y_max+1]

def lidar_bv_append(scan,calib,img_size):
    #return the additional data for MV3D_img
    import utils.transform as transform
    P = transform.calib_to_P(calib)
    indices = transform.clip3DwithinImage(scan[:,0:3].transpose(),P,img_size)
    scan = scan[indices,:]
    bv_index,scan_filtered,bv_size = point_in_bv_indexes(scan)
    N = bv_index.shape[0]
    img_points = transform.projectToImage(scan_filtered,P)
    img_index = np.round(img_points).astype(int)
    img_index = np.vstack((img_index,np.zeros((1,N))))
    return {'bv_index':bv_index,'img_index':img_index,'bv_size':bv_size,'img_size':img_size}

def point_cloud_2_top(points,
                      res=res,
                      zres=zres,
                      side_range=side_range,  # left-most to right-most
                      fwd_range=fwd_range,  # back-most to forward-most
                      height_range=height_range,  # bottom-most to upper-most
                      ):
    """ Creates an birds eye view representation of the point cloud data for MV3D.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        zres:        (float)
                    Desired resolution on Z-axis in metres to use.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        numpy array encoding height features , density and intensity.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    reflectance = points[:,3]

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    z_max = int((height_range[1] - height_range[0]) / zres)
    # z_max =
    top = np.zeros([y_max+1, x_max+1, z_max+1], dtype=np.float32)
    den_rew_mat = np.ones([y_max+1, x_max+1, z_max], dtype=np.float32)

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and(
        (x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and(
        (y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)


    # # ASSIGN EACH POINT TO A HEIGHT SLICE
    # # n_slices-1 is used because values above max_height get assigned to an
    # # extra index when we call np.digitize().
    # bins = np.linspace(height_range[0], height_range[1], num=n_slices-1)
    # slice_indices = np.digitize(z_points, bins=bins, right=False)
    # # RESCALE THE REFLECTANCE VALUES - to be between the range 0-255
    # pixel_values = scale_to_255(r_points, min=0.0, max=1.0)
    # FILL PIXEL VALUES IN IMAGE ARRAY
    # -y is used because images start from top left
    # x_max = int((side_range[1] - side_range[0]) / res)
    # y_max = int((fwd_range[1] - fwd_range[0]) / res)
    # im = np.zeros([y_max, x_max, n_slices], dtype=np.uint8)
    # im[-y_img, x_img, slice_indices] = pixel_values

    counter_all = 0
    counter_overwrite =0
    counter_overwrite_zmax = 0
    for i, height in enumerate(np.arange(height_range[0], height_range[1], zres)):

        z_filt = np.logical_and((z_points >= height),
                                (z_points < height + zres))
        zfilter = np.logical_and(filter, z_filt)
        indices = np.argwhere(zfilter).flatten()

        # KEEPERS
        xi_points = x_points[indices]
        yi_points = y_points[indices]
        zi_points = z_points[indices]
        ref_i = reflectance[indices]

        # print(f_filt.shape)

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-yi_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-xi_points / res).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after
        # shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.floor(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = zi_points - height_range[0]
        # pixel_values = zi_points

        #WZN: for debug
        '''
        for j,_ in enumerate(x_img):
            xj = x_img[j]
            yj = y_img[j]
            pj = pixel_values[j]
            counter_all+=1
            if np.abs(top[xj,yj,i]) > 0:
                counter_overwrite+=1
            if np.abs(top[xj,yj,z_max]) > 0:
                counter_overwrite_zmax+=1
            top[xj,yj,i] = 1
            top[xj,yj,z_max] = 1
        ''' 
        #WZN: save density instead of height
        dist_points = np.linalg.norm(np.vstack((xi_points,yi_points)).T,axis=1)
        #print dist_points.shape
        for j,_ in enumerate(x_img):
            top[y_img[j], x_img[j], i] += 1*density_reweight(dist_points[j:j+1])*3 #*3 is empirical reweight
        top[y_img, x_img, z_max] = ref_i

        ''' origin
        # FILL PIXEL VALUES IN IMAGE ARRAY
        top[y_img, x_img, i] = pixel_values

        # max_intensity = np.max(prs[idx])
        top[y_img, x_img, z_max] = ref_i
       '''
    
    #assert counter_overwrite<=counter_overwrite_zmax
    #print counter_all, counter_overwrite, counter_overwrite_zmax, float(counter_overwrite)/counter_all*100.0
    return top


def main():
    root_dir = '/data/RPN/mscnn-master/data/training' ##"/sdb-4T/kitti/object/testing"
    velodyne = os.path.join(root_dir, "velodyne/")
    bird = os.path.join(root_dir, "lidar_bv/")#"lidar_bv/")



    for i in range(0,7481):
        filename = velodyne + str(i).zfill(6) + ".bin"
        print("Processing: ", filename)
        scan = np.fromfile(filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        bird_view = point_cloud_2_top(scan, res=res, zres=zres,
                                       side_range=side_range,  # left-most to right-most
                                       fwd_range=fwd_range,  # back-most to forward-most
                                       height_range=height_range)
        #save
        np.save(bird+str(i).zfill(6)+".npy",bird_view)

    # test
    test = np.load(bird + "000008.npy")

    print(test.shape)
    for i in range(9):
        plt.figure
        test_show = test[:,:,i]
        show_max_x = np.amax(test_show,axis=1)
        print show_max_x
        print show_max_x[np.where(show_max_x>0)[0]]
        #print np.amax(test_show,axis=0)
        plt.imshow(test[:,:,i])
        plt.show()


if __name__ == "__main__":
    main()
