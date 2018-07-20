##WZN: 01092018: CAREFUL ABOUT AUMENTATION, it main ruin the saved ground truth!!
#WZN: difference is by reading two additional variables of lidat birdview


import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob
from utils.transform import lidar_3d_to_bv, _lidar_shift_to_bv_shift, lidar_cnr_to_bv_cnr, calib_to_P,projectToImage
import utils.construct_voxel as construct_voxel


def get_minibatch(roidb, num_classes, training=True):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # print("num_images: ", num_images)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                     size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    # im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    im_scales = [1] #WZN: no random scaling of the image
    im = cv2.imread(roidb[0]['image_path'])
    im = im.astype(np.float32, copy=False)
    ''' #without augment_fv
    im -= cfg.PIXEL_MEANS
    pad_w = int(cfg.PAD_IMAGE_TO[0]-im.shape[1])
    pad_h = int(cfg.PAD_IMAGE_TO[1]-im.shape[0])
    assert pad_w>=0 and pad_h>=0, 'wrong image shape'
    im = np.pad(im,[(0,pad_h),(0,pad_w),(0,0)],mode='constant')
    im_blob = im.reshape((1, im.shape[0], im.shape[1], im.shape[2]))
    img_size = np.array([im.shape[1],im.shape[0]])
    blobs = {'image_data': im_blob,
             'img_size': img_size}
    '''
    blobs = {'image_data': im}

    lidar_pc = np.copy(np.load(roidb[0]['lidar_pc_path'])) #WZN: note this is already in camera frame!!!
    #lidar_bv_append = np.load(roidb[0]['lidar_pc_path'][0:-4]+'_append.npy').item()

    #blobs['calib'] = roidb[0]['calib']

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"
    # gt boxes: (x1, y1, x2, y2, cls)
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = np.copy(roidb[0]['boxes'][gt_inds, :]) * im_scales[0]
    gt_boxes[:, 4] = np.copy(roidb[0]['gt_classes'][gt_inds])
    blobs['gt_boxes'] = gt_boxes

    '''
    # gt boxes bv: (x1, y1, x2, y2, cls)
    gt_boxes_bv = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes_bv[:, 0:4] = np.copy(roidb[0]['boxes_bv'][gt_inds, :])
    gt_boxes_bv[:, 4] = np.copy(roidb[0]['gt_classes'][gt_inds])
    blobs['gt_boxes_bv'] = gt_boxes_bv
    '''

    # gt boxes 3d: (x, y, z, l, w, h, cls)
    gt_boxes_3d = np.empty((len(gt_inds), 7), dtype=np.float32)
    gt_boxes_3d[:, 0:6] = np.copy(roidb[0]['boxes_3D_cam'][gt_inds, :])
    gt_boxes_3d[:, 6] = np.copy(roidb[0]['gt_classes'][gt_inds])
    blobs['gt_boxes_3d'] = gt_boxes_3d
    blobs['gt_rys'] = np.copy(roidb[0]['ry'].reshape([-1,1]))
    ''' #WZN: disable corners 
    '''
    
    #WZN
    blobs['im_id'] = roidb[0]['image_path'][-10:-4]
    #print 'before augmentation: ',blobs['gt_boxes_bv'].shape[0]

    #WZN: difficult level
    blobs['diff_level']= roidb[0]['diff_level']

    if cfg.TRAIN.AUGMENT_BV and training:
        #print 'cfg.AUGMENTATION_BV: ',cfg.TRAIN.AUGMENT_BV ,' training: ', training
        #print 'before: ',roidb[0]['ry'][0]
        blobs,lidar_pc,img_index2 = augment_voxel(blobs,scale=0.8,lidar_pc=lidar_pc,calib=np.copy(roidb[0]['calib']))
        #print 'after : ',roidb[0]['ry'][0]
    else:
        #print blobs['gt_rys'][0]
        P = calib_to_P(np.copy(roidb[0]['calib']),from_camera=True)
        img_points = projectToImage(lidar_pc[:,0:3].transpose(),P)
        img_index2 = np.round(img_points).astype(int)

    voxel_blob,voxel_full_size,img_index,bv_index,M_val = construct_voxel.point_cloud_2_top_sparse(lidar_pc,points_in_cam = True,calib=np.copy(roidb[0]['calib']),img_index2=img_index2)
    blobs['voxel_data'] = voxel_blob
    blobs['img_index'] = img_index
    blobs['bv_index'] = bv_index
    blobs['bv_size'] = [voxel_full_size[1], voxel_full_size[2]]
    '''DEBUG
    print 'calib: ', roidb[0]['calib']
    print 'image_id: ',blobs['im_id']
    print 'before fv augmentation, image_index: ', np.amax(blobs['img_index'],axis=1),np.amin(blobs['img_index'],axis=1)
    print 'before fv augmentation, image shape: ', blobs['image_data'].shape
    '''
    #WZN: augment front view data
    if cfg.TRAIN.AUGMENT_FV and training:
        blobs,fv_shift,im_scales_fv = augment_fv(blobs,scale=10)
    else: 
        im_scales_fv = 1.0
    '''DEBUG
    print 'after fv augmentation, image_index: ', np.amax(blobs['img_index'],axis=1),np.amin(blobs['img_index'],axis=1), ' shifts: ', fv_shift
    print 'after fv augmentation, image shape: ', blobs['image_data'].shape
    '''
    pad_w = int(cfg.PAD_IMAGE_TO[0]-blobs['image_data'].shape[1])
    pad_h = int(cfg.PAD_IMAGE_TO[1]-blobs['image_data'].shape[0])
    assert pad_w>=0 and pad_h>=0, 'wrong image shape'
    blobs['image_data'] = np.pad(blobs['image_data'],[(0,pad_h),(0,pad_w),(0,0)],mode='constant')
    blobs['image_data'] -= cfg.PIXEL_MEANS
    img_size = np.array([blobs['image_data'].shape[1],blobs['image_data'].shape[0]])
    blobs['image_data'] = blobs['image_data'].reshape((1, blobs['image_data'].shape[0], blobs['image_data'].shape[1], blobs['image_data'].shape[2]))
    blobs['img_size'] = img_size

    blobs['im_info_fv'] = np.array(
        [[blobs['image_data'].shape[1], blobs['image_data'].shape[2], im_scales_fv]],
        dtype=np.float32)
    blobs['im_info'] = np.array(
        [[voxel_full_size[2], voxel_full_size[1], im_scales[0]]],
        dtype=np.float32)
    blobs['M_val'] = M_val
    #print 'after augmentation: ', blobs['gt_boxes_bv'].shape[0]
    return blobs

def augment_voxel(blobs,scale=0.5,lidar_pc=None,calib=None):
    #just do translation to bv_image, scale is in meters
    sx,sz = np.random.uniform(-scale,scale,2)
    expansion_ratio = np.random.uniform(0.95,1.05,1)
    rotation_angle = np.random.uniform(-np.pi/10,np.pi/10,1)
    #shift gt_boxes_3D, gt_boxes_corner, gt_boxes_bv
    blobs['gt_boxes_3d'][:,0] += sx
    blobs['gt_boxes_3d'][:,2] += sz
    #expand the gt_boxes_3d
    blobs['gt_boxes_3d'][:,0:6] *= expansion_ratio
    #rotation
    rot_mat = np.array([[np.cos(rotation_angle),np.sin(rotation_angle)],[-np.sin(rotation_angle),np.cos(rotation_angle)]]).reshape(2,2)
    #print np.dot(rot_mat,blobs['gt_boxes_3d'][:,[0,2]].transpose())
    blobs['gt_boxes_3d'][:,[0,2]] = np.dot(rot_mat,blobs['gt_boxes_3d'][:,[0,2]].transpose()).transpose()
    #print 'ry before rotation: ',blobs['gt_rys']
    blobs['gt_rys'] += rotation_angle

    ''' no need as we construct bv_index in construt_voxel
    #shift lidar_bv_data
    sx_bv,sy_bv = _lidar_shift_to_bv_shift(sx,sz)
    sx_bv = np.round(sx_bv).astype(int)
    sy_bv = np.round(sy_bv).astype(int)

    #shift bv_index, clip bv_index,img_index
    blobs['bv_index'][:,0]+=sx_bv
    blobs['bv_index'][:,1]+=sy_bv
    clip_indx = np.logical_and(blobs['bv_index'][:,0]>=0,blobs['bv_index'][:,0]<blobs['bv_size'][0])
    clip_indy = np.logical_and(blobs['bv_index'][:,1]>=0,blobs['bv_index'][:,1]<blobs['bv_size'][1])
    clip_indx = np.logical_and(clip_indx,clip_indy)
    blobs['bv_index'] = blobs['bv_index'][clip_indx,:]
    blobs['img_index'] = blobs['img_index'][:,clip_indx]
    '''
    #shift calib
    #Tr = np.reshape(blobs['calib'][3,:],(3,4))
    #Tr[:,3] += np.dot(Tr[:,0:3],-np.array([sx,sy,0]))
    #blobs['calib'][3,:] = np.reshape(Tr,(-1))
    if not(lidar_pc is None):
        #return image indexes of lidar points
        if cfg.TRAIN.AUGMENT_PC:
            drop_rate = np.random.uniform(0.9,1.0,1)
            remain_index = np.random.choice(lidar_pc.shape[0], int(lidar_pc.shape[0]*drop_rate))
            lidar_pc = lidar_pc[remain_index]
        
        P = calib_to_P(calib,from_camera=True)
        img_points = projectToImage(lidar_pc[:,0:3].transpose(),P)
        img_index2 = np.round(img_points).astype(int)

        #shift
        lidar_pc[:,0]+=sx
        lidar_pc[:,2]+=sz
        #expand
        lidar_pc[:,0:3]*=expansion_ratio
        #rotation
        lidar_pc[:,[0,2]] = np.dot(rot_mat,lidar_pc[:,[0,2]].transpose()).transpose()
        
        return blobs,lidar_pc,img_index2
    else:
        return blobs

def augment_fv(blobs,scale=10):
    sx,sy = np.random.uniform(0,scale,2)
    expansion_ratio = np.random.uniform(0.95,1.05,1)
    blobs['image_data'] = cv2.resize(blobs['image_data'],None,fx=expansion_ratio, fy=expansion_ratio)
    rows,cols = blobs['image_data'].shape[0:2]
    M = np.float32([[1,0,sx],[0,1,sy]])
    blobs['image_data'] = cv2.warpAffine(blobs['image_data'],M,(cols,rows))
    #clip to original maximum size
    if blobs['image_data'].shape[0]>cfg.PAD_IMAGE_TO[1]:
        blobs['image_data'] = blobs['image_data'][0:cfg.PAD_IMAGE_TO[1],:,:]
    if blobs['image_data'].shape[1]>cfg.PAD_IMAGE_TO[0]:
        blobs['image_data'] = blobs['image_data'][:,0:cfg.PAD_IMAGE_TO[0],:]
    blobs['gt_boxes'][:,[0,2]] = blobs['gt_boxes'][:,[0,2]]*expansion_ratio + sx
    blobs['gt_boxes'][:,[1,3]] = blobs['gt_boxes'][:,[1,3]]*expansion_ratio + sy
    blobs['img_index'][0,:] = (blobs['img_index'][0,:]*expansion_ratio+sx).astype(int)
    blobs['img_index'][1,:] = (blobs['img_index'][1,:]*expansion_ratio+sy).astype(int)


    return blobs,[sx,sy],expansion_ratio




def augment_bv(blobs,scale=0.5):
    #just do translation to bv_image, scale is in meters
    sx,sy = np.random.uniform(-scale,scale,2)
    #shift gt_boxes_3D, gt_boxes_corner, gt_boxes_bv
    blobs['gt_boxes_3d'][:,0] += sx
    blobs['gt_boxes_3d'][:,1] += sy
    #WZN: disable corners
    blobs['gt_boxes_corners'][:,0:8] += sx
    blobs['gt_boxes_corners'][:,8:16] += sy
    blobs['gt_boxes_bv_corners'][:,0:8] = lidar_cnr_to_bv_cnr(blobs['gt_boxes_corners'][:,0:24])
    blobs['gt_boxes_bv'][:, 0:4] = lidar_3d_to_bv(blobs['gt_boxes_3d'])
    
    #shift lidar_bv_data
    sx_bv,sy_bv = _lidar_shift_to_bv_shift(sx,sy)
    sx_bv = np.round(sx_bv).astype(int)
    sy_bv = np.round(sy_bv).astype(int)
    blobs['lidar_bv_data'][0,:,:,:] = np.roll(blobs['lidar_bv_data'][0,:,:,:],[sy_bv,sx_bv],axis=[0,1])
    
    if sy_bv>=0:
        blobs['lidar_bv_data'][0,0:sy_bv:,:,:] = 0
    else:
        blobs['lidar_bv_data'][0,sy_bv:,:,:] = 0
    if sx_bv>=0:
        blobs['lidar_bv_data'][0,:,0:sx_bv:,:] = 0
    else:
        blobs['lidar_bv_data'][0,:,sx_bv:,:] = 0

    #shift bv_index, clip bv_index,img_index
    blobs['bv_index'][:,0]+=sx_bv
    blobs['bv_index'][:,1]+=sy_bv
    clip_indx = np.logical_and(blobs['bv_index'][:,0]>=0,blobs['bv_index'][:,0]<blobs['lidar_bv_data'].shape[1])
    clip_indy = np.logical_and(blobs['bv_index'][:,1]>=0,blobs['bv_index'][:,1]<blobs['lidar_bv_data'].shape[2])
    clip_indx = np.logical_and(clip_indx,clip_indy)
    blobs['bv_index'] = blobs['bv_index'][clip_indx,:]
    blobs['img_index'] = blobs['img_index'][:,clip_indx]
    #shift calib
    Tr = np.reshape(blobs['calib'][3,:],(3,4))
    Tr[:,3] += np.dot(Tr[:,0:3],-np.array([sx,sy,0]))
    blobs['calib'][3,:] = np.reshape(Tr,(-1))
    return blobs

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image_path'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales

# def _project_im_rois(im_rois, im_scale_factor):
#     """Project image RoIs into the rescaled training image."""
#     rois = im_rois * im_scale_factor
#     return rois

# def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
#     """Visualize a mini-batch for debugging."""
#     import matplotlib.pyplot as plt
#     for i in xrange(rois_blob.shape[0]):
#         rois = rois_blob[i, :]
#         im_ind = rois[0]
#         roi = rois[1:]
#         im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
#         im += cfg.PIXEL_MEANS
#         im = im[:, :, (2, 1, 0)]
#         im = im.astype(np.uint8)
#         cls = labels_blob[i]
#         plt.imshow(im)
#         print 'class: ', cls, ' overlap: ', overlaps[i]
#         plt.gca().add_patch(
#             plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
#                           roi[3] - roi[1], fill=False,
#                           edgecolor='r', linewidth=3)
#             )
#         plt.show()
