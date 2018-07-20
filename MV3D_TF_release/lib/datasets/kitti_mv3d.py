# WZN: Note here we unify all LIDAR points to camera frame!!!

__author__ = 'yuxiang' # derived from honda.py by fyang

import datasets
import datasets.kitti_mv3d
import os
import time
import PIL
import datasets.imdb
import numpy as np
from matplotlib import pyplot as plt
import scipy.sparse
from utils.cython_bbox import bbox_overlaps
from utils.boxes_grid import get_boxes_grid
import subprocess
import pickle
from fast_rcnn.config import cfg
import math
from rpn_msr.generate_anchors import generate_anchors_bv
from utils.transform import camera_to_lidar_cnr, lidar_to_corners_single, computeCorners3D, lidar_3d_to_bv, lidar_cnr_to_3d,bv_anchor_to_lidar,lidar_cnr_to_camera_bv,lidar_cnr_to_bv_cnr

class kitti_mv3d(datasets.imdb):
    def __init__(self, image_set, kitti_path=None,object_name='cars'):
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set
        # self._kitti_path = '$Faster-RCNN_TF/data/KITTI'
        self._kitti_path = self._get_default_path() if kitti_path is None \
                            else kitti_path
        # self._data_path = '$Faster-RCNN_TF/data/KITTI/object'
        self._data_path = os.path.join(self._kitti_path, 'object')
        self._set_label_dir()
        self.set_object(object_name)
        '''
        if object_name=='cars':
            #for cars
            self._classes = ('__background__', 'Car', 'Van', 'Truck', 'Tram')#, 'Pedestrian', 'Cyclist')
            self._class_to_ind = dict(zip(self.classes, [0,1,1,1,1]))
        elif object_name=='peds':
            #for peds and cyclists
            #self.num_classes = 3 #0 for background, 1 ped, 2 for cyc, 3 for non-interested region
            self._classes = ('__background__', 'Pedestrian')
            self._class_to_ind = dict(zip(self.classes, [0,1]))
        else:
            assert False, 'invalid training object'
        '''
        self._image_ext = '.png'
        self._lidar_ext = '.npy'
        self._lidar_pc_ext = '.npy'
        self._subset = object_name
        self._image_index = self._load_image_set_index()
        # Default to roidb handler

        self._roidb_handler = self.gt_roidb

        self.config = {'top_k': 100000}

        # statistics for computing recall
        # self._num_boxes_all = np.zeros(self.num_classes, dtype=np.int)
        # self._num_boxes_covered = np.zeros(self.num_classes, dtype=np.int)
        # self._num_boxes_proposal = 0

        assert os.path.exists(self._kitti_path), \
                'KITTI path does not exist: {}'.format(self._kitti_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def set_object(self,object_name):
        if object_name=='cars':
            #for cars
            self._classes = ('__background__', 'Car', 'Van', 'Truck', 'Tram')#, 'Pedestrian', 'Cyclist')
            self.classes_write = ('Car','Pedestrian')
            self._class_to_ind = dict(zip(self.classes, [0,1,1,1,1]))
        elif object_name=='peds':
            #for peds and cyclists
            #self.num_classes = 3 #0 for background, 1 ped, 2 for cyc, -1 for non-interested region, -2 for person_sitting (because thet have bv_boxes)
            self._classes = ('__background__', 'Pedestrian','Person_sitting')   #,'DontCare'
            self.classes_write = ('Car', 'Pedestrian')
            self._class_to_ind = dict(zip(self.classes, [0,1,1])) # I think treating them as 1 makes more positives, that's good  #,-1
        else:
            assert False, 'invalid training object'
        self._subset = object_name
        self._roidb_handler = self.gt_roidb

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self.image_index[i])

    def lidar_pc_path_at(self, i):
        if self._image_set == 'test':
            prefix = 'testing/lidar_pc'   #for voxel
        else:
            prefix = 'training/lidar_pc'  #for voxel
        # lidar_bv_path = '$Faster-RCNN_TF/data/KITTI/object/training/lidar_bv/000000.npy'
        lidar_path = os.path.join(self._data_path, prefix, self.image_index[i] + self._lidar_pc_ext)
        assert os.path.exists(lidar_path), \
                'Path does not exist: {}'.format(lidar_path)
        return lidar_path


    def lidar_path_at(self, i):
        """
        Return the absolute path to lidar i in the lidar sequence.
        """
        return self.lidar_path_from_index(self.image_index[i])

    def calib_at(self, i):
        """
        Return the calib sequence.
        """
        index = self.image_index[i]
        calib_ori =  self._load_kitti_calib(index)
        calib = np.zeros((4, 12))
        calib[0,:] = calib_ori['P2'].reshape(12)
        calib[1,:] = calib_ori['P3'].reshape(12)
        calib[2,:9] = calib_ori['R0'].reshape(9)
        calib[3,:] = calib_ori['Tr_velo2cam'].reshape(12)

        return calib

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        if self._image_set == 'test':
            prefix = 'testing/image_2'
        else:
            prefix = 'training/image_2'
        # image_path = '$Faster-RCNN_TF/data/KITTI/object/training/image_2/000000.png'
        image_path = os.path.join(self._data_path, prefix, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def lidar_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # set the prefix
        if self._image_set == 'test':
            prefix = 'testing/lidar_bv'   #for MV3D
        else:

            prefix = 'training/lidar_bv'  #for MV3D
        # lidar_bv_path = '$Faster-RCNN_TF/data/KITTI/object/training/lidar_bv/000000.npy'
        lidar_bv_path = os.path.join(self._data_path, prefix, index + self._lidar_ext)
        assert os.path.exists(lidar_bv_path), \
                'Path does not exist: {}'.format(lidar_bv_path)
        return lidar_bv_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # image_set_file = '$Faster-RCNN_TF/data/KITTI/ImageSets/train.txt'
        image_set_file = os.path.join(self._kitti_path, 'ImageSets',self._image_set + '.txt')
        self.list_dir = image_set_file
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            #WZN: return lines without '\n'
            image_index = [x.rstrip('\n') for x in f.readlines()]

        print ('image sets length: ', len(image_index))
        return image_index

    def _get_default_path(self):
        """
        Return the default path where KITTI is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'KITTI')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        WZN: first time read Kitti labels, and save a cache
        """

        cache_file = os.path.join(self.cache_path, self.name +'_' +self._subset + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print ('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_kitti_annotation(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print ('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _load_kitti_calib(self, index):
        """
        load projection matrix

        """
        if self._image_set == 'test':
            prefix = 'testing/calib'
        else:
            prefix = 'training/calib'
        calib_dir = os.path.join(self._data_path, prefix, index + '.txt')
        
#         P0 = np.zeros(12, dtype=np.float32)
#         P1 = np.zeros(12, dtype=np.float32)
#         P2 = np.zeros(12, dtype=np.float32)
#         P3 = np.zeros(12, dtype=np.float32)
#         R0 = np.zeros(9, dtype=np.float32)
#         Tr_velo_to_cam = np.zeros(12, dtype=np.float32)
#         Tr_imu_to_velo = np.zeros(12, dtype=np.float32)

#         j = 0
        with open(calib_dir) as fi:
            lines = fi.readlines()
#             assert(len(lines) == 8)
        
#         obj = lines[0].strip().split(' ')[1:]
#         P0 = np.array(obj, dtype=np.float32)
#         obj = lines[1].strip().split(' ')[1:]
#         P1 = np.array(obj, dtype=np.float32)
        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)
#         obj = lines[6].strip().split(' ')[1:]
#         P0 = np.array(obj, dtype=np.float32)
            
        return {'P2' : P2.reshape(3,4),
                'P3' : P3.reshape(3,4),
                'R0' : R0.reshape(3,3),
                'Tr_velo2cam' : Tr_velo_to_cam.reshape(3, 4)}

    def _set_label_dir(self):
        self.gt_dir = os.path.join(self._data_path, 'training/label_2')

    def _load_kitti_annotation(self, index):
        """
        Load image and bounding boxes info from txt file in the KITTI
        format.
        WZN: The non-interested area (Dontcare) is just ignored (treated same as background)
        """
        if self._image_set == 'test':
            return {'ry' : np.array([]),
                    'lwh' : np.array([]),
                    'boxes' : np.array([]), #xy box in image
                    #'boxes_bv' : boxes_bv, #xy box in bird view
                    'boxes_3D_cam' : np.array([]), #[xyz_center, lwh] in 3D, cam frame 
                    #'boxes_3D' : boxes3D_lidar, #[xyz_center, lwh] in 3D, absolute
                    'boxes3D_cam_corners' : np.array([]), #8 corners of box in 3D, cam frame
                    #'boxes_corners' : boxes3D_corners, #8 corners of box in 3D
                    #'boxes_bv_corners' : boxes_bv_corners, #4 corners of box in bird view
                    'gt_classes': np.array([]), #classes
                    'gt_overlaps' : np.array([]), #default 1, changed later
                    'xyz' : np.array([]), 
                    'alphas' :np.array([]), 
                    'diff_level': np.array([]),
                    'flipped' : False}
        else:
            # filename = '$Faster-RCNN_TF/data/KITTI/object/training/label_2/000000.txt'
            filename = os.path.join(self.gt_dir, index + '.txt')
    #         print("Loading: ", filename)

            # calib
            calib = self._load_kitti_calib(index)
            Tr = np.dot(calib['R0'],calib['Tr_velo2cam'])

            # print 'Loading: {}'.format(filename)
            with open(filename, 'r') as f:
                lines = f.readlines()
            num_objs = len(lines)
            translation = np.zeros((num_objs, 3), dtype=np.float32)
            rys = np.zeros((num_objs), dtype=np.float32)
            lwh = np.zeros((num_objs, 3), dtype=np.float32)
            boxes = np.zeros((num_objs, 4), dtype=np.float32)
            boxes_bv = np.zeros((num_objs, 4), dtype=np.float32)
            boxes3D = np.zeros((num_objs, 6), dtype=np.float32)
            boxes3D_lidar = np.zeros((num_objs, 6), dtype=np.float32)
            boxes3D_cam_cnr = np.zeros((num_objs, 24), dtype=np.float32)
            boxes3D_corners = np.zeros((num_objs, 24), dtype=np.float32)
            alphas = np.zeros((num_objs), dtype=np.float32)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            # new difficulty level for in training evaluation
            diff_level = np.zeros((num_objs), dtype=np.int32)

            # print(boxes3D.shape)

            # Load object bounding boxes into a data frame.
            ix = -1
            for line in lines:
                obj = line.strip().split(' ')
                try:
                    #WZN.strip() removes white spaces
                    cls = self._class_to_ind[obj[0].strip()]
                    # print cls
                except:
                    continue
                # ignore objects with undetermined difficult level
                level = self._get_obj_level(obj)
                if level > 3:
                     continue
                ix += 1

                # 0-based coordinates
                alpha = float(obj[3])
                x1 = float(obj[4])
                y1 = float(obj[5])
                x2 = float(obj[6])
                y2 = float(obj[7])
                h = float(obj[8])
                w = float(obj[9])
                l = float(obj[10])
                tx = float(obj[11])
                ty = float(obj[12])
                tz = float(obj[13])
                ry = float(obj[14])

                diff_level[ix]=level
                if obj[0].strip() == 'Person_sitting': 
                    diff_level[ix]=-1
                rys[ix] = ry
                lwh[ix, :] = [l, w, h]
                alphas[ix] = alpha
                translation[ix, :] = [tx, ty, tz]
                boxes[ix, :] = [x1, y1, x2, y2]
                boxes3D[ix, :] = [tx, ty, tz, l, w, h]
                # convert boxes3D cam to 8 corners(cam)
                boxes3D_cam_cnr_single = computeCorners3D(boxes3D[ix, :], ry)
                boxes3D_cam_cnr[ix, :] = boxes3D_cam_cnr_single.reshape(24)
                # convert 8 corners(cam) to 8 corners(lidar)
                boxes3D_corners[ix, :] = camera_to_lidar_cnr(boxes3D_cam_cnr_single, Tr)

                # convert 8 corners(cam) to  lidar boxes3D, note this is not ivertible because we LOSE ry!
                boxes3D_lidar[ix, :] = lidar_cnr_to_3d(boxes3D_corners[ix, :], lwh[ix,:])
                # convert 8 corners(lidar) to lidar bird view
                boxes_bv[ix, :] = lidar_3d_to_bv(boxes3D_lidar[ix, :])
                # boxes3D_corners[ix, :] = lidar_to_corners_single(boxes3D_lidar[ix, :])
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0

            rys.resize(ix+1)
            lwh.resize(ix+1, 3)
            translation.resize(ix+1, 3)
            alphas.resize(ix+1)
            boxes.resize(ix+1, 4)
            boxes_bv.resize(ix+1, 4)
            boxes3D.resize(ix+1, 6)
            boxes3D_lidar.resize(ix+1, 6)
            boxes3D_cam_cnr.resize(ix+1, 24)
            boxes3D_corners.resize(ix+1, 24)

            boxes_bv_corners = lidar_cnr_to_bv_cnr(boxes3D_corners)

            gt_classes.resize(ix+1)
            # print(self.num_classes)
            overlaps.resize(ix+1, self.num_classes)
            diff_level.resize(ix+1)
            # if index == '000142':
            #     print(index)
            #     print(overlaps)
            overlaps = scipy.sparse.csr_matrix(overlaps)
            # if index == '000142':
            #     print(overlaps)

            #if ix>=0:
            #    print index

            return {'ry' : rys,
                    'lwh' : lwh,
                    'boxes' : boxes, #xy box in image
                    #'boxes_bv' : boxes_bv, #xy box in bird view
                    'boxes_3D_cam' : boxes3D, #[xyz_center, lwh] in 3D, cam frame 
                    #'boxes_3D' : boxes3D_lidar, #[xyz_center, lwh] in 3D, absolute
                    'boxes3D_cam_corners' : boxes3D_cam_cnr, #8 corners of box in 3D, cam frame
                    #'boxes_corners' : boxes3D_corners, #8 corners of box in 3D
                    #'boxes_bv_corners' : boxes_bv_corners, #4 corners of box in bird view
                    'gt_classes': gt_classes, #classes
                    'gt_overlaps' : overlaps, #default 1, changed later
                    'xyz' : translation, 
                    'alphas' :alphas, 
                    'diff_level': diff_level,
                    'flipped' : False}

    def _get_obj_level(self, obj):
        height = float(obj[7]) - float(obj[5]) + 1
        trucation = float(obj[1])
        occlusion = float(obj[2])
        if height >= 40 and trucation <= 0.15 and occlusion <= 0:
            return 1
        elif height >= 25 and trucation <= 0.3 and occlusion <= 1:
            return 2
        #WZN: changed from <=2 to <2
        elif height >= 25 and trucation <= 0.5 and occlusion < 2:
            return 3
        else:
            return 4

    def _write_kitti_results_file(self, all_boxes, all_boxes3D):
        # use_salt = self.config['use_salt']
        # comp_id = ''
        # if use_salt:
        #     comp_id += '{}'.format(os.getpid())
        #WZN: only write 2D detection result. 
        path = os.path.join(datasets.ROOT_DIR, 'kitti/results', 'kitti_' + self._subset + '_' + self._image_set + '_' \
                                        + '-' + time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time())), 'data')
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(path, index + '.txt')
            with open(filename, 'wt') as f:
                for cls_ind, cls in enumerate(self.classes):
                    if cls=='__background__' or cls=='DontCare' or cls=='Person_sitting':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    # dets3D = all_boxes3D[cls_ind][im_ind]
                    # alphas = all_alphas[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the KITTI server expects 0-based indices
                    for k in range(dets.shape[0]):
                        # TODO
                        # alpha = dets3D[k, 0] - np.arctan2(dets3D[k, 4], dets3D[k, 6])
                        alpha = 0
                        # WZN: .lower() changes letters to lower case.
                        f.write('{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1 -1 -1 -1 -1\n' \
                                .format(cls.lower(), alpha, \
                                dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))
        return path

    def _write_kitti_results_bv_file(self, all_2d_boxes, all_ry, all_bv_boxes, calibs, all_scores,result_path=None):
        # use_salt = self.config['use_salt']
        # comp_id = ''
        # if use_salt:
        #     comp_id += '{}'.format(os.getpid())
        #WZN: only write 2D detection result.
        if result_path is None:
            result_path = 'kitti/results'
        else:
            result_path = 'kitti/results/'+result_path
        path = os.path.join(datasets.ROOT_DIR, result_path, 'kitti_' + self._subset + '_' + self._image_set + '_' \
                                        + '-' + time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time())), 'data')
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)

        print_debug=0
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(path, index + '.txt')
            with open(filename, 'wt') as f:
                for cls_ind, cls_name in enumerate(self.classes_write):
                    if cls_name == '__background__':
                        continue
                    dets2d = all_2d_boxes[cls_ind][im_ind] # should be [x1,y1,x2,y2]
                    if dets2d is None:
                        continue
                    #print im_ind, len(all_2d_boxes[cls_ind])
                    rys = all_ry[cls_ind][im_ind]
                    calib = calibs[im_ind]
                    scores = all_scores[cls_ind][im_ind].reshape([-1])
                    R0 = calib[2,:9].reshape((3,3))
                    Tr_velo2cam = calib[3,:].reshape((3,4))
                    #print R0, Tr_velo2cam
                    Tr = np.dot(R0,Tr_velo2cam)
                    detslidar = bv_anchor_to_lidar(all_bv_boxes[cls_ind][im_ind]) # should be [x,y,z,l,w,h] in lidar
                    dets_bv_cam = np.zeros((detslidar.shape[0],4))
                    ry_bv = np.zeros(detslidar.shape[0])
                    for iry, ry in enumerate(rys):
                        detscorner = lidar_to_corners_single(detslidar[iry,:],ry) # should be corners in lidar
                        dets_bv_cam[iry,:],ry_bv[iry] = lidar_cnr_to_camera_bv(detscorner, Tr)

                    # the KITTI server expects 0-based indices
                    alpha = 0
                    k=0
                    #if print_debug==0:
                    #    print cls_name.lower(), alpha, dets2d[k, 0], dets2d[k, 1], dets2d[k, 2], dets2d[k, 3],dets_bv_cam[k,3],dets_bv_cam[k,2],dets_bv_cam[k,0],dets_bv_cam[k,1],ry_bv[k], scores[k]
                    #    print scores.shape,ry_bv.shape
                    #    print_debug=1
                    for k in range(dets2d.shape[0]):
                        # WZN: .lower() changes letters to lower case.
                        f.write('{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} -1000 {:.2f} {:.2f} {:.2f} -1000 {:.2f} {:.2f} {:.3f}\n' \
                                .format(cls_name.lower(), alpha, \
                                dets2d[k, 0], dets2d[k, 1], dets2d[k, 2], dets2d[k, 3],\
                                dets_bv_cam[k,3],dets_bv_cam[k,2],dets_bv_cam[k,0],dets_bv_cam[k,1],\
                                ry_bv[k], scores[k]*1000)) # WZN: *1000 is used in MSCNN, does not change result but makes it more readble
        return path


    def _write_kitti_results_bv_cnr_file(self, all_2d_boxes, all_ry, all_3d_cnrs, calibs, all_scores,result_path=None):
        # use_salt = self.config['use_salt']
        # comp_id = ''
        # if use_salt:
        #     comp_id += '{}'.format(os.getpid())
        #WZN: only write 2D detection result.
        if result_path is None:
            result_path = 'kitti/results'
        else:
            result_path = 'kitti/results/'+result_path
        path = os.path.join(datasets.ROOT_DIR, result_path, 'kitti_' + self._subset + '_' + self._image_set + '_' \
                                        + '-' + time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time())), 'data')
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)

        print_debug=0
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(path, index + '.txt')
            with open(filename, 'wt') as f:
                for cls_ind, cls_name in enumerate(self.classes_write):
                    if cls_name == '__background__':
                        continue
                    dets2d = all_2d_boxes[cls_ind][im_ind] # should be [x1,y1,x2,y2]
                    if dets2d is None:
                        continue
                    #print im_ind, len(all_2d_boxes[cls_ind])
                    rys = all_ry[cls_ind][im_ind]
                    calib = calibs[im_ind]
                    scores = all_scores[cls_ind][im_ind].reshape([-1])
                    R0 = calib[2,:9].reshape((3,3))
                    Tr_velo2cam = calib[3,:].reshape((3,4))
                    #print R0, Tr_velo2cam
                    Tr = np.dot(R0,Tr_velo2cam)
                    #detslidar = bv_anchor_to_lidar(all_bv_boxes[cls_ind][im_ind]) # should be [x,y,z,l,w,h] in lidar
                    
                    detscorners = all_3d_cnrs[cls_ind][im_ind]
                    dets_bv_cam = np.zeros((detscorners.shape[0],4))
                    ry_bv = np.zeros(detscorners.shape[0])
                    for iry, ry in enumerate(rys):
                        #detscorner = lidar_to_corners_single(detslidar[iry,:],ry) # should be corners in lidar
                        dets_bv_cam[iry,:],ry_bv[iry] = lidar_cnr_to_camera_bv(detscorners[iry,:], Tr)

                    # the KITTI server expects 0-based indices
                    alpha = 0
                    k=0
                    #if print_debug==0:
                    #    print cls_name.lower(), alpha, dets2d[k, 0], dets2d[k, 1], dets2d[k, 2], dets2d[k, 3],dets_bv_cam[k,3],dets_bv_cam[k,2],dets_bv_cam[k,0],dets_bv_cam[k,1],ry_bv[k], scores[k]
                    #    print scores.shape,ry_bv.shape
                    #    print_debug=1
                    for k in range(dets2d.shape[0]):
                        # WZN: .lower() changes letters to lower case.
                        f.write('{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} -1000 {:.2f} {:.2f} {:.2f} -1000 {:.2f} {:.2f} {:.3f}\n' \
                                .format(cls_name.lower(), alpha, \
                                dets2d[k, 0], dets2d[k, 1], dets2d[k, 2], dets2d[k, 3],\
                                dets_bv_cam[k,3],dets_bv_cam[k,2],dets_bv_cam[k,0],dets_bv_cam[k,1],\
                                ry_bv[k], scores[k]*1000)) # WZN: *1000 is used in MSCNN, does not change result but makes it more readble
        return path

    def _write_kitti_results_voxel_file(self, all_ry, all_3d_bbox, all_scores,result_path=None):
        #WZN: only write 2D detection result. difference is here the corners are already in camera frame 
        if result_path is None:
            result_path = 'kitti/results'
        else:
            result_path = 'kitti/results/'+result_path
        path = os.path.join(datasets.ROOT_DIR, result_path, 'kitti_' + self._subset + '_' + self._image_set + '_' \
                                        + '-' + time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time())), 'data')
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)

        print_debug=0
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(path, index + '.txt')
            with open(filename, 'wt') as f:
                for cls_ind, cls_name in enumerate(self.classes_write):
                    if cls_name == '__background__':
                        continue
                    
                    rys = all_ry[cls_ind][im_ind]
                    if rys is None:
                        continue
                    dets_3d_bbox = all_3d_bbox[cls_ind][im_ind]
                    scores = all_scores[cls_ind][im_ind].reshape([-1])
                    dets_bv_cam = dets_3d_bbox[:,[0,2,3,4]]
                    #dets2d = 
                    ry_bv = rys

                    # the KITTI server expects 0-based indices
                    alpha = 0
                    k=0
                    #if print_debug==0:
                    #    print cls_name.lower(), alpha, dets2d[k, 0], dets2d[k, 1], dets2d[k, 2], dets2d[k, 3],dets_bv_cam[k,3],dets_bv_cam[k,2],dets_bv_cam[k,0],dets_bv_cam[k,1],ry_bv[k], scores[k]
                    #    print scores.shape,ry_bv.shape
                    #    print_debug=1
                    for k in range(ry_bv.shape[0]):
                        # WZN: .lower() changes letters to lower case.
                        f.write('{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} -1000 {:.2f} {:.2f} {:.2f} -1000 {:.2f} {:.2f} {:.3f}\n' \
                                .format(cls_name.lower(), alpha, \
                                0,0,100,100,\
                                dets_bv_cam[k,3],dets_bv_cam[k,2],dets_bv_cam[k,0],dets_bv_cam[k,1],\
                                ry_bv[k], scores[k]*1000)) # WZN: *1000 is used in MSCNN, does not change result but makes it more readble #dets2d[k, 0], dets2d[k, 1], dets2d[k, 2], dets2d[k, 3],\
        return path

    def _write_corners_results_file(self, all_boxes, all_boxes3D):
        # use_salt = self.config['use_salt']
        # comp_id = ''
        # if use_salt:
        #     comp_id += '{}'.format(os.getpid())
        #WZN: looks like this is still not usable
        path = os.path.join(datasets.ROOT_DIR, 'kitti/results_cnr', 'kitti_' + self._subset + '_' + self._image_set + '_' \
                                        + '-' + time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time())), 'data')
        if os.path.exists(path):
            pass
        else:
            os.makedirs(path)
        for im_ind, index in enumerate(self.image_index):
            filename = os.path.join(path, index + '.npy')
            with open(filename, 'wt') as f:
                for cls_ind, cls in enumerate(self.classes_write):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    dets3D = all_boxes3D[cls_ind][im_ind]
                    # alphas = all_alphas[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the KITTI server expects 0-based indices
                    for k in range(dets.shape[0]):
                        obj = np.hstack((dets[k], dets3D[k, 1:]))
                        # print obj.shape
                        np.save(filename, obj)
                        # # TODO
                        # alpha = dets3D[k, 0] - np.arctan2(dets3D[k, 4], dets3D[k, 6])
                        # f.write('{:s} -1 -1 {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.3f}\n' \
                        #         .format(cls.lower(), alpha, \
                        #         dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3], \
                        #         dets3D[k, 2], dets3D[k, 3], dets3D[k, 1], \
                        #         dets3D[k, 4], dets3D[k, 5], dets3D[k, 6], dets3D[k, 0], dets[k, 4]))
        print ('Done')
        # return path

    def _do_eval(self, path, output_dir='output'):
        #WZN: do 2D evaluation
        cmd = os.path.join(datasets.ROOT_DIR, 'kitti/eval/cpp/evaluate_object {}'.format(os.path.dirname(path)))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def _do_eval_bv(self, path, output_dir='output'):
        #WZN: do 2D evaluation
        cmd = os.path.join(datasets.ROOT_DIR, 'kitti/eval/cpp/evaluate_bv {} {} {}'.format(self.gt_dir,os.path.dirname(path),self.list_dir))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, all_boxes3D, output_dir):
        #WZN: call write result and 2D evaluation, no more fancy things
        self._write_kitti_results_file(all_boxes, all_boxes3D)
        # path = self._write_kitti_results_file(all_boxes, all_boxes3D)
        # if self._image_set != 'test':
        #     self._do_eval(path)

    # multiple threshold to get PR-curve
    def _do_validation_bv(self,boxes_bv,gt_blob,scores=None,thres=0.5,ignore=0.05,DEBUG=False):
        diff_level = gt_blob['diff_level']
        #ignored_height = gt_blob['ignored_height'] #same as cpp for eval (40,25,25 in pixels, but here transformed)
        #the processed bv_boxes, first we only do eval here because polygon intersection is not easy
        #diff_level is the difficulty of ground truth in KITTI, should be the same as gt_box. {-1,0,1,2}, -1 can be ignored
        positive_ind = gt_blob['gt_boxes_bv'][:,4]>0
        diff_level = diff_level[positive_ind]
        #print diff_level.T
        gt_bv = gt_blob['gt_boxes_bv'][positive_ind,0:4]
        bbox_bv = boxes_bv[:,0:4]
        #filter low scores
        assert not(scores is None), 'no score to produce PR-curve'
        scores = np.reshape(scores,[-1])
        bbox_bv = bbox_bv[scores>ignore,:]
        scores = scores[scores>ignore]

        #print scores.shape, scores.size , gt_bv.shape
        ##sort so that we can accumulately calculate
        #ind_sort = np.argsort(scores)
        if scores.size>0 and gt_bv.shape[0]>0:
            overlaps_all = bbox_overlaps(
                    np.ascontiguousarray(bbox_bv, dtype=np.float),
                    np.ascontiguousarray(gt_bv, dtype=np.float))
        else:
            overlaps_all = np.zeros([scores.size,gt_bv.shape[0]])

        t_score_range = np.arange(0.04,0.87,0.02)
        nt = t_score_range.shape[0]
        recalls = np.zeros((nt,3))
        precisions = np.zeros((nt,3))
        gt_nums = np.zeros((nt,3))
        pos_nums = np.zeros((nt,3))

        for diff in [1,2,3]:
            idiff = diff-1
            ind_diff = np.logical_and(diff_level>0,diff_level<=diff)
            for it in range(nt):
                t_score = t_score_range[it]
                ind_score = scores>t_score
                scores_above = scores[ind_score]
                overlaps = overlaps_all[ind_score,:]


                if scores_above.shape[0]==0:
                    tp = 0 
                    fp = 0
                    if gt_bv[ind_diff,:].shape[0]>0:
                        fn = np.sum(ind_diff)
                        #return 0.0,0.0,gt_bv.shape[0],0
                        #recall=0.0; precision=0.0; gt_num=gt_bv.shape[0]; pos_num=0
                    else:
                        fn = 0
                        #return 0.0,0.0,0,0
                        #recall=0.0; precision=0.0; gt_num=0; pos_num=0
                elif gt_bv.shape[0]==0:
                    tp = 0
                    fn = 0
                    fp = bbox_bv.shape[0]
                else:
                    # NOTE this is looser than actual eval!!
                    argmax_overlaps = overlaps.argmax(axis=1)
                    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps]
                    ind1 = max_overlaps<thres #if <0.5, definitely false positive
                    #ind2 = a                 #if >2 positive for one gt, it's fp but now ignore that because we have very low NMS thre
                    fp = np.sum(ind1)

                    if gt_bv[ind_diff,:].shape[0]==0:
                        tp = 0
                        fn = 0
                        #return 0.0,0.0,0,fp
                        #recall=0.0; precision=0.0; gt_num=0; pos_num=fp
                    else:   
                        #argmax_overlaps = overlaps.argmax(axis=1)
                        gt_argmax_overlaps = overlaps[:,ind_diff].argmax(axis=0)
                        gt_max_overlaps = overlaps[:,ind_diff][gt_argmax_overlaps,
                                               np.arange(overlaps[:,ind_diff].shape[1])]
                        if DEBUG:
                            #print 'prop_max_overlaps:',overlaps[np.arange(overlaps.shape[0]), argmax_overlaps]
                            print ('gt_max_overlaps:', gt_max_overlaps)
                            print (gt_max_overlaps>=thres)
                            print (np.sum(gt_max_overlaps>=thres))
                        tp = np.sum(gt_max_overlaps>=thres)
                        fn = np.sum(gt_max_overlaps<thres)

                gt_num = tp+fn
                pos_num = tp+fp
                if gt_num==0:
                    recall = 1
                else:
                    recall = float(tp)/gt_num
                if pos_num==0:
                    precision = 1
                else:
                    precision = float(tp)/pos_num

                recalls[it,idiff] = recall
                precisions[it,idiff] = precision
                gt_nums[it,idiff] = gt_num
                pos_nums[it,idiff] = pos_num

        ##the unprocessed 3d_corners project to bv
        #gt_cnr = gt_blob['gt_boxes_corners']
        return recalls,precisions,gt_nums,pos_nums

    def _calc_AP(self,recalls,precisions,plot_file=None):
        legends = ['Easy','Moderate','Hard']
        if len(recalls.shape)==1:
            ind_sort = np.argsort(recalls)
            recalls = recalls[ind_sort]
            precisions = precisions[ind_sort]
            delta_recalls = recalls-np.hstack((0,recalls[0:-1]))
            AP = np.sum(delta_recalls*precisions)
            if not(plot_file is None):
                plt.plot(recall,precision)
                plt.xlabel('recall')
                plt.ylabel('precision')
                plt.savefig(plot_file)
                plt.close()
        else:
            AP = np.zeros(recalls.shape[1])
            for j in range(recalls.shape[1]):
                ind_sort = np.argsort(recalls[:,j])
                recalls_j = recalls[ind_sort,j]
                precisions_j = precisions[ind_sort,j]
                delta_recalls = recalls_j-np.hstack((0,recalls_j[0:-1]))
                AP[j] = np.sum(delta_recalls*precisions_j)
                if not(plot_file is None):
                    plt.plot(np.hstack((0,recalls_j,recalls_j[-1])),np.hstack((precisions_j[0],precisions_j,0)),label=legends[j])
                    #plt.hold(True)
            plt.xlabel('recall')
            plt.xlim((0.0,1.0))
            plt.ylabel('precision')
            plt.ylim((0.0,1.0))
            plt.legend()
            plt.savefig(plot_file)
            plt.close()
        return AP
        

    ''' one threshold
    def _do_validation_bv(self,boxes_bv,gt_blob,scores=None,thres=0.5,ignore=0.2,DEBUG=False):
        #the processed bv_boxes, first we only do eval here because polygon intersection is not easy

        positive_ind = gt_blob['gt_boxes_bv'][:,4]>0
        gt_bv = gt_blob['gt_boxes_bv'][positive_ind,0:4]
        bbox_bv = boxes_bv[:,0:4]
        #filter low scores
        if scores != None:
            bbox_bv = bbox_bv[scores>ignore,:]
        if bbox_bv.shape[0]==0:
            tp = 0 
            fp = 0
            if gt_bv.shape[0]>0:
                return 0.0,0.0,gt_bv.shape[0],0
            else:
                return 0.0,0.0,0,0
        elif gt_bv.shape[0]==0:
            tp = 0
            fp = bbox_bv.shape[0]
            fn = 0
            return 0.0,0.0,0,fp
        else:   
            overlaps = bbox_overlaps(
                np.ascontiguousarray(bbox_bv, dtype=np.float),
               np.ascontiguousarray(gt_bv, dtype=np.float))
            argmax_overlaps = overlaps.argmax(axis=1)
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
            if DEBUG:
                print 'prop_max_overlaps:',overlaps[np.arange(overlaps.shape[0]), argmax_overlaps]
                print 'gt_max_overlaps:', gt_max_overlaps
                print gt_max_overlaps>=thres
                print np.sum(gt_max_overlaps>=thres)
            tp = np.sum(gt_max_overlaps>=thres)
            fn = np.sum(gt_max_overlaps<thres)
            fp = bbox_bv.shape[0]-tp
        gt_num = tp+fn
        pos_num = tp+fp
        recall = float(tp)/gt_num
        precision = float(tp)/pos_num
        #the unprocessed 3d_corners project to bv
        gt_cnr = gt_blob['gt_boxes_corners']
        return recall,precision,gt_num,pos_num
    ''' 
if __name__ == '__main__':
    d = datasets.kitti_mv3d('train')
    res = d.roidb
    from IPython import embed; embed()
