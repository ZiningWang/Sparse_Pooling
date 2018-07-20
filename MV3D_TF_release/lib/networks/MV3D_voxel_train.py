#WZN: Use corner as target instead of bounding boxes

import tensorflow as tf
import numpy as np
from networks.network import Network
from networks.group_pointcloud import FeatureNet
from networks.VoxelRPN import MiddleAndRPN
from utils.config_voxels import cfg as cfg_voxels
from utils.sparse_pool_utils import produce_sparse_pooling_input

n_classes = 2 # background, car
_feat_stride = [2, 2]
_feat_stride_fv = [8,16,32]
_feat_stride_pool = [8,2] # 0 is img, 1 is bv
anchor_scales = [1.0, 1.0]

class fusion_voxel_train(Network):
    def __init__(self, trainable=True,use_bn=False,use_focal=False,use_fusion=False,use_dropout=False):
        self.training = True
        self.use_bn = use_bn
        self.use_focal = use_focal
        self.use_fusion = use_fusion
        self.use_dropout = use_dropout
        if not use_bn:
            print ('not using batch normalization when merging layers')
        self.inputs = []
        self.vox_feature = tf.placeholder(tf.float32, shape=[None,cfg_voxels.VOXEL_POINT_COUNT, 7])
        self.vox_coordinate = tf.placeholder(tf.int64, [None, 4], name='coordinate')
        self.vox_number = tf.placeholder(tf.int64, [None], name='number')
        self.image_data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.im_info_fv = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.gt_boxes_3d = tf.placeholder(tf.float32, shape=[None, 7])
        self.gt_rys = tf.placeholder(tf.float32, shape=[None, 1])
        self.keep_prob = tf.placeholder(tf.float32)

        #WZN: new sparse tranfomation layer
        #Note here img_index_flip is different from what is implemented in the test
        #it has three columns (batch_ind,x_id,y_id) instead of (x_id,y_id)
        self.Mij_tf = tf.placeholder(tf.int64,shape=(None,2))
        self.M_val_tf = tf.placeholder(tf.float32,shape=(None))
        self.M_size_tf= tf.placeholder(tf.int64,shape=(2))
        self.M_tf = tf.SparseTensor(indices=self.Mij_tf,values=self.M_val_tf,
                          dense_shape=self.M_size_tf)
        self.img_index_flip_tf = tf.placeholder(tf.int32,shape=(None,3))

        VFE_input = [self.vox_feature,self.vox_coordinate,self.vox_number]
        self.feature = FeatureNet(VFE_input,training=self.training, batch_size=1)
        self.rpn = MiddleAndRPN(input=self.feature.outputs, training=self.training,output_score=not(use_fusion))


        self.layers = dict({'vox_feature':self.vox_feature,
                            'vox_coordinate':self.vox_coordinate,
                            'vox_number':self.vox_number,
                            'image_data':self.image_data,
                            'im_info':self.im_info,
                            'im_info_fv':self.im_info_fv,
                            'gt_boxes':self.gt_boxes,
                            'gt_boxes_3d': self.gt_boxes_3d,
                            'gt_ry': self.gt_rys,
                            #'Mij_tf':self.Mij_tf,
                            #'M_val_tf':self.M_val_tf,
                            #'M_size_tf':self.M_size_tf,
                            'img_index_flip_tf':self.img_index_flip_tf,
                            'M_tf':self.M_tf,
                            'lidar_features': self.rpn.conv_feature,
                            'feature_output':self.feature.outputs})
        
        self.trainable = trainable
        self.setup()

    ''' WZN not used in stage 1
        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)
    '''

    def set_training(self,training_state):
        self.training = training_state

    def produce_sparse_pooling_input(self,img_index,im_size,bv_index,bv_size,M_val=None,stride=_feat_stride_pool):
        out = produce_sparse_pooling_input({'img_index':img_index,'img_size':im_size,'bv_index':bv_index,'bv_size':bv_size},M_val=M_val,stride=stride)
        return out['Mij_pool'],out['M_val'],out['M_size'],out['img_index_flip_pool']

    def setup(self):
        fout_num = 768
        # Lidar Bird View
        
        # RGB
        (self.feed('image_data')
              .conv(3, 3, 64, 1, 1, name='conv1_1')
              .conv(3, 3, 64, 1, 1, name='conv1_2')
              .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
              .conv(3, 3, 128, 1, 1, name='conv2_1')
              .conv(3, 3, 128, 1, 1, name='conv2_2')
              .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
              .conv(3, 3, 256, 1, 1, name='conv3_1')
              .conv(3, 3, 256, 1, 1, name='conv3_2')
              .conv(3, 3, 256, 1, 1, name='conv3_3')
              .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
              .conv(3, 3, 512, 1, 1, name='conv4_1')
              .conv(3, 3, 512, 1, 1, name='conv4_2')
              .conv(3, 3, 512, 1, 1, name='conv4_3')
              .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
              .conv(3, 3, 512, 1, 1, name='conv5_1')
              .conv(3, 3, 512, 1, 1, name='conv5_2')
              .conv(3, 3, 512, 1, 1, name='conv5_3')
              .max_pool(2, 2, 2, 2, padding='VALID', name='pool5')
              .conv(3, 3, 512, 1, 1, name='conv6_1'))
        (self.feed('conv4_3')
             .conv(3,3,512,1,1,name='loss1_conv1'))


        #======================================================================
        '''
        #WZN:only for test, DELETE!!!!!
        pooled_size = [1,tf.shape(self.layers['lidar_bv_data'])[1],tf.shape(self.layers['lidar_bv_data'])[2],3]
        (self.feed('M_tf','image_data','img_index_flip_tf')
             .sparse_pool(pooled_size,name='image_data_pooled'))
        (self.feed('conv5_3')
             # .deconv(shape=None, c_o=512, stride=2, ksize=3,  name='deconv_2x_1')
             .conv(3,3,512,1,1,name='rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*2*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))
        '''
        # Fusoin ====================================================================
        (self.feed('loss1_conv1')
             .Deconv2D(512,256,3,(1,1),(0,0), training=self.training, name='deconv1_2'))
        (self.feed('conv5_3')
             .Deconv2D(512,256,2,(2,2),(0,0), training=self.training, name='deconv2_2'))
        (self.feed('conv6_1')
             .Deconv2D(512,256,4,(4,4),(0,0), training=self.training, name='deconv3_2'))
        (self.feed('deconv1_2','deconv2_2','deconv3_2')
              .concat(axis=3,name='img_features'))

        
        pooled_size = [1,tf.shape(self.layers['lidar_features'])[1],tf.shape(self.layers['lidar_features'])[2],fout_num]
        (self.feed('M_tf','img_features','img_index_flip_tf')
             .sparse_pool(pooled_size,name='img_features_pooled'))
        #merge features
        (self.feed('lidar_features','img_features_pooled')
            .concat(axis=3,name='conv_lidar_fused')
            .dropout(self.keep_prob, name='drop_fused'))
        
        #========= RPN ============
        #Bird view
        
        #Front View
        (self.feed('loss1_conv1')
             .conv(7,7,2*2,1, 1, padding='VALID', relu = False, name='rpn_fv1_cls_score'))
        (self.feed('conv5_3')
             .conv(7,7,2*2,1, 1, padding='VALID', relu = False, name='rpn_fv2_cls_score'))
        (self.feed('conv6_1')
             .conv(7,7,2*2,1, 1, padding='VALID', relu = False, name='rpn_fv3_cls_score'))

        #(self.feed('conv6_1_2')
        #     .conv(2,2,512,1,1,name='rpn_fv_conv/2x2')
        #     .conv(1,1,2 ,1 , 1, padding='VALID', relu = False, name='rpn_fv4_cls_score'))

        # offset with/without dropout layer
        if self.use_fusion:
          if self.use_dropout:
            (self.feed('drop_fused')
                .conv(1,1,4, 1, 1, padding='VALID', relu = False, name='rpn_cls_score'))
            (self.feed('drop_fused')
                .conv(1,1,14, 1, 1, padding='VALID',relu = False, name='rpn_bbox_pred'))
          else:
            (self.feed('conv_lidar_fused')
                .conv(1,1,4, 1, 1, padding='VALID', relu = False, name='rpn_cls_score'))
            (self.feed('conv_lidar_fused')
                .conv(1,1,14, 1, 1, padding='VALID',relu = False, name='rpn_bbox_pred'))
        else:
          self.layers['rpn_cls_score']=self.rpn.p_map
          self.layers['rpn_bbox_pred']=self.rpn.r_map

        
        use_reward=False
        if self.use_focal==2:
            print ('using reward in anchor target layer')
            use_reward=True
        #print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        
        #print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        self.description = 'using bbox predition + anchor labeling for bv, and anchor labeling only for fv'
        #bv
        (self.feed('rpn_cls_score','gt_boxes_3d', 'gt_ry', 'im_info')
            .anchor_target_layer_voxel(_feat_stride[0], name = 'rpn_data' , use_reward=use_reward))
        #fv
        (self.feed('rpn_fv1_cls_score','gt_boxes', 'im_info_fv')
            .anchor_fv_target_layer(_feat_stride_fv[0], anchor_scales, name = 'rpn_fv1_data',num_class=2))
        (self.feed('rpn_fv2_cls_score','gt_boxes', 'im_info_fv')
            .anchor_fv_target_layer(_feat_stride_fv[1], anchor_scales, name = 'rpn_fv2_data',num_class=2))
        (self.feed('rpn_fv3_cls_score','gt_boxes', 'im_info_fv')
            .anchor_fv_target_layer(_feat_stride_fv[2], anchor_scales, name = 'rpn_fv3_data',num_class=2))
        # Loss of rpn_cls & rpn_boxes
        # anchor_num * xyzhlw
        

        #========= RoI Proposal ============
        # Lidar Bird View, for inference only
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(2*2,name = 'rpn_cls_prob_reshape')) #number_anchors*2

        self.layers['rpn_anchors_3d_bbox'] = self.get_output('rpn_data')[2]

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','rpn_anchors_3d_bbox','im_info')
             .proposal_layer_voxel(_feat_stride[0], 'TRAIN', name = 'rpn_rois'))

        #(self.feed('rpn_rois', 'gt_boxes_bv', 'gt_boxes_3d', 'gt_boxes_corners', 'calib')
        #     .proposal_target_layer_3d(n_classes, name='roi_data_3d'))

        #(self.feed('roi_data_3d')
        #     .proposal_transform(target='img', name='roi_data_img'))
        #(self.feed('roi_data_3d')
        #     .proposal_transform(target='bv', name='roi_data_bv'))

        #front view
        #(self.feed('rpn_fv_cls_score')
        #    .reshape_layer(2,name = 'rpn_fv_cls_score_reshape'))
