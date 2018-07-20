# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time
import utils.transform as transform

np.set_printoptions(precision=3)
DEBUG = False
vis = False
DoSummary=True

# The focal loss in RetinaNet
def _focal_loss(logits,labels,gamma=2,alpha=0.25):
    # you may want to feed in the positive and negative losses separately because they have different alphas
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    probs = tf.nn.softmax(logits)
    probs_index = tf.stack([tf.reshape(tf.range(tf.shape(probs)[0]),[-1]),labels],1)
    alpha_weights = (2*alpha-1)*tf.cast(labels,tf.float32)+(1-alpha)
    loss_weights = alpha_weights*tf.pow((1-tf.gather_nd(probs,probs_index)),gamma)
    focal_loss_out = tf.multiply(loss_weights,cross_entropy)
    return focal_loss_out


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None, val_imdb=None, val_roidb=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        self.val_imdb = val_imdb
        self.val_roidb = val_roidb
        # For checkpoint
        self.saver = saver
        if self.net.use_focal==1:
            print ('using focal loss to reweight the losses in proposal stage')
        elif self.net.use_focal==2:
            print ('using global focal loss and overlap as reward for classification')


    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print ('Wrote snapshot to: {:s}'.format(filename))

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma
        
        diffs = tf.subtract(bbox_pred, bbox_targets)

        smooth_l1_sign = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(diffs, diffs), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(diffs), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
        outside_mul = smooth_l1_result

        return outside_mul


    def train_model(self, sess, max_iters):
        """Network training loop."""
        print (self.net.description)

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
        if cfg.TRAIN.VAL:
            if self.val_imdb==None or self.val_roidb==None:
                print ('No validation set, using training set for validation')
                data_layer_val = get_data_layer(self.roidb, self.imdb.num_classes)
                data_names = ['train']
                imdbs = [self.imdb]
                print ('number of training sample:', len(self.roidb))
                data_layers = [data_layer_val]
            else:
                data_layer_val = get_data_layer(self.val_roidb, self.imdb.num_classes)
                data_names = ['valid','train']
                imdbs = [self.val_imdb,self.imdb]
                print ('number of training sample:', len(self.roidb),'  number of validation sample:', len(self.val_roidb))
                data_layers = [data_layer_val,data_layer]
        
        
        # RPN
        # classification loss
        #defined in network.network
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_data = self.net.get_output('rpn_data')
        rpn_label = tf.reshape(rpn_data[0],[-1])
        t_anchor = rpn_data[4]
        #front view
        rpn_fv1_cls_score = tf.reshape(self.net.get_output('rpn_fv1_cls_score'),[-1,2])
        rpn_fv1_label = tf.reshape(self.net.get_output('rpn_fv1_data')[0],[-1])
        rpn_fv2_cls_score = tf.reshape(self.net.get_output('rpn_fv2_cls_score'),[-1,2])
        rpn_fv2_label = tf.reshape(self.net.get_output('rpn_fv2_data')[0],[-1])
        rpn_fv3_cls_score = tf.reshape(self.net.get_output('rpn_fv3_cls_score'),[-1,2])
        rpn_fv3_label = tf.reshape(self.net.get_output('rpn_fv3_data')[0],[-1])
        rpn_fv_cls_score = tf.concat((rpn_fv1_cls_score,rpn_fv2_cls_score,rpn_fv3_cls_score),0)
        rpn_fv_label = tf.concat((rpn_fv1_label,rpn_fv2_label,rpn_fv3_label),0)

        rpn_keep = tf.reshape(tf.where(tf.not_equal(rpn_label,-1)),[-1])
        rpn_fv_keep = tf.reshape(tf.where(tf.not_equal(rpn_fv_label,-1)),[-1])
        # only regression positive anchors
        rpn_bbox_keep = tf.reshape(tf.where(tf.greater(rpn_label, 0)),[-1])

        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_keep),[-1,2])
        rpn_fv_cls_score = tf.reshape(tf.gather(rpn_fv_cls_score, rpn_fv_keep),[-1,2])

        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_keep),[-1])
        rpn_fv_label = tf.reshape(tf.gather(rpn_fv_label, rpn_fv_keep),[-1])

        #WZN
        rpn_pos_keep = tf.reshape(tf.where(tf.greater(rpn_label,0)),[-1])
        rpn_pos_label = tf.gather(rpn_label, rpn_pos_keep)
        rpn_cls_score_pos = tf.reshape(tf.gather(rpn_cls_score, rpn_pos_keep),[-1,2])
        rpn_neg_keep = tf.reshape(tf.where(tf.equal(rpn_label,0)),[-1])
        rpn_neg_label = tf.gather(rpn_label, rpn_neg_keep)
        rpn_cls_score_neg = tf.gather(rpn_cls_score, rpn_neg_keep)

        rpn_fv_pos_keep = tf.reshape(tf.where(tf.greater(rpn_fv_label,0)),[-1])
        rpn_fv_pos_label = tf.gather(rpn_fv_label, rpn_fv_pos_keep)
        rpn_fv_cls_score_pos = tf.reshape(tf.gather(rpn_fv_cls_score, rpn_fv_pos_keep),[-1,2])
        rpn_fv_neg_keep = tf.reshape(tf.where(tf.equal(rpn_fv_label,0)),[-1])
        rpn_fv_neg_label = tf.gather(rpn_fv_label, rpn_fv_neg_keep)
        rpn_fv_cls_score_neg = tf.gather(rpn_fv_cls_score, rpn_fv_neg_keep)


        total_pos_num = (tf.cast(tf.shape(rpn_pos_keep)[0],tf.float32))
        rpn_num = (tf.cast(tf.shape(rpn_keep)[0],tf.float32))
        neg_num = (tf.cast(tf.shape(rpn_neg_keep)[0],tf.float32))

        total_pos_num_fv = (tf.cast(tf.shape(rpn_fv_pos_keep)[0],tf.float32))
        rpn_fv_num = (tf.cast(tf.shape(rpn_fv_keep)[0],tf.float32))
        neg_num_fv = (tf.cast(tf.shape(rpn_fv_neg_keep)[0],tf.float32))
        #
        # switch between losses
        focal_coefficient = tf.Variable(0.0, trainable=False)
        if self.net.use_focal==1:
            # separate calculation
            #rpn_cross_entropy_pos = tf.reduce_mean(_focal_loss(logits=rpn_cls_score_pos, labels=rpn_pos_label,alpha=1))
            #rpn_cross_entropy_pos = tf.where(tf.is_nan(rpn_cross_entropy_pos),0.0,rpn_cross_entropy_pos)
            #rpn_cross_entropy_neg = tf.reduce_mean(_focal_loss(logits=rpn_cls_score_neg, labels=rpn_neg_label,alpha=1))
            # overall calculation
            alpha_focal = 0.9
            print ('NOTE:not using focal loss for objects, only for non-objects, alpha in focal: ', alpha_focal)
            #rpn_cross_entropy_pos = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score_pos, labels=rpn_pos_label))
            #rpn_cross_entropy_pos = tf.where(tf.is_nan(rpn_cross_entropy_pos),0.0,rpn_cross_entropy_pos)

            rpn_cross_entropy = _focal_loss(logits=rpn_cls_score, labels=rpn_label,alpha=alpha_focal)
            rpn_cross_entropy_pos = tf.reduce_sum(tf.gather(rpn_cross_entropy,rpn_pos_keep))#/total_pos_num
            rpn_cross_entropy_neg = tf.reduce_sum(tf.gather(rpn_cross_entropy,rpn_neg_keep))#/total_pos_num
            #sum them together, suitable for two cases
            rpn_cross_entropy = (rpn_cross_entropy_pos+rpn_cross_entropy_neg)

            #rpn_fv_cross_entropy = _focal_loss(logits=rpn_fv_cls_score, labels=rpn_fv_label,alpha=alpha_focal)
        else:
            # separate calculation
            w_pos = 1.5
            w_neg = 1.0
            print ('calculating CE-based positive and negatives separately with w_pos=%.3f and w_neg=%.3f'%(w_pos,w_neg))
            rpn_cross_entropy_pos = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score_pos, labels=rpn_pos_label))
            rpn_cross_entropy_pos = tf.where(tf.is_nan(rpn_cross_entropy_pos),0.0,rpn_cross_entropy_pos)
            normal_neg = False
            if normal_neg:
                rpn_cross_entropy_neg = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score_neg, labels=rpn_neg_label))
            else:
                print ('using focal+CE for negative')
                rpn_cross_entropy = _focal_loss(logits=rpn_cls_score, labels=rpn_label,gamma=2,alpha=0.8)
                rpn_cross_entropy_neg_CE = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score_neg, labels=rpn_neg_label))
                rpn_cross_entropy_neg_focal = tf.reduce_sum(tf.gather(rpn_cross_entropy,rpn_neg_keep))
                rpn_cross_entropy_neg = focal_coefficient*rpn_cross_entropy_neg_focal+(1-focal_coefficient)*rpn_cross_entropy_neg_CE

            rpn_cross_entropy = w_pos*rpn_cross_entropy_pos+w_neg*rpn_cross_entropy_neg
            #rpn_cross_entropy = rpn_cross_entropy # try weight same as focal


        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        #  rpn_bbox_targets = tf.transpose(self.net.get_output('rpn_data')[1],[0,2,3,1])
        rpn_bbox_targets = self.net.get_output('rpn_data')[1]
        rpn_rois = self.net.get_output('rpn_rois')
        t_proposal = rpn_rois[3]

        rpn_bbox_pred = tf.reshape(tf.gather(tf.reshape(rpn_bbox_pred, [-1, 7]), rpn_bbox_keep),[-1, 7])
        rpn_bbox_targets = tf.reshape(tf.gather(tf.reshape(rpn_bbox_targets, [-1,7]),rpn_bbox_keep), [-1, 7])

        rpn_box_num = (tf.cast(tf.shape(rpn_bbox_keep)[0],tf.float32))

        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets)

        rpn_loss_box = tf.reduce_mean(rpn_smooth_l1)
        rpn_loss_box = tf.where(tf.is_nan(rpn_loss_box),0.0,rpn_loss_box)
        # final loss
        loss_box_weight=1
        loss = rpn_cross_entropy + rpn_loss_box #+ rpn_fv_cross_entropy 

        #positive accuracy
        if DoSummary:
            max_pos_score_ind = tf.cast(tf.argmax(rpn_cls_score_pos,axis=1),tf.int32)
            acc_positive = tf.equal(max_pos_score_ind,rpn_pos_label)
            acc_positive = tf.reduce_mean(tf.cast(acc_positive,tf.float32))
            acc_positive = tf.where(tf.is_nan(acc_positive),0.0,acc_positive)
            acc_negative = tf.equal(tf.cast(tf.argmax(rpn_cls_score_neg,axis=1),tf.int32),rpn_neg_label)
            acc_negative = tf.reduce_mean(tf.cast(acc_negative,tf.float32))

            max_pos_score_ind_fv = tf.cast(tf.argmax(rpn_fv_cls_score_pos,axis=1),tf.int32)
            acc_positive_fv = tf.equal(max_pos_score_ind_fv,rpn_fv_pos_label)
            acc_positive_fv = tf.reduce_mean(tf.cast(acc_positive_fv,tf.float32))
            acc_positive_fv = tf.where(tf.is_nan(acc_positive_fv),0.0,acc_positive_fv)
            acc_negative_fv = tf.equal(tf.cast(tf.argmax(rpn_fv_cls_score_neg,axis=1),tf.int32),rpn_fv_neg_label)
            acc_negative_fv = tf.reduce_mean(tf.cast(acc_negative_fv,tf.float32))

        if self.net.use_focal>0:
            lr0 = 1e-5
            decay_after = 0.67 
        else:
            lr0 = 3e-4#5e-6
            decay_after = 0.67 #before 01122018 is 0.5 #original 0.00001
        learning_rate = tf.Variable(lr0, trainable=False)
        print ('learning_rate: ',lr0, 'linearly decay after: ', decay_after, 'total iterations')
        print ('using focal loss: ', self.net.use_focal)
        if cfg.TRAIN.RPN_HAS_BATCH:
            print ('batch size for proposal: ',cfg.TRAIN.RPN_BATCHSIZE)
        else:
            print ('using all anchors for proposal')
        print ('bv data augmentation:', cfg.TRAIN.AUGMENT_BV)
        # train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        if self.net.use_bn:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        else:
            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


        # iintialize variables
        sess.run(tf.global_variables_initializer())
        if not DEBUG:
            if self.pretrained_model is not None:
               print (('Loading pretrained model weights from {:s}').format(self.pretrained_model))
               self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()
        decay_iters =  max_iters*decay_after
        lr_cur = lr0
        focal_coefficient0 = 0.0
        focal_coefficient_cur = focal_coefficient0
        focal_iters = max_iters*0#0.1
        ts_anchor = 0
        log_path = os.path.join(self.output_dir, 'kitti_' + imdbs[0]._subset + '_'  \
                    + '-' + time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time())), 'log')
        if os.path.exists(log_path):
            pass
        else:
            os.makedirs(log_path)
        filename = os.path.join(log_path, 'log.txt')

        with open(filename, 'a+') as f_log:
            f_log.write('learning_rate: %f, linearly decay after: %f total iterations \n'%(lr0,decay_after))
            f_log.write('using focal loss: %d \n'%(self.net.use_focal))
            if cfg.TRAIN.RPN_HAS_BATCH:
                f_log.write('batch size for proposal: %d \n'%(cfg.TRAIN.RPN_BATCHSIZE))
            else:
                f_log.write('using all anchors for proposal \n')
            f_log.write('bv data augmentation: %d \n'%(cfg.TRAIN.AUGMENT_BV))
            f_log.write('fv data augmentation: %d \n'%(cfg.TRAIN.AUGMENT_FV))
            f_log.write('droping point cloud points: %d \n'%(cfg.TRAIN.AUGMENT_PC))
            f_log.write('using dropout: %d \n'%(self.net.use_dropout))
            f_log.write('rewrighting bounding box prediction weight: %f \n'%(loss_box_weight))
            f_log.write('using focal+CE for negative (divided by positive numbers) \n')
        for iter in range(max_iters):
            if iter%cfg.TRAIN.DISPLAY==0:
                if iter > decay_iters:
                    ratio = 1.0 * (max_iters - (iter+1))  # epoch_n + 1 because learning rate is set for next epoch
                    ratio = max(0, ratio / (max_iters - decay_iters))
                    lr_cur = lr0 * ratio
                    sess.run(learning_rate.assign(lr_cur))
                if iter > focal_iters:
                    focal_coefficient_cur = focal_coefficient0
                    sess.run(focal_coefficient.assign(focal_coefficient_cur))
                with open(filename, 'a+') as f_log:
                    f_log.write('current learning rate: %.7f, focal_coefficient_cur: %.3f, focal_coefficient0: %.3f \n'%(lr_cur,focal_coefficient_cur,focal_coefficient0))
            # get one batch
            blobs = data_layer.forward()
            #image size is WxH, img_index is WxH
            Mij_pool,M_val,M_size,img_index_flip_pool = self.net.produce_sparse_pooling_input(blobs['img_index'],blobs['img_size'],
                                                        blobs['bv_index'],blobs['bv_size'],M_val=blobs['M_val'])

            # Make one SGD update
            feed_dict={self.net.image_data: blobs['image_data'],
                       self.net.vox_feature: blobs['voxel_data']['feature_buffer'],
                       self.net.vox_coordinate: blobs['voxel_data']['coordinate_buffer'],
                       self.net.vox_number: blobs['voxel_data']['number_buffer'],
                       self.net.im_info: blobs['im_info'],
                       self.net.im_info_fv: blobs['im_info_fv'],
                       self.net.keep_prob: 0.5,
                       self.net.gt_boxes: blobs['gt_boxes'],
                       self.net.gt_boxes_3d: blobs['gt_boxes_3d'],
                       self.net.gt_rys: blobs['gt_rys'],        
                       self.net.Mij_tf: Mij_pool,
                       self.net.M_val_tf: M_val,
                       self.net.M_size_tf: M_size,
                       self.net.img_index_flip_tf: img_index_flip_pool}
            run_options = None
            run_metadata = None

            timer.tic()
            #t0=time.time()
            if DoSummary:
                t_anchor_out,\
                pos_acc,neg_acc,rpn_box_num_out,rpn_loss_cls_pos_value, rpn_loss_cls_neg_value, rpn_loss_box_value, _ = sess.run([t_anchor,
                    acc_positive,acc_negative,rpn_box_num, rpn_cross_entropy_pos,rpn_cross_entropy_neg, rpn_loss_box, train_op],
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata)
                ts_anchor += t_anchor_out 
                if rpn_box_num_out>0:
                    focal_coefficient0 = 0.998*focal_coefficient0+0.002*pos_acc
                '''
                rpn_cls_score_debug, _ = sess.run([rpn_cls_score, train_op],
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata)
                '''
                #train_writer.add_summary(summary,iter)
            else:
                rpn_bbox_pred_out,loss_value,rpn_loss_cls_value, rpn_loss_box_value, _ = sess.run([rpn_bbox_pred, 
                    loss, rpn_cross_entropy, rpn_loss_box, train_op],
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata)
            #print 'no back prop in training:', time.time()-t0
            timer.toc()


            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if DEBUG:
                cfg.TRAIN.DISPLAY = 1
            with open(filename, 'a+') as f_log:
                if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                    f_log.write('iter: %d / %d, total loss: %.4f, loss_cls_pos: %.4f, loss_cls_neg: %.4f, loss_box: %.4f, lr: %f \n'%\
                            (iter+1, max_iters, rpn_loss_cls_neg_value+rpn_loss_cls_pos_value+rpn_loss_box_value,
                                rpn_loss_cls_pos_value, rpn_loss_cls_neg_value, rpn_loss_box_value, lr_cur))
                    if DoSummary: 
                        #print 'positive object scores:', pos_cls_scores, 'ind:', pos_cls_scores_ind
                        f_log.write('object num: %d, obejct acc: %.4f, background acc: %.4f  \n'%\
                        (rpn_box_num_out,pos_acc,neg_acc))
                    print ('speed: {:.3f}s / iter, anchor time: {:.3f}'.format(timer.average_time,ts_anchor/cfg.TRAIN.DISPLAY))
                    ts_anchor = 0

            #WZN: validation
            if (iter+1) % (cfg.TRAIN.VAL_INTERVAL) == 0:
                for ival, data_layer_select in enumerate(data_layers): #_val
                    val_cls_loss = 0
                    val_cls_num = 0
                    val_pos_cls_num = 0
                    val_bbox_loss = 0
                    val_bbox_num = 0
                    acc_pos_val = 0
                    prec_pos_val = 0
                    val_gt_num_bv = 0
                    val_pos_num_bv = 0
                    recall_bv_val = 0
                    precision_bv_val = 0
                    val_pos_pred_num = 0
                    val_neg_cls_num = 0
                    val_cls_neg_loss = 0
                    self.net.set_training(False)
                    t_val = 0
                    ts_anchor_val = 0
                    ts_proposal_val = 0
                    #create variables to write results
                    all_2d_boxes = [None]*3
                    all_ry = [None]*3
                    all_3d_bboxes = [None]*3
                    all_scores = [None]*3
                    nimg = len(imdbs[ival].image_index)
                    for icls in range(3):
                        all_2d_boxes[icls]=[None]*nimg
                        all_ry[icls]=[None]*nimg
                        all_3d_bboxes[icls]=[None]*nimg
                        all_scores[icls]=[None]*nimg
                    if imdbs[ival]._subset=='peds':
                        icls=1 #pedestrian
                    elif imdbs[ival]._subset=='cars':
                        icls=0
                    for im_ind, index in enumerate(imdbs[ival].image_index):
                    #for val_i in xrange(cfg.TRAIN.VAL_ITER):
                        blobs_val = data_layer_select.forward(training=False)
                        assert index == blobs_val['im_id'], [index,blobs_val['im_id']]
                        Mij_pool,M_val,M_size,img_index_flip_pool = self.net.produce_sparse_pooling_input(blobs_val['img_index'],blobs_val['img_size'],
                                                        blobs_val['bv_index'],blobs_val['bv_size'],M_val=blobs_val['M_val'])
                        feed_dict={self.net.image_data: blobs_val['image_data'],
                                self.net.vox_feature: blobs_val['voxel_data']['feature_buffer'],
                                self.net.vox_coordinate: blobs_val['voxel_data']['coordinate_buffer'],
                                self.net.vox_number: blobs_val['voxel_data']['number_buffer'],
                                self.net.im_info: blobs_val['im_info'],
                                self.net.im_info_fv: blobs_val['im_info_fv'],
                                self.net.keep_prob: 1,
                                self.net.gt_boxes: blobs_val['gt_boxes'],
                                self.net.gt_boxes_3d: blobs_val['gt_boxes_3d'],   
                                self.net.gt_rys: blobs_val['gt_rys'],        
                                self.net.Mij_tf: Mij_pool,
                                self.net.M_val_tf: M_val,
                                self.net.M_size_tf: M_size,
                                self.net.img_index_flip_tf: img_index_flip_pool}
                        '''
                        
                        '''
                        # voxelnet!
                        t0 = time.time()
                        t_anchor_out, t_proposal_out,\
                        total_pos_num_out,neg_num_out,rpn_num_out, acc_pos,acc_neg, rpn_box_num_out,rpn_loss_cls_value, rpn_loss_cls_neg_value, rpn_loss_box_value, rpn_rois_value= sess.run([\
                            t_anchor,t_proposal,
                            total_pos_num,neg_num,rpn_num,acc_positive,acc_negative,rpn_box_num, rpn_cross_entropy, rpn_cross_entropy_neg, rpn_loss_box, rpn_rois],
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
                        t_val += (time.time()-t0)
                        ts_anchor_val += t_anchor_out
                        ts_proposal_val += t_proposal_out
                        ''' WZN: fusion
                        total_pos_num_fv_out,neg_num_fv_out,rpn_num_out, acc_pos_fv,acc_neg_fv, rpn_box_num_out,rpn_fv_loss_cls_value,\
                        rpn_loss_cls_value, rpn_loss_cls_neg_value, rpn_loss_box_value, rpn_rois_value= \
                        sess.run([total_pos_num_fv,neg_num_fv,rpn_num,acc_positive_fv,acc_negative_fv,rpn_box_num, rpn_fv_cross_entropy,\
                        rpn_cross_entropy, rpn_cross_entropy_neg, rpn_loss_box, rpn_rois],
                            feed_dict=feed_dict,
                            options=run_options,
                            run_metadata=run_metadata)
                        t_val += (time.time()-t0)
                        '''
                        #stack results
                        all_scores[icls][im_ind]=rpn_rois_value[2]
                        all_3d_bboxes[icls][im_ind]=rpn_rois_value[1][:,1:]
                        all_ry[icls][im_ind]=rpn_rois_value[1][:,-1]
                        '''#DEBUG use
                        if im_ind==0:
                            calib = calibs[im_ind]
                            R0 = calib[2,:9].reshape((3,3))
                            Tr_velo2cam = calib[3,:].reshape((3,4))
                            print rpn_rois_value[0]
                            print R0, Tr_velo2cam
                            print all_bv_boxes[icls][im_ind]
                            Tr = np.dot(R0,Tr_velo2cam)
                            detslidar = transform.bv_anchor_to_lidar(all_bv_boxes[icls][im_ind])
                            detscorner = transform.lidar_to_corners_single(detslidar[0,:],0) # should be corners in lidar
                            dets_bv_cam,ry_bv = transform.lidar_cnr_to_camera_bv(detscorner, Tr)
                            
                            #print blobs_val
                            print 'in image: ', rpn_rois_value[1][:,1:]
                            print detslidar
                            print detscorner.reshape(3,8)
                            print dets_bv_cam
                            print ry_bv
                        '''
                        
                        #recall_bv_val += recall_bv*gt_num_bv
                        #val_gt_num_bv += gt_num_bv
                        #precision_bv_val += precision_bv*pos_num_bv
                        #val_pos_num_bv += pos_num_bv

                        pos_label_num = total_pos_num_out
                        neg_label_num = neg_num_out
                        acc_pos_val += pos_label_num*acc_pos
                        prec_pos_val += pos_label_num*acc_pos
                        val_pos_pred_num += (1-acc_neg)*neg_label_num+pos_label_num*acc_pos
                        val_pos_cls_num += pos_label_num
                        val_neg_cls_num += neg_label_num
                        val_cls_num += rpn_num_out
                        val_cls_loss += rpn_num_out*rpn_loss_cls_value
                        val_cls_neg_loss += neg_label_num*rpn_loss_cls_neg_value
                        val_bbox_num += rpn_box_num_out
                        val_bbox_loss += rpn_box_num_out*rpn_loss_box_value
                        #print neg_label_num,pos_label_num, rpn_loss_cls_neg_value,blobs_val['im_id']
                        
                      
                    val_cls_loss /= val_cls_num
                    val_cls_neg_loss /= val_neg_cls_num
                    val_bbox_loss /= (val_bbox_num+1e-5)
                    acc_pos_val /= (val_pos_cls_num+1e-5)
                    prec_pos_val /= (val_pos_pred_num+1e-5)
                    recall_bv_val /= (val_gt_num_bv+1e-5)
                    precision_bv_val /= (val_pos_num_bv+1e-5)
                    with open(filename, 'a+') as f_log:
                        f_log.write(data_names[ival] + 'iter: %d / %d, cls_loss: %.4f, cls_neg_loss: %.4f, bbox_loss: %.4f, cls_num: %d, pos_cls_num_fv: %d, pos_recall_fv: %.4f, pos_precision_fv: %.4f, bbox_num: %d \n'%\
                                (iter+1, max_iters ,val_cls_loss, val_cls_neg_loss, val_bbox_loss, val_cls_num, val_pos_cls_num, acc_pos_val, prec_pos_val, val_bbox_num))
                    #plot_path = self.output_dir + '/figures/'
                    #if not os.path.exists(plot_path):
                    #    os.makedirs(plot_path)
                    #    print 'making: ', plot_path
                    #AP = imdbs[ival]._calc_AP(recall_bv_val,precision_bv_val,plot_file=os.path.join(plot_path, (data_names[ival]+str(iter)+'.png')))
                    #print 'recall_bv: ', recall_bv_val.T,' precision_bv: ', precision_bv_val.T, 'Average precision: ', AP
                    #print 'tp+fn: ', val_gt_num_bv, 'tp+fp: ', val_pos_num_bv
                    
                    t0 = time.time()
                    #write results to file
                    write_path = imdbs[ival]._write_kitti_results_voxel_file(all_ry, all_3d_bboxes, all_scores,result_path=(os.path.basename(self.output_dir)+'_'+data_names[ival]+'/iter'+str(iter+1)))
                    imdbs[ival]._do_eval_bv(write_path, output_dir='output')
                    t_val_write = (time.time()-t0)
                    if imdbs[ival]._subset=='cars':
                        write_name='vehicle'
                    elif imdbs[ival]._subset=='peds':
                        write_name='pedestrian'
                    else:
                        assert False, 'wrong object name to read'

                    PR_file = os.path.join(write_path,('../plot/'+write_name+'_detection_ground.txt'))
                    try:
                        PRs = np.loadtxt(PR_file)
                        APs = np.sum(PRs[0:-1,1:4]*(PRs[1:,0:1]-PRs[0:-1,0:1]),axis=0)
                        conclusion_path = os.path.join(write_path,'../../../conclusion.txt')
                        with open(conclusion_path,'a+') as conclusion_file:
                            conclusion_file.write('iteration '+str(iter)+': ')
                            conclusion_file.write('\nrecall            :\n')
                            PRs[:,0].tofile(conclusion_file," ",format='%.3f')
                            conclusion_file.write('\nprec_easy, AP: %.2f :\n'%APs[0])
                            PRs[:,1].tofile(conclusion_file," ",format='%.3f')
                            conclusion_file.write('\nprec_mod , AP: %.2f :\n'%APs[1])
                            PRs[:,2].tofile(conclusion_file," ",format='%.3f')
                            conclusion_file.write('\nprec_hard, AP: %.2f :\n'%APs[2])
                            PRs[:,3].tofile(conclusion_file," ",format='%.3f')
                            conclusion_file.write('iter: %d / %d, cls_loss: %.4f, cls_neg_loss: %.4f, bbox_loss: %.4f, cls_num: %d, pos_cls_num_fv: %d, pos_recall_fv: %.4f, pos_precision_fv: %.4f, bbox_num: %d'%\
                                (iter+1, max_iters ,val_cls_loss, val_cls_neg_loss, val_bbox_loss, val_cls_num, val_pos_cls_num, acc_pos_val, prec_pos_val, val_bbox_num))
                        with open(filename, 'a+') as f_log:
                            f_log.write('APs: %.3f, %.3f, %.3f'%(APs[0],APs[1],APs[2]))
                    except:
                        #f_log.write('No object detected')
                        print ('No object detected')

                    with open(filename, 'a+') as f_log:
                        f_log.write('speed: {:.3f}s / iter, anchor: {:.3f}s, proposal: {:.3f}s, write: {:.3f}s / iter \n'.\
                        format(t_val/im_ind,ts_anchor_val/im_ind, ts_proposal_val/im_ind, t_val_write/im_ind))


                self.net.set_training(True)
                

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                if vis:
                    rpn_data, rpn_rois, rcnn_roi = sess.run([
                                             self.net.get_output('rpn_data'),
                                             self.net.get_output('rpn_rois'),
                                             self.net.get_output('roi_data_3d')],
                                             feed_dict=feed_dict)

                    vis_detections(blobs['lidar_bv_data'], blobs['image_data'], blobs['calib'], rpn_data, rpn_rois, rcnn_roi,  blobs['gt_boxes_3d'])

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

    def evaluate_model(self, sess, testset=False):
        """Network training loop."""
        print (self.net.description)

        if self.val_imdb==None or self.val_roidb==None:
            print ('No validation set, using training set for validation')
            data_layer_val = get_data_layer(self.roidb, self.imdb.num_classes)
            data_names = ['train']
            imdbs = [self.imdb]
            print ('number of training sample:', len(self.roidb))
            data_layers = [data_layer_val]
        else:
            data_layer_val = get_data_layer(self.val_roidb, self.imdb.num_classes)
            data_names = ['valid']
            imdbs = [self.val_imdb]
            print ('number of training sample:', len(self.roidb),'  number of validation sample:', len(self.val_roidb))
            data_layers = [data_layer_val]
        # RPN
        # classification loss
        #defined in network.network
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_data = self.net.get_output('rpn_data')
        rpn_label = tf.reshape(rpn_data[0],[-1])
        t_anchor = rpn_data[4]
        #front view
        rpn_fv1_cls_score = tf.reshape(self.net.get_output('rpn_fv1_cls_score'),[-1,2])
        rpn_fv1_label = tf.reshape(self.net.get_output('rpn_fv1_data')[0],[-1])
        rpn_fv2_cls_score = tf.reshape(self.net.get_output('rpn_fv2_cls_score'),[-1,2])
        rpn_fv2_label = tf.reshape(self.net.get_output('rpn_fv2_data')[0],[-1])
        rpn_fv3_cls_score = tf.reshape(self.net.get_output('rpn_fv3_cls_score'),[-1,2])
        rpn_fv3_label = tf.reshape(self.net.get_output('rpn_fv3_data')[0],[-1])
        rpn_fv_cls_score = tf.concat((rpn_fv1_cls_score,rpn_fv2_cls_score,rpn_fv3_cls_score),0)
        rpn_fv_label = tf.concat((rpn_fv1_label,rpn_fv2_label,rpn_fv3_label),0)

        rpn_keep = tf.reshape(tf.where(tf.not_equal(rpn_label,-1)),[-1])
        rpn_fv_keep = tf.reshape(tf.where(tf.not_equal(rpn_fv_label,-1)),[-1])
        # only regression positive anchors
        rpn_bbox_keep = tf.reshape(tf.where(tf.greater(rpn_label, 0)),[-1])

        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_keep),[-1,2])
        rpn_fv_cls_score = tf.reshape(tf.gather(rpn_fv_cls_score, rpn_fv_keep),[-1,2])

        rpn_label = tf.reshape(tf.gather(rpn_label, rpn_keep),[-1])
        rpn_fv_label = tf.reshape(tf.gather(rpn_fv_label, rpn_fv_keep),[-1])

        #WZN
        rpn_pos_keep = tf.reshape(tf.where(tf.greater(rpn_label,0)),[-1])
        rpn_pos_label = tf.gather(rpn_label, rpn_pos_keep)
        rpn_cls_score_pos = tf.reshape(tf.gather(rpn_cls_score, rpn_pos_keep),[-1,2])
        rpn_neg_keep = tf.reshape(tf.where(tf.equal(rpn_label,0)),[-1])
        rpn_neg_label = tf.gather(rpn_label, rpn_neg_keep)
        rpn_cls_score_neg = tf.gather(rpn_cls_score, rpn_neg_keep)

        rpn_fv_pos_keep = tf.reshape(tf.where(tf.greater(rpn_fv_label,0)),[-1])
        rpn_fv_pos_label = tf.gather(rpn_fv_label, rpn_fv_pos_keep)
        rpn_fv_cls_score_pos = tf.reshape(tf.gather(rpn_fv_cls_score, rpn_fv_pos_keep),[-1,2])
        rpn_fv_neg_keep = tf.reshape(tf.where(tf.equal(rpn_fv_label,0)),[-1])
        rpn_fv_neg_label = tf.gather(rpn_fv_label, rpn_fv_neg_keep)
        rpn_fv_cls_score_neg = tf.gather(rpn_fv_cls_score, rpn_fv_neg_keep)


        total_pos_num = (tf.cast(tf.shape(rpn_pos_keep)[0],tf.float32))
        rpn_num = (tf.cast(tf.shape(rpn_keep)[0],tf.float32))
        neg_num = (tf.cast(tf.shape(rpn_neg_keep)[0],tf.float32))

        total_pos_num_fv = (tf.cast(tf.shape(rpn_fv_pos_keep)[0],tf.float32))
        rpn_fv_num = (tf.cast(tf.shape(rpn_fv_keep)[0],tf.float32))
        neg_num_fv = (tf.cast(tf.shape(rpn_fv_neg_keep)[0],tf.float32))
        #
        # switch between losses
        focal_coefficient = tf.Variable(0.0, trainable=False)
        if self.net.use_focal==1:
            # separate calculation
            #rpn_cross_entropy_pos = tf.reduce_mean(_focal_loss(logits=rpn_cls_score_pos, labels=rpn_pos_label,alpha=1))
            #rpn_cross_entropy_pos = tf.where(tf.is_nan(rpn_cross_entropy_pos),0.0,rpn_cross_entropy_pos)
            #rpn_cross_entropy_neg = tf.reduce_mean(_focal_loss(logits=rpn_cls_score_neg, labels=rpn_neg_label,alpha=1))
            # overall calculation
            alpha_focal = 0.9
            print ('NOTE:not using focal loss for objects, only for non-objects, alpha in focal: ', alpha_focal)
            #rpn_cross_entropy_pos = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score_pos, labels=rpn_pos_label))
            #rpn_cross_entropy_pos = tf.where(tf.is_nan(rpn_cross_entropy_pos),0.0,rpn_cross_entropy_pos)

            rpn_cross_entropy = _focal_loss(logits=rpn_cls_score, labels=rpn_label,alpha=alpha_focal)
            rpn_cross_entropy_pos = tf.reduce_sum(tf.gather(rpn_cross_entropy,rpn_pos_keep))#/total_pos_num
            rpn_cross_entropy_neg = tf.reduce_sum(tf.gather(rpn_cross_entropy,rpn_neg_keep))#/total_pos_num
            #sum them together, suitable for two cases
            rpn_cross_entropy = (rpn_cross_entropy_pos+rpn_cross_entropy_neg)

            #rpn_fv_cross_entropy = _focal_loss(logits=rpn_fv_cls_score, labels=rpn_fv_label,alpha=alpha_focal)
        else:
            # separate calculation
            w_pos = 1.5
            w_neg = 1.0
            print ('calculating CE-based positive and negatives separately with w_pos=%.3f and w_neg=%.3f'%(w_pos,w_neg))
            rpn_cross_entropy_pos = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score_pos, labels=rpn_pos_label))
            rpn_cross_entropy_pos = tf.where(tf.is_nan(rpn_cross_entropy_pos),0.0,rpn_cross_entropy_pos)
            normal_neg = False
            if normal_neg:
                rpn_cross_entropy_neg = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score_neg, labels=rpn_neg_label))
            else:
                print ('using focal+CE for negative')
                rpn_cross_entropy = _focal_loss(logits=rpn_cls_score, labels=rpn_label,gamma=2,alpha=0.8)
                rpn_cross_entropy_neg_CE = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score_neg, labels=rpn_neg_label))
                rpn_cross_entropy_neg_focal = tf.reduce_sum(tf.gather(rpn_cross_entropy,rpn_neg_keep))
                rpn_cross_entropy_neg = focal_coefficient*rpn_cross_entropy_neg_focal+(1-focal_coefficient)*rpn_cross_entropy_neg_CE

            rpn_cross_entropy = w_pos*rpn_cross_entropy_pos+w_neg*rpn_cross_entropy_neg
            #rpn_cross_entropy = rpn_cross_entropy # try weight same as focal
        

        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        #  rpn_bbox_targets = tf.transpose(self.net.get_output('rpn_data')[1],[0,2,3,1])
        rpn_bbox_targets = self.net.get_output('rpn_data')[1]
        rpn_rois = self.net.get_output('rpn_rois')
        t_proposal = rpn_rois[3]

        rpn_bbox_pred = tf.reshape(tf.gather(tf.reshape(rpn_bbox_pred, [-1, 7]), rpn_bbox_keep),[-1, 7])
        rpn_bbox_targets = tf.reshape(tf.gather(tf.reshape(rpn_bbox_targets, [-1,7]),rpn_bbox_keep), [-1, 7])

        rpn_box_num = (tf.cast(tf.shape(rpn_bbox_keep)[0],tf.float32))

        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets)

        rpn_loss_box = tf.reduce_mean(rpn_smooth_l1)
        rpn_loss_box = tf.where(tf.is_nan(rpn_loss_box),0.0,rpn_loss_box)
        # final loss
        loss_box_weight=1
        loss = rpn_cross_entropy + rpn_loss_box #+ rpn_fv_cross_entropy 

        #positive accuracy
        if DoSummary:
            max_pos_score_ind = tf.cast(tf.argmax(rpn_cls_score_pos,axis=1),tf.int32)
            acc_positive = tf.equal(max_pos_score_ind,rpn_pos_label)
            acc_positive = tf.reduce_mean(tf.cast(acc_positive,tf.float32))
            acc_positive = tf.where(tf.is_nan(acc_positive),0.0,acc_positive)
            acc_negative = tf.equal(tf.cast(tf.argmax(rpn_cls_score_neg,axis=1),tf.int32),rpn_neg_label)
            acc_negative = tf.reduce_mean(tf.cast(acc_negative,tf.float32))

            max_pos_score_ind_fv = tf.cast(tf.argmax(rpn_fv_cls_score_pos,axis=1),tf.int32)
            acc_positive_fv = tf.equal(max_pos_score_ind_fv,rpn_fv_pos_label)
            acc_positive_fv = tf.reduce_mean(tf.cast(acc_positive_fv,tf.float32))
            acc_positive_fv = tf.where(tf.is_nan(acc_positive_fv),0.0,acc_positive_fv)
            acc_negative_fv = tf.equal(tf.cast(tf.argmax(rpn_fv_cls_score_neg,axis=1),tf.int32),rpn_fv_neg_label)
            acc_negative_fv = tf.reduce_mean(tf.cast(acc_negative_fv,tf.float32))


        log_path = os.path.join(self.output_dir, 'kitti_' + imdbs[0]._subset + '_'  \
                    + '-' + time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time())), 'log')
        if os.path.exists(log_path):
            pass
        else:
            os.makedirs(log_path)
        filename = os.path.join(log_path, 'log.txt')

        # initialize variables
        sess.run(tf.global_variables_initializer())
        if not DEBUG:
            if self.pretrained_model is not None:
                print ('Loading pretrained model '
                      'weights from {:s}').format(self.pretrained_model)
                with open(filename, 'a+') as f_log:
                    f_log.write(('Loading pretrained model '
                        'weights from {:s}').format(self.pretrained_model))
                self.net.load(self.pretrained_model, sess, self.saver, True)
        timer = Timer()
        focal_coefficient0 = 0.0
        focal_coefficient_cur = focal_coefficient0
        ts_anchor = 0
        

        with open(filename, 'a+') as f_log:
            f_log.write('using focal loss: %d \n'%(self.net.use_focal))
            if cfg.TRAIN.RPN_HAS_BATCH:
                f_log.write('batch size for proposal: %d'%(cfg.TRAIN.RPN_BATCHSIZE))
            else:
                f_log.write('using all anchors for proposal')
            f_log.write('bv data augmentation: %d \n'%(cfg.TRAIN.AUGMENT_BV))
            f_log.write('using focal+CE for negative (divided by positive numbers) \n')
        for ival, data_layer_select in enumerate(data_layers): #_val
            val_cls_loss = 0
            val_cls_num = 0
            val_pos_cls_num = 0
            val_bbox_loss = 0
            val_bbox_num = 0
            acc_pos_val = 0
            prec_pos_val = 0
            val_gt_num_bv = 0
            val_pos_num_bv = 0
            recall_bv_val = 0
            precision_bv_val = 0
            val_pos_pred_num = 0
            val_neg_cls_num = 0
            val_cls_neg_loss = 0
            self.net.set_training(False)
            t_val = 0
            ts_anchor_val = 0
            ts_proposal_val = 0
            #create variables to write results
            all_2d_boxes = [None]*3
            all_ry = [None]*3
            all_3d_bboxes = [None]*3
            all_scores = [None]*3
            all_detection_count = 0
            ignored_detection_count = 0
            nimg = len(imdbs[ival].image_index)
            for icls in range(3):
                all_2d_boxes[icls]=[None]*nimg
                all_ry[icls]=[None]*nimg
                all_3d_bboxes[icls]=[None]*nimg
                all_scores[icls]=[None]*nimg
            if imdbs[ival]._subset=='peds':
                icls=1 #pedestrian
            elif imdbs[ival]._subset=='cars':
                icls=0
            for im_ind, index in enumerate(imdbs[ival].image_index):
            #for val_i in xrange(cfg.TRAIN.VAL_ITER):
                if im_ind%cfg.TRAIN.DISPLAY==0:
                    print (im_ind,'/',len(imdbs[ival].image_index))
                blobs_val = data_layer_select.forward(training=False)
                assert index == blobs_val['im_id'], [index,blobs_val['im_id']]
                
                Mij_pool,M_val,M_size,img_index_flip_pool = self.net.produce_sparse_pooling_input(blobs_val['img_index'],blobs_val['img_size'],
                                                blobs_val['bv_index'],blobs_val['bv_size'],M_val=blobs_val['M_val'])
                feed_dict={self.net.image_data: blobs_val['image_data'],
                        self.net.vox_feature: blobs_val['voxel_data']['feature_buffer'],
                        self.net.vox_coordinate: blobs_val['voxel_data']['coordinate_buffer'],
                        self.net.vox_number: blobs_val['voxel_data']['number_buffer'],
                        self.net.im_info: blobs_val['im_info'],
                        self.net.im_info_fv: blobs_val['im_info_fv'],
                        self.net.keep_prob: 1,
                        self.net.gt_boxes: blobs_val['gt_boxes'],
                        self.net.gt_boxes_3d: blobs_val['gt_boxes_3d'],   
                        self.net.gt_rys: blobs_val['gt_rys'],        
                        self.net.Mij_tf: Mij_pool,
                        self.net.M_val_tf: M_val,
                        self.net.M_size_tf: M_size,
                        self.net.img_index_flip_tf: img_index_flip_pool}
                '''
                
                '''
                # voxelnet!
                t0 = time.time()
                t_anchor_out, t_proposal_out,\
                total_pos_num_out,neg_num_out,rpn_num_out, acc_pos,acc_neg, rpn_box_num_out,rpn_loss_cls_value, rpn_loss_cls_neg_value, rpn_loss_box_value, rpn_rois_value= sess.run([\
                    t_anchor,t_proposal,
                    total_pos_num,neg_num,rpn_num,acc_positive,acc_negative,rpn_box_num, rpn_cross_entropy, rpn_cross_entropy_neg, rpn_loss_box, rpn_rois],
                    feed_dict=feed_dict,
                    options=None,
                    run_metadata=None)
                t_val += (time.time()-t0)
                ts_anchor_val += t_anchor_out
                ts_proposal_val += t_proposal_out
                ''' WZN: fusion
                total_pos_num_fv_out,neg_num_fv_out,rpn_num_out, acc_pos_fv,acc_neg_fv, rpn_box_num_out,rpn_fv_loss_cls_value,\
                rpn_loss_cls_value, rpn_loss_cls_neg_value, rpn_loss_box_value, rpn_rois_value= \
                sess.run([total_pos_num_fv,neg_num_fv,rpn_num,acc_positive_fv,acc_negative_fv,rpn_box_num, rpn_fv_cross_entropy,\
                rpn_cross_entropy, rpn_cross_entropy_neg, rpn_loss_box, rpn_rois],
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata)
                t_val += (time.time()-t0)
                '''
                #stack results
                all_scores[icls][im_ind]=rpn_rois_value[2]
                all_3d_bboxes[icls][im_ind]=rpn_rois_value[1][:,1:]
                all_ry[icls][im_ind]=rpn_rois_value[1][:,-1]
                all_detection_count += rpn_rois_value[1][:,1:].shape[0]
                if rpn_rois_value[1][:,1:].shape[0]>0:
                    pixel_heights = all_3d_bboxes[icls][im_ind][:,5]*722/all_3d_bboxes[icls][im_ind][:,2]
                    print ('pixel  height: ', pixel_heights)
                    print ('true distance: ', all_3d_bboxes[icls][im_ind][:,2])
                    ignored_detection_count += np.sum((pixel_heights<25))
                    remained_index = pixel_heights>25
                    all_scores[icls][im_ind] = all_scores[icls][im_ind][remained_index]
                    all_3d_bboxes[icls][im_ind] = all_3d_bboxes[icls][im_ind][remained_index,:]
                    all_ry[icls][im_ind] = all_ry[icls][im_ind][remained_index]                                  
                    

                pos_label_num = total_pos_num_out
                neg_label_num = neg_num_out
                acc_pos_val += pos_label_num*acc_pos
                prec_pos_val += pos_label_num*acc_pos
                val_pos_pred_num += (1-acc_neg)*neg_label_num+pos_label_num*acc_pos
                val_pos_cls_num += pos_label_num
                val_neg_cls_num += neg_label_num
                val_cls_num += rpn_num_out
                val_cls_loss += rpn_num_out*rpn_loss_cls_value
                val_cls_neg_loss += neg_label_num*rpn_loss_cls_neg_value
                val_bbox_num += rpn_box_num_out
                val_bbox_loss += rpn_box_num_out*rpn_loss_box_value
                #print neg_label_num,pos_label_num, rpn_loss_cls_neg_value,blobs_val['im_id']
                
              
            val_cls_loss /= val_cls_num
            val_cls_neg_loss /= val_neg_cls_num
            val_bbox_loss /= (val_bbox_num+1e-5)
            acc_pos_val /= (val_pos_cls_num+1e-5)
            prec_pos_val /= (val_pos_pred_num+1e-5)
            recall_bv_val /= (val_gt_num_bv+1e-5)
            precision_bv_val /= (val_pos_num_bv+1e-5)
            with open(filename, 'a+') as f_log:
                f_log.write('cls_loss: %.4f, cls_neg_loss: %.4f, bbox_loss: %.4f, cls_num: %d, pos_cls_num_fv: %d, pos_recall_fv: %.4f, pos_precision_fv: %.4f, bbox_num: %d \n'%\
                        (val_cls_loss, val_cls_neg_loss, val_bbox_loss, val_cls_num, val_pos_cls_num, acc_pos_val, prec_pos_val, val_bbox_num))
            print (ignored_detection_count,'/', all_detection_count,' ignored detections')
            
            t0 = time.time()
            #write results to file
            #import pdb; pdb.set_trace()
            write_path = imdbs[ival]._write_kitti_results_voxel_file(all_ry, all_3d_bboxes, all_scores,result_path=(os.path.basename(self.output_dir)+'_'+data_names[ival]+'/iter'+str(1)))
            #pdb.set_trace()
            if not testset:
                imdbs[ival]._do_eval_bv(write_path, output_dir='output')
                t_val_write = (time.time()-t0)
                if imdbs[ival]._subset=='cars':
                    write_name='vehicle'
                elif imdbs[ival]._subset=='peds':
                    write_name='pedestrian'
                else:
                    assert False, 'wrong object name to read'

                PR_file = os.path.join(write_path,('../plot/'+write_name+'_detection_ground.txt'))
                try:
                    PRs = np.loadtxt(PR_file)
                    APs = np.sum(PRs[0:-1,1:4]*(PRs[1:,0:1]-PRs[0:-1,0:1]),axis=0)
                    conclusion_path = os.path.join(write_path,'../../../conclusion.txt')
                    with open(conclusion_path,'a+') as conclusion_file:
                        conclusion_file.write('iteration '+str(1)+': ')
                        conclusion_file.write('\nrecall            :\n')
                        PRs[:,0].tofile(conclusion_file," ",format='%.3f')
                        conclusion_file.write('\nprec_easy, AP: %.2f :\n'%APs[0])
                        PRs[:,1].tofile(conclusion_file," ",format='%.3f')
                        conclusion_file.write('\nprec_mod , AP: %.2f :\n'%APs[1])
                        PRs[:,2].tofile(conclusion_file," ",format='%.3f')
                        conclusion_file.write('\nprec_hard, AP: %.2f :\n'%APs[2])
                        PRs[:,3].tofile(conclusion_file," ",format='%.3f')
                        conclusion_file.write('cls_loss: %.4f, cls_neg_loss: %.4f, bbox_loss: %.4f, cls_num: %d, pos_cls_num_fv: %d, pos_recall_fv: %.4f, pos_precision_fv: %.4f, bbox_num: %d'%\
                            (val_cls_loss, val_cls_neg_loss, val_bbox_loss, val_cls_num, val_pos_cls_num, acc_pos_val, prec_pos_val, val_bbox_num))
                    with open(filename, 'a+') as f_log:
                        f_log.write('APs: %.3f, %.3f, %.3f'%(APs[0],APs[1],APs[2]))
                except:
                    #f_log.write('No object detected')
                    print ('No object detected')
            else:
                t_val_write=0.0

            with open(filename, 'a+') as f_log:
                f_log.write('speed: {:.3f}s / iter, anchor: {:.3f}s, proposal: {:.3f}s, write: {:.3f}s / iter \n'.\
                format(t_val/im_ind,ts_anchor_val/im_ind, ts_proposal_val/im_ind, t_val_write/im_ind))



def vis_detections(lidar_bv, image, calib, rpn_data, rpn_rois, rcnn_roi, gt_boxes_3d):
    import matplotlib.pyplot as plt
    from utils.transform import lidar_3d_to_corners, corners_to_bv
    from fast_rcnn.bbox_transform import bbox_transform_inv_cnr
    from utils.draw import show_lidar_corners, show_image_boxes, scale_to_255
    from utils.cython_nms import nms, nms_new


    image = image.reshape((image.shape[1], image.shape[2], image.shape[3]))
    image += cfg.PIXEL_MEANS
    image = image.astype(np.uint8, copy=False)
    lidar_bv = lidar_bv.reshape((lidar_bv.shape[1], lidar_bv.shape[2], lidar_bv.shape[3]))[:,:,8]
    # visualize anchor_target_layer output
    rpn_anchors_3d = rpn_data[3][:,1:7]
    rpn_bv = rpn_data[2][:,1:5]
    # rpn_label = rpn_data[0]
    # print rpn_label.shape
    # print rpn_label[rpn_label==1]
    rpn_boxes_cnr = lidar_3d_to_corners(rpn_anchors_3d)
    img = show_lidar_corners(image, rpn_boxes_cnr, calib)
    img_bv = show_image_boxes(scale_to_255(lidar_bv, min=0, max=2), rpn_bv)

    print (img.shape)
    # plt.ion()
    plt.title('anchor target layer before regression')
    plt.subplot(211)
    plt.imshow(img_bv)
    plt.subplot(212)
    plt.imshow(img)
    plt.show()

    # visualize proposal_layer output
    boxes_3d = rpn_rois[2][:, 1:7]
    boxes_bv = rpn_rois[0][:, 0:5]
    boxes_img = rpn_rois[1][:, 0:5]


    # keep = nms(boxes_img, cfg.TEST.NMS)
    # boxes_img = boxes_img[keep]
    # boxes_3d = boxes_3d[keep]
    # boxes_cnr = lidar_3d_to_corners(boxes_3d[:100])
    print (boxes_3d.shape)
    print (boxes_bv.shape)
    # image_cnr = show_lidar_corners(image, boxes_cnr, calib)

    image_bv = show_image_boxes(lidar_bv, boxes_bv[:, 1:5])
    image_img = show_image_boxes(image, boxes_img[:, 1:5])
    plt.title('proposal_layer ')
    plt.subplot(211)
    plt.imshow(image_bv)
    plt.subplot(212)
    plt.imshow(image_img)
    plt.show()


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print ('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print ('done')

    print ('Preparing training data...')
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print ('done')

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    #Note this one may ignore images with no objects
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print ('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after))
    return filtered_roidb


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=10000, val_imdb=None, val_roidb=None):
    """Train a Fast R-CNN network."""
    #roidb already has the regression targets for each window
    #roidb = filter_roidb(roidb)
    #if val_roidb!=None:
    #    val_roidb = filter_roidb(val_roidb)

    #should not use filter anymore
    saver = tf.train.Saver(max_to_keep=10)
    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model, val_imdb=val_imdb, val_roidb=val_roidb)
        print ('Solving...')
        sw.train_model(sess, max_iters)
        print ('done solving')

def evaluate_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=10000, val_imdb=None, val_roidb=None, testset=False):
    """Train a Fast R-CNN network."""
    #roidb already has the regression targets for each window
    #roidb = filter_roidb(roidb)
    #if val_roidb!=None:
    #    val_roidb = filter_roidb(val_roidb)

    #should not use filter anymore
    saver = tf.train.Saver(max_to_keep=10)
    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model, val_imdb=val_imdb, val_roidb=val_roidb)
        print ('Evaluating...')
        sw.evaluate_model(sess,testset=testset)
        print ('done evaluating')
