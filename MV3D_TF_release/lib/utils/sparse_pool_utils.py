import tensorflow as tf
import numpy as np
import avod.utils.transform as transform
slim = tf.contrib.slim

def gen_sparse_pooling_input_avod(points,voxel_indices,stereo_calib,im_size,bv_size):
    #generate sparse pooling input online, following the pipeline of avod, which uses raw lidar point cloud and
    #instead of projected point cloud in camera frame.
    # im_size is the WxH size of the front-view image, bv_size is the WxH size of the bird's-eye view image
    #( however, output of bv is HxW) 062318: no, bv_size input is HxW
    bv_index = np.vstack((voxel_indices[:,0],voxel_indices[:,1])).transpose()
    P = stereo_calib.p2
    #actually the input of avod is already clipped, but we would like to clip it again to ensure the same as MV3D_TF
    slice_index = transform.clip3DwithinImage(points.transpose(),P,im_size)
    bv_index = bv_index[slice_index]
    slice_points = points[slice_index]
    img_points = transform.projectToImage(slice_points.transpose(),P)
    img_index = np.round(img_points).astype(int)
    img_index = np.vstack((img_index,np.zeros((1,img_index.shape[1]))))
    return {'bv_index':bv_index,'img_index':img_index,'bv_size':np.array([bv_size[0],bv_size[1]]),'img_size':np.array(im_size)}

def produce_sparse_pooling_input(input_dict,M_val=None,stride=[1,1]):
    #stride[0] is bv stride, [1] is img stride
    bv_index = input_dict['bv_index']
    img_index = input_dict['img_index']
    bv_size = input_dict['bv_size']
    im_size = input_dict['img_size']

    assert img_index.shape[0]==3, 'wrong img_index shape, should be 3xN instead '+ str(img_index.shape) #[u,v,1]
    img_index[0:2,:] = np.floor(img_index[0:2,:] / stride[0])
    im_size = np.floor(im_size/stride[0])
    #image size is WxH, img_index is WxH
    img_index[0,img_index[0,:]>=im_size[0]]=im_size[0]-1
    img_index[1,img_index[1,:]>=im_size[1]]=im_size[1]-1
    #flip because the image tensor is HxW
    img_index_flip_pool = np.floor(np.fliplr(img_index.transpose())).astype(int)

    bv_size = np.floor(np.array(bv_size)/stride[1]) #this assumes all paddings are 'Valid'
    bv_index_down = np.floor(bv_index/stride[1])
    #bv_size is HxW, bv_index is WxH
    bv_index1_pool = (bv_index_down[:,1]*bv_size[1]+bv_index_down[:,0]).astype(int)
    ind_inside = bv_index1_pool<int(bv_size[0]*bv_size[1])

    img_index_flip_pool = img_index_flip_pool[ind_inside]
    bv_index1_pool = bv_index1_pool[ind_inside]
    N = bv_index1_pool.shape[0]

    Mij_pool = np.vstack((bv_index1_pool,np.array(range(N)))).transpose()
    M_size = np.array([bv_size[0]*bv_size[1],N]).astype(int)

    #import pdb
    #pdb.set_trace()
    if M_val is None:
        M_val = np.ones(N)
    return {'Mij_pool':Mij_pool,'M_val':M_val,'M_size':M_size,'img_index_flip_pool':img_index_flip_pool,'bev_index_flip_pool':np.zeros((0,3))}


def sparse_pool_layer(inputs,feature_depths, M, img_index_flip = None ,bv_index = None, use_bn=False, training=True):
    #note feature_depths is the depths of the pooled features
    input_bv = inputs[0]
    input_img = inputs[1]
    if not (img_index_flip is None):
        feature_depth_bv = feature_depths[0]
        pooled_size = [1,tf.shape(input_bv)[1],tf.shape(input_bv)[2],feature_depth_bv]
        img_pooled = _sparse_pool_op(M,input_img,img_index_flip,pooled_size)
        if use_bn:
            bv_fused = concat_bn_op([input_bv,img_pooled],axis=3,training=training)
        else:
            bv_fused = tf.concat(values=[input_bv,img_pooled], axis=3)
    else:
        bv_fused = input_bv

    #import pdb
    #pdb.set_trace()

    if not (bv_index is None):
        feature_depth_img = feature_depths[1]
        pooled_size = [1,tf.shape(input_img)[1],tf.shape(input_img)[2],feature_depth_img]
        bv_pooled = _sparse_pool_trans_op(M,input_bv,img_index_flip,pooled_size)
        if use_bn:
            img_fused = concat_bn_op([input_img,bv_pooled],axis=3,training=training)
        else:
            img_fused = tf.concat(values=[input_img,bv_pooled], axis=3)
        #return ValueError('Not implemented: from bv to img fused')
    else:
        img_fused = input_img

    return bv_fused,img_fused



def _sparse_pool_op(M,input,source_index,pooled_size):
    #0 is sparse transformation matrix, 1 is source feature, 2 is scource pooling index
    #only support batch size 1
    img_pooled_tf = tf.gather_nd(input,source_index)
    bv_flat_tf = tf.sparse_tensor_dense_matmul(M,img_pooled_tf)
    return tf.reshape(bv_flat_tf,pooled_size)

def _sparse_pool_trans_op(M,input,source_index,pooled_size):
    #the other path of the sparse pooling
    #pooled_size is different from the sparse_pool_op
    #source_index is the img_index by default
    #only support batch size 1
    input_size = tf.shape(input) #(batch, height, width, depth)
    bv_pooled_sequence = tf.sparse_tensor_dense_matmul(tf.sparse_transpose(M),tf.reshape(input,[-1,input_size[3]]))
    # use scatter_nd_add
    #bv_pooled = tf.Variable(tf.zeros(pooled_size))
    #return tf.scatter_nd_add(bv_pooled,source_index,bv_pooled_sequence)
    # use scatter_nd, however, the current verison of tf seems not to accumulate duplicate indexes
    bv_pooled = tf.scatter_nd(source_index,bv_pooled_sequence,pooled_size)
    return bv_pooled


def concat_bn_op(inputs,axis,training):
    #concatenate two with batch normalization
    input0_bn = slim.batch_norm(inputs[0],training=training)
    input1_bn = slim.batch_norm(inputs[1],training=training)
    return tf.concat(values=[input0_bn,input1_bn], axis=axis)