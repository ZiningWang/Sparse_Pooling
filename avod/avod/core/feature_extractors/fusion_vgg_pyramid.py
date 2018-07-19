import tensorflow as tf

from avod.utils.sparse_pool_utils import sparse_pool_layer

slim = tf.contrib.slim


class FusionVggPyr:
    """Define two feature extractors
    """

    def __init__(self,extractor_config_bev,extractor_config_img,M_tf,img_index_flip,bv_index=None):
        # M_tf is the matrix for sparse pooling layer
        self.config_bev = extractor_config_bev
        self.config_img = extractor_config_img
        self.M_tf = M_tf
        self.img_index_flip = img_index_flip
        self.bv_index = bv_index

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    #WZN: build up the vgg pyramid layer with two scources from bev lidar and fv img separately,
    # with the same architecture, fusing feature maps from both sides right at the bottlenecks of backbones

    def build(self,
            inputs_bev,
            inputs_img,
            input_pixel_size_bev,
            input_pixel_size_img,
            is_training,
            scope_bev = 'bev_vgg_pyr',
            scope_img = 'img_vgg_pyr'):

        #WZN: first build two convs
        convs_bev, end_points_bev1 = self._build_individual_before_fusion(inputs_bev,input_pixel_size_bev,is_training,scope_bev,backbone_name='bev')
        convs_img, end_points_img1 = self._build_individual_before_fusion(inputs_img,input_pixel_size_img,is_training,scope_img,backbone_name='img') 

        #do the sparse pooling operation and get the fused conv4_bev and conv4_img
        feature_depths = [self.config_img.vgg_conv4[1],self.config_bev.vgg_conv4[1]] 
        #WZN: the depth of the output feature map (for pooled features bev, img respectively)
        #only fusion on the bev_map
        conv4_bev = convs_bev[-1]
        conv4_img = convs_img[-1]
        convs_bev[-1], convs_img[-1] = sparse_pool_layer([conv4_bev,conv4_img], feature_depths, self.M_tf, 
                        img_index_flip = self.img_index_flip, bv_index = self.bv_index, use_bn=False, training=is_training)
    

        #WZN: then do the deconv to the fused layers
        feature_maps_bev, end_points_bev2 = self._build_individual_after_fusion(convs_bev,input_pixel_size_bev,is_training,scope_bev,backbone_name='bev')
        feature_maps_img, end_points_img2 = self._build_individual_after_fusion(convs_img,input_pixel_size_img,is_training,scope_img,backbone_name='img')

        #merge the dicts, (for each backbone)
        #WZN: requires python3.5 or higher version
        end_points_bev = {**end_points_bev1, **end_points_bev2}
        end_points_img = {**end_points_img1, **end_points_img2}

        return feature_maps_bev, feature_maps_img, end_points_bev, end_points_img

    def _build_individual_before_fusion(self,
              inputs,
              input_pixel_size,
              is_training,
              scope,
              backbone_name):
        """ Modified VGG for BEV feature extraction with pyramid features

        Args:
            inputs: a tensor of size [batch_size, height, width, channels].
            input_pixel_size: size of the input (H x W)
            is_training: True for training, False for validation/testing.
            scope: Optional scope for the variables.

        Returns:
            The last op containing the log predictions and end_points dict.
        """
        if backbone_name == 'bev':
            vgg_config = self.config_bev
            scope_mid_name = 'bev_vgg_pyr'
        elif backbone_name == 'img':
            vgg_config = self.config_img
            scope_mid_name = 'img_vgg_pyr'
        else:
            error('Unknown name of single sensor backbone')

        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, scope_mid_name, [inputs]) as sc:

                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    if backbone_name == 'bev':
                        # Pad 700 to 704 to allow even divisions for max pooling
                        padded = tf.pad(inputs, [[0, 0], [4, 0], [0, 0], [0, 0]])
                    elif backbone_name == 'img':
                        padded = inputs
                    else:
                        error('Unknown name of single sensor backbone')

                    # Encoder
                    conv1 = slim.repeat(padded,
                                        vgg_config.vgg_conv1[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv1[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv1')
                    pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')

                    conv2 = slim.repeat(pool1,
                                        vgg_config.vgg_conv2[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv2[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv2')
                    pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')

                    conv3 = slim.repeat(pool2,
                                        vgg_config.vgg_conv3[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv3[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv3')
                    pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')

                    conv4 = slim.repeat(pool3,
                                        vgg_config.vgg_conv4[0],
                                        slim.conv2d,
                                        vgg_config.vgg_conv4[1],
                                        [3, 3],
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params={
                                            'is_training': is_training},
                                        scope='conv4')

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)  
                
                return [conv1,conv2,conv3,conv4], end_points

    def _build_individual_after_fusion(self,
              convs,
              input_pixel_size,
              is_training,
              scope,
              backbone_name):    
    # WZN: in the current stage, only one pooling is performed at the bottleneck, no further pooling during the devonv operations
        if backbone_name == 'bev':
            vgg_config = self.config_bev
            scope_mid_name = 'bev_vgg_pyr'
        elif backbone_name == 'img':
            vgg_config = self.config_img
            scope_mid_name = 'img_vgg_pyr'
        else:
            error('Unknown name of single sensor backbone')

        conv1 = convs[0]
        conv2 = convs[1]
        conv3 = convs[2]
        conv4 = convs[3]
        with slim.arg_scope(self.vgg_arg_scope(
                weight_decay=vgg_config.l2_weight_decay)):
            with tf.variable_scope(scope, scope_mid_name) as sc:
                end_points_collection = sc.name + '_end_points'

                # Collect outputs for conv2d, fully_connected and max_pool2d.
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    outputs_collections=end_points_collection):

                    # Decoder (upsample and fuse features)
                    upconv3 = slim.conv2d_transpose(
                        conv4,
                        vgg_config.vgg_conv3[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv3')

                    concat3 = tf.concat(
                        (conv3, upconv3), axis=3, name='concat3')
                    pyramid_fusion3 = slim.conv2d(
                        concat3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion3')

                    upconv2 = slim.conv2d_transpose(
                        pyramid_fusion3,
                        vgg_config.vgg_conv2[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv2')

                    concat2 = tf.concat(
                        (conv2, upconv2), axis=3, name='concat2')
                    pyramid_fusion_2 = slim.conv2d(
                        concat2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion2')

                    upconv1 = slim.conv2d_transpose(
                        pyramid_fusion_2,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        stride=2,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='upconv1')

                    concat1 = tf.concat(
                        (conv1, upconv1), axis=3, name='concat1')
                    pyramid_fusion1 = slim.conv2d(
                        concat1,
                        vgg_config.vgg_conv1[1],
                        [3, 3],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training},
                        scope='pyramid_fusion1')

                    # Slice off padded area
                    if backbone_name == 'bev':
                        sliced = pyramid_fusion1[:, 4:]
                    elif backbone_name == 'img':
                        sliced = pyramid_fusion1
                    else:
                        error('Unknown name of single sensor backbone')

                feature_maps_out = sliced

                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                return feature_maps_out, end_points



    

