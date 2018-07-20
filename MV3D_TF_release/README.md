WZN2018: This is a simplified version where keeps only my latest modification.
WZN2018: This is a more simplified version, only keeping the necessary modifications to reproduce the SHPL in the paper with one-stage network. 

# MV3D_TF(In progress)

This is an experimental Tensorflow implementation of the fusion of VoxelNet and MSCNN. The code structure is based on MV3D_TF. However, the 
network and training process used are different from the original code by a very large extent. 


### Requirements: software

1. Code is tested with TensorFlow 1.8, CUDA9.0, cudnnv7.1 , python3.5.2, TITAN Xp

2. Python packages you might not have: `cython`, `python-opencv`, `easydict`

### Requirements: hardware

1. 9G of GPU memory is sufficient for the training with both MSCNN and VoxelNet

### Installation 

1. Clone the Faster R-CNN repository
```Shell
  git clone --recursive https://github.com/RyannnG/MV3D_TF.git
```

2. Build the Cython modules
   ```Shell
    cd $MV3D/lib
    make
   ```

3. Downloads KITTI object datasets.

```Shell
 % Specify KITTI data path so that the structure is like

 % {kitti_dir}/object/training/image_2
 %                            /calib
 %							             /velodyne
       

 % {kitti_dir}/object/testing/image_2
 %                           /calib
 %							             /velodyne
```


4. Preprocess Lidar data with Voxel representation
  The lidar input representation is in Voxel which is totally different from that of MV3D. The preprocessing clips the Lidar data by the visible area of the camera and transform the point cloud into camera frame.

   ```shell
   # edit the kitti_path in tools/data_preprocess.py
   # then start make data
   python tools/preprocess.py
   ```

5. Create symlinks for the KITTI dataset

```Shell
   cd $MV3D/data/KITTI
   ln -s {kitti_dir}/object object
```

5. Download pre-trained ImageNet models

   Download the pre-trained MSCNN models (VGG16 trained with the code from TODO:LINK TO MSCNN) [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) [[Dropbox]](https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy?dl=0)

```Shell
    mv mscnn_ped_cyc_kitti_trainval_2nd_iter_15000.mat $MV3D/tests/mscnn_ped_cyc_kitti_trainval_2nd_iter_15000.mat
```


6. Run script to train model 
```Shell
 cd $MV3D/experiments/scripts
 CUDA_VISIBLE_DEVICES=2 python3.5 train_voxelnet_fusion.py cars train --exp_name example_train
```
There are other arguments available:

```Shell
 CUDA_VISIBLE_DEVICES=2 python3.5 train_voxelnet_fusion.py cars train --exp_name example_train --use_fusion True --use_focal False --use_dropout False
```

It is also possible to the VoxelNet only with the one-stage detection framework.

### Network Structure

Key idea: Use Lidar bird view to generate anchor boxes, then project those boxes on image to do classification.

![structure](examples/mv3d_4.png)

### Examples

Image and corresponding Lidar map 

**Note:**

In image:

+ Boxes  without regression

In Lidar:

+ white box: without regression (correspond with image)
+ purple box: with regression

1.

![figure_20](examples/figure_27.png)

![figure_20](examples/jlidar27.png)

2.

![figure_20](examples/figure_30.png)

![figure_20](examples/lidar30.png)

3. ​

![figure_20](examples/figure_13.png)

![figure_20](examples/lidar13.png)

4.

![figure_20](examples/figure_29.png)

![figure_20](examples/lidar29.png)

### Existing Errors

Mostly due to regression error

![figure_20](examples/figure_10.png)

(error in box 5,6,9)

![figure_20](examples/lidar10.png)

![figure_20](examples/figure_33.png)

(error in 8, 9, 10)

![figure_20](examples/lidar33.png)

### References

[Lidar Birds Eye Views](http://ronny.rest/blog/post_2017_03_26_lidar_birds_eye/)

[part.2: Didi Udacity Challenge 2017 — Car and pedestrian Detection using Lidar and RGB](https://medium.com/@hengcherkeng/part-1-didi-udacity-challenge-2017-car-and-pedestrian-detection-using-lidar-and-rgb-fff616fc63e8)

[Faster_RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)

[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[TFFRCNN](https://github.com/CharlesShang/TFFRCNN)

