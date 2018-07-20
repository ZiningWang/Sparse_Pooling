WZN2018: This is a simplified version where keeps only my latest modification.
WZN2018: This is a more simplified version, only keeping the necessary modifications to reproduce the SHPL in the paper with one-stage network. 

# One-stage Fusion-Based Detection Network fusing VoxelNet and MSCNN for Object Detection on KITTI

This is an Tensorflow implementation of the fusion of VoxelNet and MSCNN, which is the one-stage detection network mentioned in [**Fusing Bird View LIDAR Point Cloud and Front View Camera Image for Deep Object Detection**](https://arxiv.org/abs/1711.06703). The code structure is based on MV3D_TF(an old version, unavailable anymore). However, the network and training process used are different from the original code by a very large extent. 


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

   Download the pre-trained MSCNN models (VGG16 trained with the code from [[MSCNN]](https://github.com/zhaoweicai/mscnn)) [[Google Drive]](https://drive.google.com/file/d/1RZZrxmfUzkGLEYrVYkXDEk3lNXR36xJS/view?usp=sharing) 

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

### Examples



