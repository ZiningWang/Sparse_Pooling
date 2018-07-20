


# Implementation of Sparse Non-homogeneous Pooling on Fusion-based Detection Networks
Sparse Pooling for LIDAR and Camera Fusion by Zining Wang

## Claim

Sparse Pooling is for fusing feature maps from different views and sources. Sparsity comes from the sparse correspondences between cells, such as point cloud. The layer is not only restricted to detection but also useful for segmentation. The main part is based on [**Fusing Bird View LIDAR Point Cloud and Front View Camera Image for Deep Object Detection**](https://arxiv.org/abs/1711.06703), but the SHPL introduced in the paper is also implemented to other fusion-based detection networks in this repository.

As the majority of the code comes from many existing repositories, including [MV3D_TF](https://github.com/leeyevi/MV3D_TF), [Avod-FPN](https://github.com/kujason/avod), [VoxelNet](https://github.com/jeasinema/VoxelNet-tensorflow) and [MSCNN](https://github.com/zhaoweicai/mscnn). If there is any thing inappropriate, please contact me through wangzining@berkeley.edu.

Results will be updated.

## Usage
The following two networks can be downloaded and run separately.

`avod` based on [Avod-FPN](https://github.com/kujason/avod) is the additional part. It aims to show the SHPL can be easily added to other networks to allow efficient fusion of feature maps.

`MV3D_TF_release` is the network introduced in the paper. It fuses [VoxelNet](https://github.com/jeasinema/VoxelNet-tensorflow) and [MSCNN](https://github.com/zhaoweicai/mscnn) and uses focal loss for one-stage detection. 



## Code added/changed
This is for avod.
### Added:  
`core/feature_extractors/fusion_bgg_pyramid.py`:  For fusion right after vgg  
`utils/transform.py`:  Basic utility functions to preprocess input data  
`utils/sparse_pool_utils.py`: Main functions to construct the Sparse Non-homogeneous Pooling Layer

### Changed:
`bev_slices.py`: See code under `#WZN`. Add additional inputs of voxel indices for Sparse Pooling  
`rpn_model.py`:  See code under `#WZN`. Add placeholders and Sparse Pooling layer before rpn (controled by new parameters in config)  
`model.proto`:   Add 5 more parameters to control Sparse Pooling


## Reference
The 'Non-homogeneous' term came from [Spatial Transformer Networks](https://github.com/kevinzakka/spatial-transformer-network)


## LICENSE
Copyright (c) 2018 [Zining Wang](https://github.com/ZiningWang)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.