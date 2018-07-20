import numpy as np
import os
#fusion wrong
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0110_0047_valid/iter30000/kitti_peds_val_-01-09-22-36-27/plot'
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0110_0047_valid/iter20000/kitti_peds_val_-01-09-17-18-22/plot'
result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0110_0047_valid/iter80000/kitti_peds_val_-01-10-21-57-13/plot'
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0110_0047_train/iter70000/kitti_peds_train_-01-10-17-26-38/plot'
#voxelnet
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0109_2148_valid/iter40000/kitti_peds_val_-01-09-22-34-18/plot'
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0109_2148_valid/iter30000/kitti_peds_val_-01-09-16-54-04/plot'
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0109_2148_train/iter30000/kitti_peds_train_-01-09-18-06-21/plot'
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0109_2148_train/iter40000/kitti_peds_train_-01-09-23-25-23/plot'
result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0109_2148_valid/iter50000/kitti_peds_val_-01-10-03-22-34/plot'
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0109_2148_valid/iter90000/kitti_peds_val_-01-10-20-39-46/plot'
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0109_2148_train/iter70000/kitti_peds_train_-01-10-12-46-34/plot'
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0109_2148_valid/iter100000/kitti_peds_val_-01-11-00-55-43/plot'

#fusion correct
result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0112_0025_valid/iter45000/kitti_peds_val_-01-12-02-50-58/plot'
#result_dir = '/data/RPN/fromBDD/MV3D_TF/kitti/results/voxelnet_train_peds_at0112_0025_valid/iter60000/kitti_peds_val_-01-12-07-46-56/plot'

filename = 'pedestrian_detection_ground.txt'
PR_file = os.path.join(result_dir,filename)

PRs = np.loadtxt(PR_file)
APs = np.sum(PRs[0:-1,1:4]*(PRs[1:,0:1]-PRs[0:-1,0:1]),axis=0)
print (APs[0], APs[1], APs[2])
