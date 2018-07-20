#This is for transforming the .npy file saved in python2 to python3
import numpy as np 
import os
import scipy.io
#make matlab file
params = scipy.io.loadmat('mscnn_ped_cyc_kitti_trainval_2nd_iter_15000.mat')
for layer in params.keys():
	print(layer)
import pdb
pdb.set_trace()
