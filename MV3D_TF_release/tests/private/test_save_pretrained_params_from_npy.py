#This is for transforming the .npy file saved in python2 to python3
import numpy as np 
import os
import scipy.io
params = np.load('mscnn_ped_cyc_kitti_trainval_2nd_iter_15000.npy').item()
#make matlab file
scipy.io.savemat('mscnn_ped_cyc_kitti_trainval_2nd_iter_15000.mat',params)

''' #make txt files(failed)
os.mkdir('npy_params_csv')
for layer_name in params.keys():
	layer = params[layer_name]
	if type(layer) is dict:
		os.mkdir(layer_name)
		for layer_param_name in layer:
			layer_param = layer[layer_param_name]
			np.savetxt(layer_param_name+'.csv',layer_param)
	else:
		np.savetxt(layer_name+'.csv',layer)
'''
