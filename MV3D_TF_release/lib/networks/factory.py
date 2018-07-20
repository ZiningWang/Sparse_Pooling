
__sets = {}

#import networks.VGGnet_train
#import networks.VGGnet_test
import pdb
import tensorflow as tf
from networks.MV3D_train import MV3D_train
from networks.MV3D_test import MV3D_test
#from networks.MV3D_resnet_train import  MV3D_resnet_train
from networks.MV3D_voxel_train import fusion_voxel_train
#__sets['VGGnet_train'] = networks.VGGnet_train()

#__sets['VGGnet_test'] = networks.VGGnet_test()


def get_network(name,use_bn=False,use_focal=False,use_fusion=False,use_dropout=False):
    """Get a network by name."""
    #if not __sets.has_key(name):
    #    raise KeyError('Unknown dataset: {}'.format(name))
    #return __sets[name]
    if name.split('_')[1] == 'test':
       return MV3D_test()
    elif name.split('_')[1] == 'train':
       return MV3D_train(use_focal=use_focal)
    elif name.split('_')[1] == 'restrain':
      raise KeyError('Not included in this version.')
      #return  MV3D_resnet_train(use_bn=use_bn,use_focal=use_focal)
    elif name.split('_')[1] == 'voxeltrain':
      return  fusion_voxel_train(use_bn=use_bn,use_focal=use_focal,use_fusion=use_fusion,use_dropout=use_dropout)
    else:
       raise KeyError('Unknown dataset: {}'.format(name))


def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
