import os,sys
lib_path = os.path.abspath(os.path.join('../../lib'))
print(lib_path)
sys.path.append(lib_path)
from fast_rcnn.train_mv_voxel import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
import pdb
import tensorflow as tf
from utils.config_voxels import cfg as cfg_voxels

class args_init(object):
    def __init__(self):
        self.device='gpu'
        self.device_id=0
        self.solver=None
        self.max_iters=200002
        self.pretrained_model=None
        self.cfg_file=None
        self.imdb_name=None
        self.val_imdb_name=None
        self.randomize=False
        self.network_name=None


#config from input arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('object_name', type=str,default='cars')
parser.add_argument('train_dataset', type=str,default='train')
parser.add_argument('--val_dataset', type=str,default=None)
parser.add_argument('--exp_name', type=str,default='N')
parser.add_argument('--use_bn', type=bool,default=True)
# use_focal=True means focal loss for all labels, use_local=False means focal for positives only
parser.add_argument('--use_focal', type=bool,default=False)
parser.add_argument('--use_fusion', type=bool,default=True)
parser.add_argument('--use_dropout', type=bool,default=True)
args_in = parser.parse_args()
object_name = args_in.object_name

#config from file
args = args_init()
args.pretrained_model= '../../tests/mscnn_ped_cyc_kitti_trainval_2nd_iter_15000.mat'

args.imdb_name='kitti_'+args_in.train_dataset
#for training validation set is val
if args_in.train_dataset=='train':
    args.val_imdb_name='kitti_val'
else:
    args.val_imdb_name=None
#if validation set is specified
if not (args_in.val_dataset is None):
    args.val_imdb_name='kitti_'+args_in.val_dataset

args.network_name='kitti_voxeltrain'
print(args.imdb_name)
if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
print('Using config:')
pprint.pprint(cfg)
imdb = get_imdb(args.imdb_name)
imdb.set_object(object_name)
if args.val_imdb_name==None:
    val_imdb=None
    val_roidb=None
else:
    val_imdb=get_imdb(args.val_imdb_name)
    val_imdb.set_object(object_name)
    val_roidb = get_training_roidb(val_imdb)

print('Loaded dataset `{:s}` for training'.format(imdb.name))
if not args.randomize:
    np.random.seed(cfg.RNG_SEED)
#labels
roidb = get_training_roidb(imdb)
    
#output dir
output_dir = get_output_dir(imdb, 'voxelnet_train_'+object_name+'_'+args_in.exp_name)
print('Output will be saved to `{:s}`'.format(output_dir))
#network load, rest every time

#input data
from roi_data_layer.layer import RoIDataLayer
data_layer = RoIDataLayer(roidb, imdb.num_classes)
blobs = data_layer.forward()
for key in blobs:
    print(key)
data_layer = RoIDataLayer(roidb, imdb.num_classes)
blobs = data_layer.forward()
#print blobs
print(blobs['im_info'])
print(cfg_voxels.INPUT_WIDTH, cfg_voxels.INPUT_HEIGHT)
print(blobs['gt_rys'].shape)
print(blobs['voxel_data']['coordinate_buffer'].shape)


#set network
#use_focal: focal loss for negative; use_bn: batch norm after sparse pooling; use_fusion: use img data;
tf.reset_default_graph()
network = get_network(args.network_name,use_bn=args_in.use_bn,use_focal=args_in.use_focal,use_fusion=args_in.use_fusion,use_dropout=args_in.use_dropout)
print('Using network:', args.network_name)

train_net(network, imdb, roidb, output_dir,
          pretrained_model=args.pretrained_model,
          max_iters=args.max_iters,val_imdb=val_imdb,val_roidb=val_roidb)
