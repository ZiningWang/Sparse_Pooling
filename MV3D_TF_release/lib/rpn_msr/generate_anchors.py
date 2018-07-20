# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
from utils.config_voxels import cfg as cfg_voxels

DEFAULT_ANCHOR = [cfg_voxels.ANCHOR_L,cfg_voxels.ANCHOR_W,cfg_voxels.ANCHOR_H]
# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors_bv(base_size=[[3.9, 1.6], [1.0, 0.6]], res=0.1):#(base_size=[[3.9, 1.6], [1.0, 0.6]], res=0.1): #[[1.6, 1.2], [1.0, 0.7]]
    """
    Generate anchor (reference) windows for lidar bird view
    """

    base_anchors = np.vstack([[0, 0, int(base[0]/res), int(base[1]/res)] for base in base_size])
    
    base_anchors[:,0] -= base_anchors[:,2]//2
    base_anchors[:,1] -= base_anchors[:,3]//2
    base_anchors[:,2] -= base_anchors[:,2]//2
    base_anchors[:,3] -= base_anchors[:,3]//2
    #WZN: a fwd anchor with a side anchor together, so totally len(base_size)*2 anchors
    anchors = np.vstack((base_anchors, base_anchors[:,[1,0,3,2]]))

    return anchors

def generate_anchors_3d_angle(base_size=[DEFAULT_ANCHOR]):#(base_size=[[3.9, 1.6], [1.0, 0.6]], res=0.1):[3.9,1.6,1.56][1.0, 0.6, 1.73]
    """
    Generate anchor (reference) windows for lidar 3D with angles!!
    (x1,y1,x2,y2), (x1,y1,x2,y2)
    """

    base_anchors = np.vstack([[0, 0, 0, base[0], base[1], base[2], 0] for base in base_size])

    angle90 = base_anchors[:,6:7]+np.pi/2
    anchors_3d_bbox = np.vstack((base_anchors, np.hstack((base_anchors[:,0:6],angle90))))
    
    base_anchors[:,0] -= base_anchors[:,3]/2  #x1
    #base_anchors[:,1] -= base_anchors[:,5]/2 #y1 
    base_anchors[:,2] -= base_anchors[:,4]/2  #z1
    base_anchors[:,3] -= base_anchors[:,3]/2  #x2
    base_anchors[:,4] -= base_anchors[:,4]/2  #z2
    #base_anchors[:,5] -= base_anchors[:,5]/2 #y2
    #WZN: a fwd anchor with a side anchor together, so totally len(base_size)*2 anchors
    anchors_2d_cnr = np.vstack((base_anchors[:,[0,2,3,4]], base_anchors[:,[0,2,3,4]]))

    return anchors_3d_bbox.astype(np.float32),anchors_2d_cnr.astype(np.float32)

def generate_anchors(base_size=32, ratios=np.array([1.54]),
                     scales=np.array([6,5,4,3,2,1.25])):  #WZN: for front view
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print (time.time() - t)
    print (a)
    from IPython import embed; embed()
