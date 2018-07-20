#!/usr/bin/env python
# -*- cooing:UTF-8 -*-

# File Name : utils.py
# Purpose :
# Creation Date : 09-12-2017
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import numpy as np
import shapely.geometry
import shapely.affinity

from utils.config_voxels import cfg
#from box_overlaps import *


def lidar_to_bird_view(x, y, factor=1):
    # using the cfg.INPUT_XXX
    return (x - cfg.X_MIN) / cfg.VOXEL_X_SIZE * factor, (y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor


def camera_to_lidar(x, y, z):
    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_R_RECT_0)), p)
    p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_T_VELO_2_CAM)), p)
    p = p[0:3]
    return tuple(p)


def lidar_to_camera(x, y, z):
    p = np.array([x, y, z, 1])
    p = np.matmul(np.array(cfg.MATRIX_T_VELO_2_CAM), p)
    p = np.matmul(np.array(cfg.MATRIX_R_RECT_0), p)
    p = p[0:3]
    return tuple(p)


def camera_to_lidar_point(points):
    # (N, 3) -> (N, 3)
    ret = []
    for p in points:
        x, y, z = p
        p = np.array([x, y, z, 1])
        p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_R_RECT_0)), p)
        p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_T_VELO_2_CAM)), p)
        p = p[0:3]
        ret.append(p)
    return np.array(ret).reshape(-1, 3)


def lidar_to_camera_point(points):
    # (N, 3) -> (N, 3)
    ret = []
    for p in points:
        x, y, z = p
        p = np.array([x, y, z, 1])
        p = np.matmul(np.array(cfg.MATRIX_T_VELO_2_CAM), p)
        p = np.matmul(np.array(cfg.MATRIX_R_RECT_0), p)
        p = p[0:3]
        ret.append(p)
    return np.array(ret).reshape(-1, 3)


def camera_to_lidar_box(boxes):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(
            x, y, z), h, w, l, -ry - np.pi / 2
        ret.append([x, y, z, h, w, l, rz])
    return np.array(ret).reshape(-1, 7)


def lidar_to_camera_box(boxes):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidar_to_camera(
            x, y, z), h, w, l, -rz - np.pi / 2
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret).reshape(-1, 7)


def center_to_corner_box2d(boxes_center):
    # (N, 5) -> (N, 4, 2)
    N = boxes_center.shape[0]
    boxes3d_center = np.zeros((N, 7))
    boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
    boxes3d_corner = center_to_corner_box3d(boxes3d_center)

    return boxes3d_corner[:, 0:4, 0:2]


def center_to_corner_box3d(boxes_center):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet\
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    return ret


def corner_to_center_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 5)
    N = boxes_corner.shape[0]
    boxes3d_corner = np.zeros((N, 8, 3))
    boxes3d_corner[:, 0:4, 0:2] = boxes_corner
    boxes3d_corner[:, 4:8, 0:2] = boxes_corner
    boxes3d_center = corner_to_center_box3d(boxes3d_corner)

    return boxes3d_center[:, [0, 1, 4, 5, 6]]


def corner_to_standup_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)

    return standup_boxes2d


# TODO: 0/90 may be not correct
def anchor_to_standup_box2d(anchors):
    # (N, 4) -> (N, 4) x,y,w,l -> x1,y1,x2,y2
    anchor_standup = np.zeros_like(anchors)
    # r == 0
    anchor_standup[::2, 0] = anchors[::2, 0] - anchors[::2, 3] / 2
    anchor_standup[::2, 1] = anchors[::2, 1] - anchors[::2, 2] / 2
    anchor_standup[::2, 2] = anchors[::2, 0] + anchors[::2, 3] / 2
    anchor_standup[::2, 3] = anchors[::2, 1] + anchors[::2, 2] / 2
    # r == pi/2
    anchor_standup[1::2, 0] = anchors[1::2, 0] - anchors[1::2, 2] / 2
    anchor_standup[1::2, 1] = anchors[1::2, 1] - anchors[1::2, 3] / 2
    anchor_standup[1::2, 2] = anchors[1::2, 0] + anchors[1::2, 2] / 2
    anchor_standup[1::2, 3] = anchors[1::2, 1] + anchors[1::2, 3] / 2

    return anchor_standup


def corner_to_center_box3d(boxes_corner):
    # (N, 8, 3) -> (N, 7)
    ret = []
    for roi in boxes_corner:
        if cfg.CORNER2CENTER_AVG:  # average version
            roi = np.array(roi)
            h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
            w = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                ptnp.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            ) / 4
            l = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            ) / 4
            x, y, z = np.sum(roi, axis=0) / 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
        else:  # max version
            h = max(abs(roi[:4, 1] - roi[4:, 1]))
            w = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            )
            l = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            )
            x, y, z = np.sum(roi, axis=0) / 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
        ret.append([x, y, z, h, w, l, ry])
    return ret


# this just for visulize
def lidar_box3d_to_camera_box(boxes3d, cal_projection=False):
    # (N, 7) -> (N, 4)/(N, 8, 2)  x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
    num = len(boxes3d)
    boxes2d = np.zeros((num, 4), dtype=np.int32)
    projections = np.zeros((num, 8, 2), dtype=np.float32)

    boxes3d_corner = center_to_corner_box3d(boxes3d)
    # TODO: here maybe some problems, check Mt/Kt
    Mt = np.array(cfg.MATRIX_Mt)
    Kt = np.array(cfg.MATRIX_Kt)

    for n in range(num):
        box3d = boxes3d_corner[n]
        Ps = np.hstack((box3d, np.ones((8, 1))))
        Qs = np.matmul(Ps, Mt)
        Qs = Qs[:, 0:3]
        qs = np.matmul(Qs, Kt)
        zs = qs[:, 2].reshape(8, 1)
        qs = (qs / zs)

        projections[n] = qs[:, 0:2]
        minx = int(np.min(qs[:, 0]))
        maxx = int(np.max(qs[:, 0]))
        miny = int(np.min(qs[:, 1]))
        maxy = int(np.max(qs[:, 1]))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d


def lidar_to_bird_view_img(lidar, factor=1):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    birdview = np.zeros(
        (cfg.INPUT_HEIGHT * factor, cfg.INPUT_WIDTH * factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if cfg.X_MIN < x < cfg.X_MAX and cfg.Y_MIN < y < cfg.Y_MAX:
            x, y = int((x - cfg.X_MIN) / cfg.VOXEL_X_SIZE *
                       factor), int((y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) *
                       5 * factor, a_min=0, a_max=255)
    birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview


def draw_lidar_box3d_on_image(img, boxes3d, scores, gt_boxes3d=np.array([]),
                              color=(255, 255, 0), gt_color=(255, 0, 255), thickness=1):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = img.copy()
    projections = lidar_box3d_to_camera_box(boxes3d, cal_projection=True)
    gt_projections = lidar_box3d_to_camera_box(gt_boxes3d, cal_projection=True)

    # draw projections
    for qs in projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

    # draw gt projections
    for qs in gt_projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

    return img.astype(np.uint8)


def draw_lidar_box3d_on_birdview(birdview, boxes3d, scores, gt_boxes3d=np.array([]),
                                 color=(255, 255, 0), gt_color=(255, 0, 255), thickness=1, factor=1):
    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = birdview.copy()
    corner_boxes3d = center_to_corner_box3d(boxes3d)
    corner_gt_boxes3d = center_to_corner_box3d(gt_boxes3d)
    # draw gt
    for box in corner_gt_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 gt_color, thickness, cv2.LINE_AA)

    # draw detections
    for box in corner_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 color, thickness, cv2.LINE_AA)

    return img.astype(np.uint8)


def label_to_gt_box3d(labels, cls='Car', coordinate='camera'):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar'
    # Output:
    #   (N, N', 7)
    boxes3d = []
    if cls == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls == 'Pedestrian':
        acc_cls = ['Pedestrian']
    else:
        acc_cls = ['Cyclist']

    for label in labels:
        boxes3d_a_label = []
        for line in label:
            ret = line.split()
            if ret[0] in acc_cls:
                h, w, l, x, y, z, r = [float(i) for i in ret[-7:]]
                box3d = np.array([x, y, z, h, w, l, r])
                boxes3d_a_label.append(box3d)
        if coordinate == 'lidar':
            boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label))

        boxes3d.append(np.array(boxes3d_a_label).reshape(-1, 7))
    return boxes3d


# def cal_iou2d(box1, box2):
#     # Input:
#     #   box1/2: x, y, w, l, r
#     # Output:
#     #   iou
#     x1, y1, w1, l1, r1 = box1
#     x2, y2, w2, l2, r2 = box2
#     c1 = shapely.geometry.box(-w1/2.0, -l1/2.0, w1/2.0, l1/2.0)
#     c2 = shapely.geometry.box(-w2/2.0, -l2/2.0, w2/2.0, l2/2.0)
#
#     c1 = shapely.affinity.rotate(c1, r1, use_radians=True)
#     c2 = shapely.affinity.rotate(c2, r2, use_radians=True)
#
#     c1 = shapely.affinity.translate(c1, x1, y1)
#     c2 = shapely.affinity.translate(c1, x2, y2)
#
#     intersect = c1.intersection(c2)
#
#     return intersect.area/(c1.area + c2.area - intersect.area)
#
#
#
# def cal_iou3d(box1, box2):
#     # Input:
#     #   box1/2: x, y, z, h, w, l, r
#     # Output:
#     #   iou
#     def cal_z_intersect(cz1, h1, cz2, h2):
#         b1z1, b1z2 = cz1 - h1/2, cz1 + h1/2
#         b2z1, b2z2 = cz2 - h2/2, cz2 + h2/2
#         if b1z1 > b2z2 or b2z1 > b1z2:
#             return 0
#         elif b2z1 <= b1z1 <= b2z2:
#             if b1z2 <= b2z2:
#                 return h1/h2
#             else:
#                 return (b2z2-b1z1)/(b1z2-b2z1)
#         elif b1z1 < b2z1 < b1z2:
#             if b2z2 <= b1z2:
#                 return h2/h1
#             else:
#                 return (b1z2-b2z1)/(b2z2-b1z1)
#
#     x1, y1, z1, h1, w1, l1, r1 = box1
#     x2, y2, z2, h2, w2, l2, r2 = box2
#     c1 = shapely.geometry.box(-w1/2.0, -l1/2.0, w1/2.0, l1/2.0)
#     c2 = shapely.geometry.box(-w2/2.0, -l2/2.0, w2/2.0, l2/2.0)
#
#     c1 = shapely.affinity.rotate(c1, r1, use_radians=True)
#     c2 = shapely.affinity.rotate(c2, r2, use_radians=True)
#
#     c1 = shapely.affinity.translate(c1, x1, y1)
#     c2 = shapely.affinity.translate(c1, x2, y2)
#
#     z_intersect = cal_z_intersect(z1, h1, z2, h2)
#
#     intersect = c1.intersection(c2)
#
#     return intersect.area*z_intersect/(c1.area*h1 + c2.area*h2 - intersect.area*z_intersect)
#
#
#
# @jit
# def cal_box3d_iou(boxes3d, gt_boxes3d, cal_3d=False):
#     # Inputs:
#     #   boxes3d: (N1, 7) x,y,z,h,w,l,r
#     #   gt_boxed3d: (N2, 7) x,y,z,h,w,l,r
#     # Outputs:
#     #   iou: (N1, N2)
#
#     N1, N2 = len(boxes3d), len(gt_boxes3d)
#     output = np.zeros((N1, N2))
#     for idx in range(N1):
#         for idy in range(N2):
#             if cal_3d:
#                 output[idx, idy] = cal_iou3d(boxes3d[idx], gt_boxes3d[idy])
#             else:
#                 output[idx, idy] = cal_iou2d(boxes3d[idx, [0,1,4,5,6]], gt_boxes3d[idy, [0,1,4,5,6]])
#
#     return output
#
#
#
# @jit
# def cal_box2d_iou(boxes2d, gt_boxes2d):
#     # Inputs:
#     #   boxes2d: (N1, 5) x,y,w,l,r
#     #   gt_boxes2d: (N2, 5) x,y,w,l,r
#     # Outputs:
#     #   iou: (N1, N2)
#
#     N1, N2 = len(boxes2d), len(gt_boxes2d)
#     output = np.zeros((N1, N2))
#     for idx in range(N1):
#         for idy in range(N2):
#             output[idx, idy] = cal_iou2d(boxes2d[idx], gt_boxes2d[idy])
#
#     return output

#WZN: new
def lidar_in_cam_to_voxel(ptsCam):
    # from camera frame to voxel coordinates
    ptsVoxel = np.zeros_like(ptsCam)
    if len(ptsCam.shape)==2:
        ptsVoxel[:,0] = (ptsCam[:,0]-cfg.Y_MIN)/cfg.VOXEL_Y_SIZE
        ptsVoxel[:,2] = (ptsCam[:,1]-cfg.Z_MIN)/cfg.VOXEL_Z_SIZE
        ptsVoxel[:,1] = (ptsCam[:,2]-cfg.X_MIN)/cfg.VOXEL_X_SIZE
    elif len(ptsCam.shape)==3: #(Nx3x8 corners)
        ptsVoxel[:,0,:] = (ptsCam[:,0,:]-cfg.Y_MIN)/cfg.VOXEL_Y_SIZE
        ptsVoxel[:,2,:] = (ptsCam[:,1,:]-cfg.Z_MIN)/cfg.VOXEL_Z_SIZE
        ptsVoxel[:,1,:] = (ptsCam[:,2,:]-cfg.X_MIN)/cfg.VOXEL_X_SIZE    
    else:
        assert False, 'wrong input shape'
    return ptsVoxel

#WZN: new
def bbox_transform_voxel(ex_rois_3d, gt_rois_3d):
    assert ex_rois_3d.shape[1]==7,'wrong anchor dimension, should contain angle'
    assert gt_rois_3d.shape[1]==7,'wrong ground truth dimension, should contain angle'

    ex_rois_3d_diag = np.sqrt(ex_rois_3d[:, 3]**2 + ex_rois_3d[:, 4]**2)
    #WZN: 3D from first [x, y, z, l, w, h, theta] to second
    # x, y, z, l, w, h, theta
    ex_ctr_x = ex_rois_3d[:, 0]
    ex_ctr_y = ex_rois_3d[:, 1]
    ex_ctr_z = ex_rois_3d[:, 2]
    ex_lengths = ex_rois_3d[:, 3]
    ex_widths = ex_rois_3d[:, 4]
    ex_heights = ex_rois_3d[:, 5]

    gt_ctr_x = gt_rois_3d[:, 0]
    gt_ctr_y = gt_rois_3d[:, 1]
    gt_ctr_z = gt_rois_3d[:, 2]
    gt_lengths = gt_rois_3d[:, 3]
    gt_widths = gt_rois_3d[:, 4]
    gt_heights = gt_rois_3d[:, 5]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_rois_3d_diag
    targets_dy = (gt_ctr_y - ex_ctr_y) / cfg.ANCHOR_H
    targets_dz = (gt_ctr_z - ex_ctr_z) / ex_rois_3d_diag
    targets_dl = np.log(gt_lengths / ex_lengths)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    #print 'gt_angles: ', gt_rois_3d[:,6]
    #print 'ex_angles: ', ex_rois_3d[:,6] 
    targets_dtheta = gt_rois_3d[:,6] - ex_rois_3d[:,6] 
    # transform all targets to [-pi,pi]
    targets_dtheta[targets_dtheta>=np.pi] -= np.pi
    targets_dtheta[targets_dtheta<=-np.pi] += np.pi
    # WZN: we don't care about orientation, transform all targets to [-pi/2,pi/2]
    targets_dtheta[targets_dtheta>=(np.pi/2)] -= np.pi
    targets_dtheta[targets_dtheta<=-(np.pi/2)] += np.pi

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dz, targets_dl, targets_dw, targets_dh, cfg.ANGLE_WEIGHT*targets_dtheta)).transpose()
    return targets

def bbox_transform_inv_voxel(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    assert boxes.shape[1]==7,'wrong box dimension, should contain angle'
    assert deltas.shape[1]==7,'wrong delta dimension, should contain angle'

    boxes = boxes.astype(deltas.dtype, copy=False)

    lengths = boxes[:, 3]
    widths = boxes[:, 4]
    heights = boxes[:, 5]
    ctr_x = boxes[:, 0]
    ctr_y = boxes[:, 1]
    ctr_z = boxes[:, 2]

    rois_3d_diag = np.sqrt(lengths**2 + widths**2)

    dx = deltas[:, 0] 
    dy = deltas[:, 1]
    dz = deltas[:, 2]
    dl = deltas[:, 3]
    dw = deltas[:, 4]
    dh = deltas[:, 5]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    '''
    pred_ctr_x = dx * lengths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * widths[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_ctr_z = dz * heights[:, np.newaxis] + ctr_z[:, np.newaxis]
    pred_l = np.exp(dl) * lengths[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    '''
    # l
    pred_boxes[:, 3] = np.exp(dl) * lengths
    # w
    pred_boxes[:, 4] = np.exp(dw) * widths
    # h
    pred_boxes[:, 5] = np.exp(dh) * heights
    # x
    pred_boxes[:, 0] = dx * rois_3d_diag + ctr_x
    # y
    pred_boxes[:, 1] = dy * heights + ctr_y
    # z
    pred_boxes[:, 2] = dz * rois_3d_diag + ctr_z
    #theta
    pred_boxes[:, 6] = boxes[:, 6] + deltas[:, 6]/cfg.ANGLE_WEIGHT
    return pred_boxes


if __name__ == '__main__':
    pass