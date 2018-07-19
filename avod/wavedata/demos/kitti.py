#!/usr/bin/env python3

import os
import random as random

import matplotlib.pyplot as plt

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.visualization import vis_utils

ROOTDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def main():
    # Start of the Kitti demo code
    print('=== Python Kitti Wrapper Demo ===')

    # Setting Paths
    data_set = 'training'
    cam = 2

    root_dir = os.path.expanduser('~') + '/Kitti/object/'

    image_dir = os.path.join(root_dir, data_set) + '/image_' + str(cam)
    label_dir = os.path.join(root_dir, data_set) + '/label_' + str(cam)
    calib_dir = os.path.join(root_dir, data_set) + '/calib'

    img_idx = int(random.random()*100)
    print('img_idx', img_idx)

    # Run Visualization Function
    f, ax1, ax2 = vis_utils.visualization(image_dir, img_idx)

    # Run the main loop to run throughout the images
    frame_calibration_info = calib_utils.read_calibration(calib_dir, img_idx)

    p = frame_calibration_info.p2

    # Load labels
    objects = obj_utils.read_labels(label_dir, img_idx)

    # For all annotated objects
    for obj in objects:

        # Draw 2D and 3D boxes
        vis_utils.draw_box_2d(ax1, obj)
        vis_utils.draw_box_3d(ax2, obj, p)

    # Render results
    plt.draw()
    plt.show()


if __name__ == "__main__":
    main()
