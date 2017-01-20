import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import calibrateCamera


def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def correct_distortion(filepath='../test_images/test1.jpg', camera_calibration_values='./camera_calibration_values.pickle'):
    """
    :param filepath:
    :param camera_calibration_values:
    :return:

    """
    try:
        camera_calibration_values = open(camera_calibration_values, 'rb')
        camera_calibration_values = pickle.load(camera_calibration_values)
        mtx = camera_calibration_values['mtx']
        dist = camera_calibration_values['dist']
    except:
        # calibrate_camera
        rms, mtx, dist = calibrateCamera.cal_mtx_dist()

    # Read in the image
    img = cv2.imread(filepath)

    undistorted = cal_undistort(img, mtx, dist)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()

if __name__ == "__main__":
    correct_distortion('../camera_cal/calibration1.jpg')