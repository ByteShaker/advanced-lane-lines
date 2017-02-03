import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

import calibration.calibrateCamera as calibrateCamera
import toolbox.multiple_image_out as mio

def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def correct_distortion(img, mtx=None, dist=None, camera_calibration_values='../calibration/camera_calibration_values.pickle', verbose=False):
    """
    :param filepath:
    :param camera_calibration_values:
    :return:

    """
    if (mtx == None) | (dist == None):
        try:
            camera_calibration_values = open(camera_calibration_values, 'rb')
            camera_calibration_values = pickle.load(camera_calibration_values)
            mtx = camera_calibration_values['mtx']
            dist = camera_calibration_values['dist']
        except:
            # calibrate_camera
            rms, mtx, dist = calibrateCamera.cal_mtx_dist()

    undistorted = cal_undistort(img, mtx, dist)

    return mtx, dist, undistorted



if __name__ == "__main__":
    # Read in the image
    img = cv2.imread('../test_images/test6.jpg')

    mtx, dist, undistorted = correct_distortion(img)

    result = mio.two_images(img, undistorted)

    cv2.putText(result, "Original", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)
    cv2.putText(result, "Undistorted", (img.shape[1]+100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 5)

    result = cv2.resize(result, None, fx=.5, fy=.5, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('Two Images', result)
    #cv2.waitKey(0)

    cv2.imwrite('../output_images/original_undistorted.jpg', result)