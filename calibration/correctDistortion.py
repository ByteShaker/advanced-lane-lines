import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import calibration.calibrateCamera as calibrateCamera

def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def correct_distortion(img, mtx=None, dist=None, camera_calibration_values='../calibration/camera_calibration_values.pickle'):
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

    img = cv2.imread('../test_images/straight_lines2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    undistorted = correct_distortion(img)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()

    undistorted = cv2.cvtColor(undistorted, cv2.COLOR_RGB2BGR)
    cv2.imwrite('../output_images/straight_lines2.jpg', undistorted)