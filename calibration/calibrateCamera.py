import numpy as np
import cv2
import matplotlib.pyplot as plt

import glob
import pickle

def finding_corners(fname, nx_ny_list=[(9,5)], verbose = False):
    """
    >>> ret, instance_objpoints, instance_imgpoints = finding_corners('./camera_cal/calibration1.jpg', nx_ny_list=[(9,5)])
    >>> len(instance_objpoints[0])
    45
    """
    instance_objpoints = []
    instance_imgpoints = []

    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret = False
    for i in range(len(nx_ny_list)):
        nx, ny = nx_ny_list[i]
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret==True:
            break

    # If found, draw corners
    if ret == True:
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        instance_imgpoints.append(corners)
        instance_objpoints.append(objp)

        if verbose == True:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            plt.imshow(img)
            plt.show()

    return ret, instance_objpoints, instance_imgpoints

def define_points(images, nx_ny_list):
    """
    :param filepath:
    :param nx:
    :param ny:
    :return:

    >>> images = glob.glob('./camera_cal/calibration*')
    >>> objpoints, imgpoints = define_points(images, nx_ny_list = [(9,6),(9,5),(7,6)])
    >>> len(objpoints)
    19

    """
    objpoints = []
    imgpoints = []

    #get imagepoints for every single cal image
    for img in images:
        ret, instance_objpoints, instance_imgpoints = finding_corners(img, nx_ny_list)

        if ret == True:
            objpoints.extend(instance_objpoints)
            imgpoints.extend(instance_imgpoints)

    return objpoints, imgpoints

def cal_mtx_dist(filepath='./camera_cal/calibration*'):
    """
    :param filepath:
    :return:

    >>> rms, mtx, dist = cal_mtx_dist()
    >>> rms < 2.
    True
    """
    # get list of all calibration images
    images = glob.glob(filepath)

    nx_ny_list = [(9, 6), (9, 5), (7, 6)]
    objpoints, imgpoints = define_points(images, nx_ny_list)

    reference_img = cv2.imread(images[0], 0)
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, reference_img.shape[::-1], None, None)

    camera_calibration_values = {'rms': rms, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
    with open('camera_calibration_values.pickle', 'wb') as file:
        pickle.dump(camera_calibration_values, file, protocol=pickle.HIGHEST_PROTOCOL)

    return rms, mtx, dist

if __name__ == "__main__":
    import doctest
    doctest.testmod()