import numpy as np
import cv2
import matplotlib.pyplot as plt

import glob
import pickle

import toolbox.multiple_plots_out as mpo

def perform_initial_sourcepoints():
    src = np.float32(
            [[602, 444],
             [220, 705],
             [1083, 705],
             [678, 444]])

    dst = np.float32(
            [[540, 0],
             [540, 720],
             [740, 720],
             [740, 0]])

    return (src, dst)

def warper(img, src, dst, direction='forward'):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    if direction == 'forward':
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    elif direction == 'backward':
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/test6.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    src, dst = perform_initial_sourcepoints()
    warped_image = warper(image, src, dst)

    mpo.plot_cluster([image, warped_image], img_text=['Original Image', 'Warped Image'])