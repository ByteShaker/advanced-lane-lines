import numpy as np
import cv2
import matplotlib.pyplot as plt

import toolbox.multiple_plots_out as mpo
import toolbox.multiple_image_out as mio

import glob
import pickle

def brighten_img(image):
    # Image data
    maxIntensity = 255.0  # depends on dtype of image data

    # Parameters for manipulating image data
    phi = 1
    theta = 1

    # Increase intensity such that
    # dark pixels become much brighter,
    # bright pixels become slightly bright
    brightend_image = (maxIntensity / phi) * (image / (maxIntensity / theta)) ** 0.5
    brightend_image = np.array(brightend_image, dtype=np.uint8)

    #cv2.imshow('newImage0', newImage0)
    return brightend_image

def darken_img(image):
    '''
    Simple and fast image transforms to mimic:
     - brightness
     - contrast
     - erosion
     - dilation
    '''

    # Image data
    maxIntensity = 255.0  # depends on dtype of image data

    # Parameters for manipulating image data
    theta = 20
    phi = theta**8

    # Decrease intensity such that
    # dark pixels become much darker,
    # bright pixels become slightly dark
    darkend_image = (maxIntensity / phi) * ((image / (maxIntensity / theta)) ** 8)
    contrasted_image = np.array(darkend_image, dtype=np.uint8)

    #cv2.imshow('newImage1', newImage1)
    return contrasted_image

def add_contrast(img,clipLimit=3.0, tileGridSize=(5,5), verbose=False):
    #-----Converting image to LAB Color model-----------------------------------
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    cl = clahe.apply(l)
    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


    if verbose:
        contrast_image = mio.image_cluster([img, lab, l, a, b, cl, limg, final], img_text=['img', 'lab', 'l', 'a', 'b', 'cl', 'limg', 'final'])
        cv2.imshow('Contrast_LAB', contrast_image)

    return final


def layer_select(img, selector = 'hls_S', thresh=(0, 255)):

    if (selector == 'hls_H') | (selector == 'hls_L') | (selector == 'hls_S'):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the selected channel
        h = hls[:, :, 0]
        l = hls[:, :, 1]
        s = hls[:, :, 2]

        if selector == 'hls_H':
            retval, selected_layer_binary = cv2.threshold(h.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        elif selector == 'hls_L':
            retval, selected_layer_binary = cv2.threshold(l.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        elif selector == 'hls_S':
            retval, selected_layer_binary = cv2.threshold(s.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)

    elif (selector == 'gray'):
        if len(img.shape) >= 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        retval, selected_layer_binary = cv2.threshold(img.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)

    else:
        print('Selected layer not supported')
    # 3) Return a binary image of threshold result
    return selected_layer_binary

if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/straight_lines1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    s_binary = layer_select(image, 'hls_S', (220,255))

    mpo.plot_cluster([image, s_binary], img_text=['Original Image', 'Thresholded S'])

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    contrasted_image = add_contrast(image, verbose=True)
    cv2.imshow('contrasted_image', contrasted_image)
    cv2.waitKey(0)