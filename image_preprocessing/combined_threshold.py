import numpy as np
import cv2
import matplotlib.pyplot as plt

import pandas as pd

import calibration.correctDistortion as correctDistortion

import image_preprocessing.image_gradient as img_gradient
import image_preprocessing.image_color as img_color
import image_preprocessing.image_position as img_position

import perspective_transform.image_transform as img_transform

import toolbox.multiple_image_out as mio
import toolbox.multiple_plots_out as mpo

#Todo: Ersetzt durch cv2.bitwise_and
def bitwise_AND_images(img1, img2, return_value=255):
    combo = np.zeros_like(img1, dtype=np.uint8)
    combo[((img1 >= 1) & (img2 >= 1))] = return_value
    return combo

#Todo: Ersetzt durch cv2.bitwise_or
def bitwise_OR_images(img1, img2, return_value=255):
    combo = np.zeros_like(img1, dtype=np.uint8)
    combo[((img1 >= 1) | (img2 >= 1))] = return_value
    return combo

def combine_img_pos(image):
    combo_img_pos = img_position.position_select(image)

    return combo_img_pos

def combine_mag_dir_pos(image, type = 'hls_S'):

    if type == 'hls_S':
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        h = hls[:, :, 0]
        l = hls[:, :, 1]
        s = hls[:, :, 2]

    #edges = cv2.Canny(s, 50, 150)

    mag_binary = img_gradient.mag_thresh(s, 3, (15, 255))
    dir_binary = img_gradient.dir_threshold(s, 3, (0 * np.pi / 180, 65 * np.pi / 180))
    position_binary = img_position.position_select(s)

    combined = np.zeros_like(dir_binary)
    combined[((mag_binary == 1) & (dir_binary == 1)) & (position_binary == 1)] = 1

    return combined

def combine_edgeGray_edgeS(one_color_channel, s):

    edges_Gray = cv2.Canny(one_color_channel, 60, 120)
    edges_S = cv2.Canny(s, 60, 120)

    combo = cv2.bitwise_or(edges_S, edges_Gray)

    return combo

def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)

def gamma_equalize(img, repetition=1, correction=2, thresh=[50,255]):
    for i in range(repetition):
        #cv2.threshold(img.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
        img = cv2.equalizeHist(img)
        img = gamma_correction(img, correction)

    return img

def combined_thresholds_complete(image, verbose=False):
    #warped_image_cont = img_color.add_contrast(image, clipLimit=4.0, tileGridSize=(5,5))
    #warped_image_blur = cv2.medianBlur(image, 3)
    #warped_image_cont_blur = cv2.medianBlur(warped_image_cont, 3)

    #img_cont = mio.image_cluster([image, warped_image_cont, warped_image_blur, warped_image_cont_blur])
    #cv2.imshow('Image Cont', img_cont)


    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_h, hsv_s, hsv_v = cv2.split(hsv)

    hsv_gamma_equal_dark = gamma_equalize(hsv_v, 5, 5)
    #hsv_gamma_equal_light = gamma_equalize(hsv_v, 3, .8)

    hsv = cv2.merge((hsv_h, hsv_s, hsv_gamma_equal_dark))
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    mag_binary_V = img_gradient.mag_thresh(hsv_gamma_equal_dark, 7, (40, 255))
    #color_binary_V = img_color.layer_select(hsv_gamma_equal_dark, 'gray', (150, 255))
    dir_binary_V = img_gradient.dir_threshold(hsv_gamma_equal_dark, 21, (0 * np.pi / 180, 55 * np.pi / 180))
    combo_complete_dark = cv2.bitwise_and(dir_binary_V, mag_binary_V)

    #combo_complete_dark = cv2.bitwise_or(color_binary_V, combo_complete_dark)

    process_2_binary_text = ["Original ->","HSV_V_Gamma ->","Original_Gamma ->","Mag_Gradient ->","Dir_Gradient ->","Combined_binary"]
    process_2_binary = mio.image_cluster([image, hsv_gamma_equal_dark, hsv, mag_binary_V, dir_binary_V, combo_complete_dark], process_2_binary_text)
    cv2.imshow('hurra', process_2_binary)


    #hls = cv2.cvtColor(hsv, cv2.COLOR_BGR2HLS)
    #h, l, s = cv2.split(hls)

    #s = cv2.equalizeHist(s)
    #s = gamma_correction(s, 10)

    # one_color_channel = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    #
    # #combo = combine_edgeGray_edgeS(one_color_channel, s)
    #
    # color_binary_S = img_color.layer_select(s, 'gray', (60, 255))
    # color_binary_Gray = img_color.layer_select(one_color_channel, 'gray', (80, 255))
    #
    # mag_binary_S = img_gradient.mag_thresh(s, 9, (60, 255))
    # mag_binary_Gray = img_gradient.mag_thresh(one_color_channel, 9, (80, 255))
    #
    # dir_binary_S = img_gradient.dir_threshold(s, 5, (0 * np.pi / 180, 55 * np.pi / 180))
    # dir_binary_Gray = img_gradient.dir_threshold(one_color_channel, 5, (0 * np.pi / 180, 45 * np.pi / 180))
    #
    # combo_S_dir_mag = cv2.bitwise_and(dir_binary_S, mag_binary_S)
    # combo_Gray_dir_mag = cv2.bitwise_and(dir_binary_Gray, mag_binary_Gray)
    #
    # combo_S_dir_mag_blur = cv2.blur(dir_binary_S, (11, 11))
    # combo_Gray_dir_mag_blur = cv2.blur(dir_binary_Gray, (11, 11))
    #
    # combo_S = cv2.bitwise_and(combo_S_dir_mag_blur, color_binary_S)
    # combo_Gray = cv2.bitwise_and(combo_Gray_dir_mag_blur, color_binary_Gray)
    #
    # #gray_blurred = cv2.blur(color_binary_Gray, (21, 21))
    # #s_blurred = cv2.blur(color_binary_S, (21, 21))
    # #dir_binary = cv2.bitwise_or(dir_binary_S, dir_binary_Gray)
    # #test_combo = cv2.bitwise_and(cv2.bitwise_or(s_blurred, dir_binary), cv2.bitwise_or(gray_blurred, dir_binary))
    #
    # combo_complete = cv2.bitwise_or(combo_S, combo_Gray)
    # combo_complete_2 = cv2.bitwise_or(combo_S_dir_mag, combo_Gray_dir_mag)
    #
    # if verbose:
    #     new_image = mio.image_cluster(
    #         [one_color_channel, s, color_binary_Gray, color_binary_S, mag_binary_Gray, mag_binary_S, dir_binary_Gray, dir_binary_S, combo_Gray_dir_mag_blur, combo_S_dir_mag_blur, combo_Gray, combo_S, combo_complete, hsv, combo_complete_2],
    #         img_text=['one_color_channel', 's', 'color_binary_Gray', 'color_binary_S', 'mag_binary_Gray', 'mag_binary_S', 'dir_binary_Gray', 'dir_binary_S', 'combo_Gray_dir_mag_blur', 'combo_S_dir_mag_blur', 'combo_Gray', 'combo_S', 'combo_complete'],
    #         new_img_shape=(720, 1280), cluster_shape=(4, 4))
    #
    #     cv2.imshow('Combination', new_image)
    #     #cv2.waitKey(0)

    return combo_complete_dark

if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/test6.jpg')
    mtx, dist, image = correctDistortion.correct_distortion(image)

    src, dst = img_transform.calc_new_sourcepoints()
    warped_image = img_transform.warper(image, src, dst)

    combined_thresholds_complete(warped_image)

    cv2.waitKey(0)