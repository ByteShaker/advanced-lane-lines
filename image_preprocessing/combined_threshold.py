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

def combined_thresholds_1(image):

    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    one_color_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    combo = combine_edgeGray_edgeS(one_color_channel, s)
    #combo = combine_img_pos(combo)

    #mag_binary_S = img_gradient.mag_thresh(s, 9, (30, 255))
    #mag_binary_Gray = img_gradient.mag_thresh(one_color_channel, 9, (30, 255))
    #dir_binary = img_gradient.dir_threshold(combo, 15, (35 * np.pi / 180, 65 * np.pi / 180))

    #combo1 = cv2.bitwise_and(dir_binary, mag_binary_S)
    #combo2 = cv2.bitwise_and(dir_binary, mag_binary_Gray)

    #combo3 = bitwise_OR_images(combo1, combo2)

    return combo

def combined_thresholds_complete(image, verbose=False):
    warped_image_cont = img_color.add_contrast(image, clipLimit=4.0, tileGridSize=(5,5))
    warped_image_blur = cv2.medianBlur(image, 3)
    warped_image_cont_blur = cv2.medianBlur(warped_image_cont, 3)

    img_cont = mio.image_cluster([image, warped_image_cont, warped_image_blur, warped_image_cont_blur])
    cv2.imshow('Image Cont', img_cont)

    hls = cv2.cvtColor(warped_image_cont, cv2.COLOR_BGR2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    one_color_channel = cv2.cvtColor(warped_image_cont_blur, cv2.COLOR_BGR2GRAY)

    #combo = combine_edgeGray_edgeS(one_color_channel, s)

    color_binary_S = img_color.layer_select(s, 'gray', (100, 255))
    color_binary_Gray = img_color.layer_select(one_color_channel, 'gray', (180, 255))

    mag_binary_S = img_gradient.mag_thresh(s, 21, (30, 255))
    mag_binary_Gray = img_gradient.mag_thresh(one_color_channel, 21, (30, 255))

    dir_binary_S = img_gradient.dir_threshold(mag_binary_S, 9, (0 * np.pi / 180, 55 * np.pi / 180))
    dir_binary_Gray = img_gradient.dir_threshold(mag_binary_Gray, 9, (0 * np.pi / 180, 45 * np.pi / 180))

    #combo_S_dir_mag = cv2.bitwise_and(dir_binary_S, mag_binary_S)
    #combo_Gray_dir_mag = cv2.bitwise_and(dir_binary_Gray, mag_binary_Gray)

    combo_S_dir_mag_blur = cv2.blur(dir_binary_S, (11, 11))
    combo_Gray_dir_mag_blur = cv2.blur(dir_binary_Gray, (11, 11))

    combo_S = cv2.bitwise_and(combo_S_dir_mag_blur, color_binary_S)
    combo_Gray = cv2.bitwise_and(combo_Gray_dir_mag_blur, color_binary_Gray)

    combo_complete = cv2.bitwise_or(combo_S, combo_Gray)

    if verbose:
        new_image = mio.image_cluster(
            [one_color_channel, s, color_binary_Gray, color_binary_S, mag_binary_Gray, mag_binary_S, dir_binary_Gray, dir_binary_S, combo_Gray_dir_mag_blur, combo_S_dir_mag_blur, combo_Gray, combo_S, combo_complete],
            img_text=['one_color_channel', 's', 'color_binary_Gray', 'color_binary_S', 'mag_binary_Gray', 'mag_binary_S', 'dir_binary_Gray', 'dir_binary_S', 'combo_Gray_dir_mag_blur', 'combo_S_dir_mag_blur', 'combo_Gray', 'combo_S', 'combo_complete'],
            new_img_shape=(720, 1280), cluster_shape=(4, 4))

        cv2.imshow('Combination', new_image)
        #cv2.waitKey(0)

    return combo_complete

if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/test1.jpg')
    mtx, dist, image = correctDistortion.correct_distortion(image)

    one_color_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    hlv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hlv_H = hlv[:, :, 0]
    hlv_S = hlv[:, :, 1]
    hlv_V = hlv[:, :, 2]

    combo_edgeGray_edgeS = combine_edgeGray_edgeS(image, s)
    combo = combine_img_pos(combo_edgeGray_edgeS)

    l_binary = img_color.layer_select(image, 'l', (200, 255))
    s_binary = img_color.layer_select(image, 's', (100, 255))
    gray_binary = img_color.layer_select(image, 'gray', (150, 225))

    mag_binary_S = img_gradient.mag_thresh(s, 3, (30, 255))
    mag_binary_Gray = img_gradient.mag_thresh(one_color_channel, 3, (30, 255))
    dir_binary = img_gradient.dir_threshold(combo, 15, (35 * np.pi / 180, 65 * np.pi / 180))

    retval, hlv_V_binary = cv2.threshold(hlv_V.astype('uint8'), 150, 255, cv2.THRESH_BINARY)

    combo1 = bitwise_AND_images(dir_binary, mag_binary_S)
    combo2 = bitwise_AND_images(hlv_V_binary, dir_binary)
    combo3 = bitwise_AND_images(dir_binary, mag_binary_Gray)

    combo4 = bitwise_OR_images(combo1, combo3)


    gradx = img_gradient.abs_sobel_thresh(one_color_channel, 'x', 9, (20, 255))
    grady = img_gradient.abs_sobel_thresh(one_color_channel, 'y', 9, (20, 255))


    position_binary = img_position.position_select(s)

    combined = np.zeros_like(dir_binary)

    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    #combined[((gradx == 1) & (grady == 1))] = 1
    #combined[((mag_binary == 1) & (dir_binary == 1)) & (position_binary == 1)] = 1
    #combined[(((s_binary >= 200) | (l_binary >= 200)) | ((mag_binary == 1) & (dir_binary == 1))) & (position_binary == 1)] = 1
    #combined[(((s_binary == 255) | (l_binary == 255))) & (position_binary == 1)] = 1

    src, dst = img_transform.perform_initial_sourcepoints()
    warped_combined = img_transform.warper(combo4, src, dst)

    warped_combined1 = img_transform.warper(combo_edgeGray_edgeS, src, dst)

    dir_binary = img_gradient.dir_threshold(warped_combined1, 15, (0 * np.pi / 180, 35 * np.pi / 180))

    new_image = mio.image_cluster([one_color_channel, s, cv2.Canny(one_color_channel, 120, 180), cv2.Canny(s, 120, 180), combo_edgeGray_edgeS, warped_combined1, dir_binary], new_img_shape=(1200, 2260), cluster_shape=(4,2))

    cv2.imshow('Combined Tresholds Overview', new_image)
    cv2.waitKey(0)

    histogram = np.sum(warped_combined[int(warped_combined.shape[0] * 2/3):, :], axis=0)
    histogram_df = pd.DataFrame(histogram)

    Series = histogram_df[0].rolling(window=200, axis=0, center=True, win_type='gaussian').mean(std=10.0)

    histogram_df['Peak_Filter'] = Series
    mpo.plot_cluster([Series])