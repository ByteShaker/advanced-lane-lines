import numpy as np
import cv2
import matplotlib.pyplot as plt

import toolbox.multiple_plots_out as mpo
import toolbox.multiple_image_out as mio

def convolutional_unsharpen():
    return 0

def abs_sobel_thresh(one_color_channel_image, orient='x', sobel_kernel=3, grad_thresh=(0, 255)):
    # Sobel Kernel needs to be odd
    if sobel_kernel % 2 == 0:
        sobel_kernel = sobel_kernel + 1
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(one_color_channel_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(one_color_channel_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        print('No Orientation for Sobel Operator')
    # Calculate Absolute Value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(abs_sobel) / 255  # Will use this to scale back to 8-bit scale
    abs_sobel = (abs_sobel / scale_factor).astype(np.uint8)  # rescaling to 8-bit
    # 6) Create a binary mask where mag thresholds are met
    grad_binary = np.zeros_like(abs_sobel)
    grad_binary[(abs_sobel >= grad_thresh[0]) & (abs_sobel <= grad_thresh[1])] = 1
    return grad_binary

def mag_thresh(one_color_channel_image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Sobel Kernel needs to be odd
    if sobel_kernel%2==0:
        sobel_kernel = sobel_kernel + 1
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(one_color_channel_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(one_color_channel_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    mag = np.sqrt(np.power(sobelx, 2.) + np.power(sobely, 2.))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(mag) / 255  # Will use this to scale back to 8-bit scale
    mag = (mag / scale_factor).astype(np.uint8)  # rescaling to 8-bit
    # 6) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(mag)
    # mag_binary_filled = np.zeros_like(mag)
    mag_binary[(mag >= mag_thresh[0]) & (mag <= mag_thresh[1])] = 255
    # for row in range(mag_binary.shape[0]):
    #     for col in range(mag_binary.shape[1]):
    #         if mag_binary[row][col] == 255:
    #             for row_fill_area in range(5):
    #                 row_filled = row + row_fill_area - 2
    #                 if (row_filled >= 0) & (row_filled < 720):
    #                     for col_fill_area in range(5):
    #                         col_filled = col + col_fill_area - 2
    #                         if (col_filled >= 0) & (col_filled < 1280):
    #                             mag_binary_filled[row_filled][col_filled] = 255

    return mag_binary

def dir_threshold(one_color_channel_image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Sobel Kernel needs to be odd
    if sobel_kernel % 2 == 0:
        sobel_kernel = sobel_kernel + 1
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(one_color_channel_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(one_color_channel_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    with np.errstate(divide='ignore', invalid='ignore'):
        direction = np.arctan(sobely/sobelx)
        direction = np.absolute(direction)

        dir_binary = np.zeros_like(direction, dtype=np.uint8)
        dir_binary[(direction > thresh[0]) & (direction < thresh[1])] = 255

    return dir_binary

if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/test5.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    one_color_channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gradx = abs_sobel_thresh(one_color_channel, 'x', 9, (5, 200))
    grady = abs_sobel_thresh(one_color_channel, 'y', 9, (5, 200))
    mag_binary= mag_thresh(one_color_channel, 9, (50,200))
    dir_binary = dir_threshold(one_color_channel, 9, (0.7, 1.2))

    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((mag_binary >= 1) & (dir_binary >= 1))] = 1

    mpo.plot_cluster([gradx, grady, mag_binary, dir_binary, combined])
    mpo.plot_cluster([image, combined], img_text=['Original Image', 'Thresholded Gradient'])