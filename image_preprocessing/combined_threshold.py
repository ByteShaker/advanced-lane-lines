import numpy as np
import cv2
import matplotlib.pyplot as plt

import calibration.correctDistortion as correctDistortion

import image_preprocessing.image_gradient as img_gradient
import image_preprocessing.image_color as img_color
import image_preprocessing.image_position as img_position

import perspective_transform.image_transform as img_transform

if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/straight_lines2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    correctDistortion.correct_distortion(image)

    one_color_channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    s_binary = img_color.hls_select(image, 's', (100, 255))

    gradx = img_gradient.abs_sobel_thresh(s, 'x', 9, (20, 255))
    grady = img_gradient.abs_sobel_thresh(s, 'y', 9, (20, 255))
    mag_binary= img_gradient.mag_thresh(s, 3, (30,255))
    dir_binary = img_gradient.dir_threshold(s, 31, (25*np.pi/180, 65*np.pi/180))

    position_binary = img_position.position_select(s)

    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((mag_binary == 1) & (dir_binary == 1)) & (position_binary == 1)] = 1
    #combined[((gradx == 1) & (grady == 1))] = 1

    src, dst = img_transform.perform_initial_sourcepoints()
    warped_combined = img_transform.warper(combined, src, dst)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped_combined, cmap='gray')
    ax2.set_title('Thresholded Gradient', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()