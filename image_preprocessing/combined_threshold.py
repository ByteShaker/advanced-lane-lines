import numpy as np
import cv2
import matplotlib.pyplot as plt

import image_preprocessing.image_gradient as img_gradient
import image_preprocessing.image_color as img_color

if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/test1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    one_color_channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    #s_binary = img_color.hls_select(image, 's', (100, 255))

    gradx = img_gradient.abs_sobel_thresh(s, 'x', 9, (5, 200))
    grady = img_gradient.abs_sobel_thresh(s, 'y', 9, (5, 200))
    mag_binary= img_gradient.mag_thresh(s, 9, (30,200))
    dir_binary = img_gradient.dir_threshold(s, 9, (0.7, 1.2))



    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((mag_binary == 1) & (dir_binary == 1))] = 1

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(combined, cmap='gray')
    ax2.set_title('Thresholded Gradient', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()