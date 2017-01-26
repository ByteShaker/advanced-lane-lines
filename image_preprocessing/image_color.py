import numpy as np
import cv2
import matplotlib.pyplot as plt

import glob
import pickle

def layer_select(img, selector = 's', thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the selected channel
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    one_color_channel = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if selector == 'h':
        retval, selected_layer_binary = cv2.threshold(h.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
    elif selector == 'l':
        retval, selected_layer_binary = cv2.threshold(l.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)
    elif selector == 's':
        retval, selected_layer_binary = cv2.threshold(s.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)

    elif selector == 'gray':
        retval, selected_layer_binary = cv2.threshold(one_color_channel.astype('uint8'), thresh[0], thresh[1], cv2.THRESH_BINARY)

    else:
        print('Selected layer not in HLS Color Space')
    # 3) Return a binary image of threshold result
    return selected_layer_binary

if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/straight_lines1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    s_binary = hls_select(image, 's', (220,255))

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(s_binary, cmap='gray')
    ax2.set_title('Thresholded S', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()