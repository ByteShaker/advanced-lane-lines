import numpy as np
import cv2
import matplotlib.pyplot as plt

import glob
import pickle

def perform_inital_position(img):
    imshape = img.shape
    initial_position = np.array([[(0,imshape[0]),
                                  (int(imshape[1]*1/3), int(imshape[0]*1/2)),
                                  (int(imshape[1]*2/3), int(imshape[0]*1/2)),
                                  (imshape[1],imshape[0])]], dtype=np.int32)
    return initial_position

def position_select(img, position_select=None, ignore_mask_color=255):
    position_select_binary = np.zeros_like(img, dtype=np.uint8)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (ignore_mask_color,) * channel_count
    else:
        ignore_mask_color = ignore_mask_color

    if position_select==None:
        position_select = perform_inital_position(img)

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(position_select_binary, position_select, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    position_select_binary = cv2.bitwise_and(img, position_select_binary)

    return position_select_binary

if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/test1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    position_select_binary = position_select(image, 255)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(position_select_binary, cmap='gray')
    ax2.set_title('Thresholded S', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()