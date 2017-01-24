import numpy as np
import cv2
import matplotlib.pyplot as plt

import glob
import pickle

def perform_initial_sourcepoints():
    src = np.float32(
            [[603, 444],
             [220, 705],
             [1083, 705],
             [675, 444]])

    dst = np.float32(
            [[400, 20],
             [400, 700],
             [880, 700],
             [880, 20]])

    return (src, dst)

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/straight_lines2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    src, dst = perform_initial_sourcepoints()
    warped_image = warper(image, src, dst)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(warped_image, cmap='gray')
    ax2.set_title('Thresholded S', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.show()