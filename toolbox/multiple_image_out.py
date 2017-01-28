import cv2
import numpy as np

import logging

def create_blank(height, width, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image

def add_image_at_position(base_img, add_img, position_percentage=(0,1,0,1)):
    # If Image On_Color_Image Convert to 3 Channels
    add_img_shape = len(add_img)
    #print(add_img)
    if add_img_shape == 2:
        add_img = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    base_img_shape = base_img.shape
    position_px = (int(position_percentage[0]*base_img_shape[0]),
                   int(position_percentage[1]*base_img_shape[0]),
                   int(position_percentage[2]*base_img_shape[1]),
                   int(position_percentage[3]*base_img_shape[1]))
    new_img_shape = ((position_px[3]-position_px[2]), (position_px[1]-position_px[0]))
    new_img = cv2.resize(add_img, new_img_shape)
    base_img[position_px[0]:position_px[1], position_px[2]:position_px[3]] = new_img

    return base_img

def two_images(img1, img2):
    img_shape = img1.shape
    new_image = create_blank(img_shape[0], img_shape[1] * 2)

    new_image = add_image_at_position(new_image, img1, (0, 1, 0, .5))
    new_image = add_image_at_position(new_image, img2, (0, 1, .5, 1))

    return new_image

def image_cluster(img_list=[], img_text=[], new_img_shape=None, cluster_shape=None):
    if cluster_shape == None:
        val_col = int(np.ceil(np.sqrt(len(img_list))))
        val_row = int(np.ceil(len(img_list)/val_col))
        cluster_shape = (val_row, val_col)
    if new_img_shape == None:
        new_img_shape = img_list[0].shape
        new_img_shape = tuple([int(new_img_shape[i]*cluster_shape[i]/max(cluster_shape)) for i in range(len(cluster_shape))])

    size_of_cluster = (cluster_shape[0] * cluster_shape[1])
    if size_of_cluster < len(img_list):
        logging.info('Cluster to small for all Images')

    new_image = create_blank(new_img_shape[0], new_img_shape[1])
    cluster_index = 0
    for row in range(cluster_shape[0]):
        for col in range(cluster_shape[1]):
            new_image = add_image_at_position(new_image, img_list[cluster_index], (row/cluster_shape[0], (row+1)/cluster_shape[0], col/cluster_shape[1], (col+1)/cluster_shape[1]))

            text_position = (col*int(new_img_shape[1]/cluster_shape[1])+100, row*int(new_img_shape[0]/cluster_shape[0])+100)
            cv2.putText(new_image, img_text[cluster_index], text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

            cluster_index += 1
            if (cluster_index >= len(img_list)):
                break
        if (cluster_index >= len(img_list)):
            break

    return new_image

if __name__ == "__main__":
    # Read in the image

    img1 = cv2.imread('../test_images/straight_lines1.jpg')
    img2 = cv2.imread('../test_images/straight_lines2.jpg')

    #new_image = two_images(img1,img2)

    new_image = image_cluster([img1, img2], img_text=['Image 1','Image 2'])

    #new_image = cv2.resize(new_image, (256 * 5, 72 * 5))
    #new_image = cv2.resize(new_image, None, fx=.5, fy=.5, interpolation=cv2.INTER_CUBIC)

    cv2.imshow('Two Images', new_image)
    cv2.waitKey(0)




