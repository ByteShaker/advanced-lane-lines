import numpy as np
import cv2
import matplotlib.pyplot as plt

import toolbox.multiple_plots_out as mpo
import toolbox.multiple_image_out as mio

import glob
import pickle


def perform_inital_position(img):
    imshape = img.shape
    initial_position = np.array([[(0,imshape[0]),
                                  (int(imshape[1]*1/3), int(imshape[0]*1/2)),
                                  (int(imshape[1]*2/3), int(imshape[0]*1/2)),
                                  (imshape[1],imshape[0])]], dtype=np.int32)
    return initial_position

def perform_image_area(img, area_percentage):
    imshape = img.shape
    img_area = np.array([[(0, imshape[0]),
                          (0, int(imshape[0] * (1-area_percentage))),
                          (imshape[1], int(imshape[0] * (1-area_percentage))),
                          (imshape[1], imshape[0])]], dtype=np.int32)
    return img_area

def calculate_line_shape_area(line_fit, shape_structure=[40,30,20], ymax=720):
    #shape_structure = [20,15,10]
    old_y = [0, ymax/2, ymax]
    old_x = [0,0,0]

    angle=[0,0,0] #  im Steigunsmaß
    new_y=[0,0,0]
    new_x=[0,0,0]
    new_line_fit = [0,0,0]

    for i in range(len(old_y)):
        angle[i] = 2 * line_fit[0] * old_y[i] + line_fit[1]
        if angle[i] == 0:
            new_y[i] = old_y[i]
            new_x[i] = shape_structure[i]
        elif angle[i] > 0:
            new_y[i] = shape_structure[i] / np.sqrt(1 + (1 / np.power(angle[i], 2)))
            new_x[i] = (-1 / angle[i]) * new_y[i]
        else:
            new_y[i] = -(shape_structure[i] / np.sqrt(1+(1/np.power(angle[i],2))))
            new_x[i] = (-1/angle[i]) * new_y[i]
        #Bisher nur delta
        new_y[i] = old_y[i] + new_y[i]
        old_x[i] = line_fit[0]*np.power(old_y[i],2) + line_fit[1]*old_y[i] + line_fit[2]
        new_x[i] = old_x[i] + new_x[i]

    new_line_fit = np.polyfit(new_y, new_x, 2)

    return new_line_fit

def perform_lane_position(img, left_lane_fit=[0,0,540], right_lane_fit=[0,0,740], area_percentage=1, lane_range_bottom=10, lane_range_change=10):
    left_lane_binary = np.zeros_like(img, dtype=np.uint8)
    right_lane_binary = np.zeros_like(img, dtype=np.uint8)
    area_binary = np.zeros_like(img, dtype=np.uint8)
    yvals = np.array(range(img.shape[0]))
    #lane_range = (lane_range_bottom + ((img.shape[0]-yvals)/ img.shape[0]) * lane_range_change)

    shape_structure_right = [40, 30, 20]
    shape_structure_left = [-40, -30, -20]

    #left_left_fitx = left_lane_fit[0] * yvals ** 2 + left_lane_fit[1] * yvals + left_lane_fit[2] - lane_range
    #left_right_fitx = left_lane_fit[0] * yvals ** 2 + left_lane_fit[1] * yvals + left_lane_fit[2] + lane_range

    left_left_fit = calculate_line_shape_area(left_lane_fit, shape_structure=shape_structure_left)
    left_left_fitx = left_left_fit[0] * yvals ** 2 + left_left_fit[1] * yvals + left_left_fit[2]
    left_right_fit = calculate_line_shape_area(left_lane_fit, shape_structure=shape_structure_right)
    left_right_fitx = left_right_fit[0] * yvals ** 2 + left_right_fit[1] * yvals + left_right_fit[2]

    diffs = [left_lane_fit[i] - left_left_fit[i] for i in range(3)]
    #print(diffs)
    delta_fitx = diffs[0] * yvals ** 2 + diffs[1] * yvals + diffs[2]
    #print(delta_fitx)
    squared_error = np.sum(np.power(delta_fitx, 2))
    #print(squared_error)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left_left = np.array([np.transpose(np.vstack([left_left_fitx, yvals]))])
    pts_left_right = np.array([np.flipud(np.transpose(np.vstack([left_right_fitx, yvals])))])
    pts_left = np.hstack((pts_left_left, pts_left_right))


    #right_left_fitx = right_lane_fit[0] * yvals ** 2 + right_lane_fit[1] * yvals + right_lane_fit[2] - lane_range
    #right_right_fitx = right_lane_fit[0] * yvals ** 2 + right_lane_fit[1] * yvals + right_lane_fit[2] + lane_range

    right_left_fit = calculate_line_shape_area(right_lane_fit, shape_structure=shape_structure_left)
    right_left_fitx = right_left_fit[0] * yvals ** 2 + right_left_fit[1] * yvals + right_left_fit[2]
    right_right_fit = calculate_line_shape_area(right_lane_fit, shape_structure=shape_structure_right)
    right_right_fitx = right_right_fit[0] * yvals ** 2 + right_right_fit[1] * yvals + right_right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_right_left = np.array([np.transpose(np.vstack([right_left_fitx, yvals]))])
    pts_right_right = np.array([np.flipud(np.transpose(np.vstack([right_right_fitx, yvals])))])
    pts_right = np.hstack((pts_right_left, pts_right_right))

    area_select = perform_image_area(img, area_percentage)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(left_lane_binary, np.int_([pts_left]), 255)
    cv2.fillPoly(right_lane_binary, np.int_([pts_right]), 255)
    cv2.fillPoly(area_binary, area_select, 255)

    cv2.imshow('Right_line', right_lane_binary)

    left_lane_position = cv2.bitwise_and(area_binary, left_lane_binary)
    right_lane_position = cv2.bitwise_and(area_binary, right_lane_binary)

    left_lane_pixels = cv2.bitwise_and(img, left_lane_position)
    right_lane_pixels = cv2.bitwise_and(img, right_lane_position)

    #new_image = mio.image_cluster([img, img, left_lane_position, right_lane_position, left_lane_pixels, right_lane_pixels])
    #cv2.imshow('test',new_image)
    #cv2.waitKey(200)

    return left_lane_pixels, right_lane_pixels

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
    calculate_line_shape_area([-0.8,1,0])

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/test1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    position_select_binary = position_select(image, ignore_mask_color=255)

    #mpo.plot_cluster([image, position_select_binary], img_text=['Original Image', 'Position Select'])