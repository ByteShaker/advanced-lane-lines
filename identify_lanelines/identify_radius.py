import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

import cv2

def fit_lane_line(lane_img):
    # Fit a second order polynomial to each fake lane line
    coo_lane_img = coo_matrix(lane_img)

    yvals = coo_lane_img.row
    xvals = coo_lane_img.col

    lane_fit = np.polyfit(yvals, xvals, 2)

    return lane_fit


def create_fitted_area(left_lane_img, right_lane_img, abs_left_lane, abs_right_lane):
    fitted_lane_img = np.zeros_like(left_lane_img, dtype=np.uint8)
    img_shape = fitted_lane_img.shape

    lane_width = (abs_right_lane - abs_left_lane)
    combined_lane_img = np.roll(left_lane_img, lane_width, axis=1)
    combined_lane_img[((combined_lane_img == 1) | (right_lane_img == 1))] = 1
    lane_fit = fit_lane_line(combined_lane_img)

    yvals = np.array(range(img_shape[0]))
    left_fitx = lane_fit[0] * yvals ** 2 + lane_fit[1] * yvals + lane_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([left_fitx-lane_width, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(fitted_lane_img, np.int_([pts]), 1)

    # Plot up the fake data
    #plt.plot(xvals, yvals, 'o', color='red')
    #plt.xlim(0, 1280)
    #plt.ylim(0, 720)
    #plt.plot(left_fitx, yvals, color='green', linewidth=3)
    #plt.gca().invert_yaxis()  # to visualize as we do the images

    return fitted_lane_img

def create_fitted_area_1(left_lane_img, right_lane_img):
    fitted_lane_img = np.zeros_like(left_lane_img, dtype=np.uint8)
    img_shape = fitted_lane_img.shape

    #lane_width = (abs_right_lane - abs_left_lane)
    left_lane_fit = fit_lane_line(left_lane_img)
    right_lane_fit = fit_lane_line(right_lane_img)

    yvals = np.array(range(img_shape[0]))
    left_fitx = left_lane_fit[0] * yvals ** 2 + left_lane_fit[1] * yvals + left_lane_fit[2]
    right_fitx = right_lane_fit[0] * yvals ** 2 + right_lane_fit[1] * yvals + right_lane_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(fitted_lane_img, np.int_([pts]), 1)

    # Plot up the fake data
    #plt.plot(xvals, yvals, 'o', color='red')
    #plt.xlim(0, 1280)
    #plt.ylim(0, 720)
    #plt.plot(left_fitx, yvals, color='green', linewidth=3)
    #plt.gca().invert_yaxis()  # to visualize as we do the images

    return fitted_lane_img, left_lane_fit, right_lane_fit