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

def calc_car_2_line(lane_pos_bottom):
    xm_per_pix = 3.7 / 200
    car_2_line_px = abs(lane_pos_bottom - 640)
    car_2_line_m = car_2_line_px * xm_per_pix
    return car_2_line_m

def calc_car_2_centerline(left_lane_pos, right_lane_pos):
    xm_per_pix = 3.7 / 200
    lane_width_px = (right_lane_pos - left_lane_pos)
    car_2_centerline_px= ((640 - left_lane_pos) - (lane_width_px / 2))
    car_2_centerline = car_2_centerline_px * xm_per_pix
    return car_2_centerline

def calc_curve_radius_px(fit_cr, y_eval):
    # Define conversions in x and y from pixels space to meters
    y_eval = y_eval
    curverad_px = ((1 + ((2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5)) / np.absolute(2 * fit_cr[0])
    # Now our radius of curvature is in meters
    return curverad_px

def calc_curve_radius(xvals, yvals, y_eval):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 200  # meteres per pixel in x dimension
    y_eval = ym_per_pix * y_eval
    fit_cr = np.polyfit(yvals * ym_per_pix, xvals * xm_per_pix, 2)
    curverad = ((1 + ((2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5)) / np.absolute(2 * fit_cr[0])
    # Now our radius of curvature is in meters
    return curverad

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

def create_fitted_area_1(lane_img):
    img_shape = lane_img.shape

    lane_fit = fit_lane_line(lane_img)

    yvals = np.array(range(img_shape[0]))
    fitx = lane_fit[0] * yvals ** 2 + lane_fit[1] * yvals + lane_fit[2]

    return fitx, lane_fit, yvals

def create_fitted_lane_img(left_line_fit, right_line_fit, yvals):
    fitted_lane_img = np.zeros((720, 1280), dtype=np.uint8)

    left_fitx = left_line_fit[0] * yvals ** 2 + left_line_fit[1] * yvals + left_line_fit[2]
    right_fitx = right_line_fit[0] * yvals ** 2 + right_line_fit[1] * yvals + right_line_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(fitted_lane_img, np.int_([pts]), 1)

    return fitted_lane_img
