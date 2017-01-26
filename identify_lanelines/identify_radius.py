import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix

def fit_lane_line(lane_img):
    # Fit a second order polynomial to each fake lane line
    coo_lane_img = coo_matrix(lane_img)

    yvals = coo_lane_img.row
    xvals = coo_lane_img.col

    lane_fit = np.polyfit(yvals, xvals, 2)

    return lane_fit


def create_fitted_area(left_lane_img, right_lane_img, abs_left_lane, abs_right_lane):
    fitted_lane_img = np.zeros_like(left_lane_img)
    img_shape = fitted_lane_img.shape

    lane_width = (abs_right_lane - abs_left_lane)
    combined_lane_img = np.roll(left_lane_img, lane_width, axis=1)
    combined_lane_img[((combined_lane_img == 1) | (right_lane_img == 1))] = 1
    lane_fit = fit_lane_line(combined_lane_img)

    for row in range(img_shape[0]):
        fitx = lane_fit[0] * row ** 2 + lane_fit[1] * row + lane_fit[2]
        for col in range(img_shape[1]):
            if ((col >= (fitx - lane_width)) & (col <= fitx)):
                fitted_lane_img[row][col] = 1

    yvals = np.array(range(720))
    left_fitx = lane_fit[0] * yvals ** 2 + lane_fit[1] * yvals + lane_fit[2]

    # Plot up the fake data
    #plt.plot(xvals, yvals, 'o', color='red')
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, yvals, color='green', linewidth=3)
    plt.gca().invert_yaxis()  # to visualize as we do the images

    return fitted_lane_img