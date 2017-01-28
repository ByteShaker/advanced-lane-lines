import numpy as np
import cv2
import matplotlib.pyplot as plt

import pandas as pd

import calibration.correctDistortion as correctDistortion
import image_preprocessing.combined_threshold as combined_threshold

import perspective_transform.image_transform as img_transform

import image_preprocessing.image_position as img_position

import identify_lanelines.identify_area as identify_area
import identify_lanelines.identify_radius as identify_radius

#import imageio
#imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

MTX=None
DIST=None

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(raw_image):
    # Correct Distortion with calculated Camera Calibration (If not present calibrate)
    global MTX, DIST
    mtx, dist, raw_image = correctDistortion.correct_distortion(raw_image, mtx=MTX, dist=DIST)
    if (MTX == None) | (DIST == None):
        MTX = mtx
        DIST = dist

    # Preprocess Image to filter for LanePixels
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    combined = combined_threshold.combined_thresholds_1(image)
    src, dst = img_transform.perform_initial_sourcepoints()
    warped_combined = img_transform.warper(combined, src, dst)

    histogram = identify_area.create_histogram(warped_combined, y_area_of_image=[2 / 3, 1])
    lanes, left_lane, right_lane = identify_area.identify_lane_position(histogram)

    abs_left_lane = left_lane
    abs_right_lane = right_lane
    #print(abs_left_lane, abs_right_lane)

    identified_left_curve_area = np.zeros_like(warped_combined)
    identified_right_curve_area = np.zeros_like(warped_combined)
    img_shape = warped_combined.shape

    img_areas = 10
    for i in range(img_areas):
        histogram = identify_area.create_histogram(warped_combined, y_area_of_image=[(img_areas - (i + 1)) / img_areas, (img_areas - i) / img_areas])
        # histogram_df = pd.DataFrame(histogram)
        # histogram_df['Peak_Filter'] = histogram_df[0].rolling(window=200, axis=0, center=True, win_type='gaussian').mean(std=10.0)
        # histogram_df['Peak_Filter'].plot()
        temp_lane_window = 100 + ((i / img_areas) * 100)
        lanes, left_lane, right_lane = identify_area.identify_lane_position(histogram, std=10, approx_left_lane=left_lane,
                                                              approx_right_lane=right_lane, peak_filter_threshold=0,
                                                              lane_window=temp_lane_window)
        #print(left_lane, right_lane)

        left_left_border, left_right_border = identify_area.border_control(left_lane, temp_lane_window, 0, img_shape[1])
        temp_left_area = np.array([[(left_left_border, img_shape[0] * ((img_areas - i) / img_areas)),
                                    (left_left_border, img_shape[0] * ((img_areas - (i + 1)) / img_areas)),
                                    (left_right_border, img_shape[0] * ((img_areas - (i + 1)) / img_areas)),
                                    (left_right_border, img_shape[0] * ((img_areas - i) / img_areas))]], dtype=np.int32)
        temp_left_img_position = img_position.position_select(warped_combined, temp_left_area)

        identified_left_curve_area[((identified_left_curve_area == 1) | (temp_left_img_position == 1))] = 1

        right_left_border, right_right_border = identify_area.border_control(right_lane, temp_lane_window, 0, img_shape[1])
        temp_right_area = np.array([[(right_left_border, img_shape[0] * ((img_areas - i) / img_areas)),
                                     (right_left_border, img_shape[0] * ((img_areas - (i + 1)) / img_areas)),
                                     (right_right_border, img_shape[0] * ((img_areas - (i + 1)) / img_areas)),
                                     (right_right_border, img_shape[0] * ((img_areas - i) / img_areas))]],
                                   dtype=np.int32)
        temp_right_img_position = img_position.position_select(warped_combined, temp_right_area)

        identified_right_curve_area[((identified_right_curve_area == 1) | (temp_right_img_position == 1))] = 1

        # f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 9))
        # y_area_of_image = [(img_areas - (i + 1)) / img_areas, (img_areas - i) / img_areas]
        # ax1.imshow(warped_combined[warped_combined.shape[0] * y_area_of_image[0]:warped_combined.shape[0] * y_area_of_image[1], :], cmap='gray')
        # ax1.set_title('Original Image', fontsize=20)
        # pd.DataFrame(histogram).plot(ax=ax2)
        # ax2.set_title('Cobined Threshold', fontsize=20)
        # lanes.plot(ax=ax3)
        # ax3.set_title('Cobined Threshold', fontsize=20)
        # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
        # plt.show()

    # identified_curve_area[((identified_curve_area == 1) & (warped_combined == 1))] = 1


    fitted_lane_img = identify_radius.create_fitted_area(identified_left_curve_area, identified_right_curve_area,
                                                         abs_left_lane, abs_right_lane)
    warped_fitted_lane_img = img_transform.warper(fitted_lane_img, src, dst, direction='backward')

    blue_image = np.zeros((warped_fitted_lane_img.shape[0], warped_fitted_lane_img.shape[1], 3), np.uint8)
    # red_image[:,:] = (0, 0, 255)
    blue_image[((warped_fitted_lane_img == 1))] = (255, 0, 0)
    combo = weighted_img(blue_image, raw_image, α=0.8, β=1., λ=0.)

    return combo


if __name__ == "__main__":

    #image = cv2.imread('../test_images/straight_lines2.jpg')
    #combo = process_image(image)
    #cv2.imshow('Window', combo)
    #cv2.waitKey()

    white_output = '../white_2.mp4'
    clip1 = VideoFileClip('../project_video.mp4')
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
