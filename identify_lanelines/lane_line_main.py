import numpy as np
import cv2
import matplotlib.pyplot as plt

import pandas as pd

import identify_lanelines.line_class as line_class

import calibration.correctDistortion as correctDistortion
import image_preprocessing.combined_threshold as combined_threshold

import perspective_transform.image_transform as img_transform

import image_preprocessing.image_position as img_position
import image_preprocessing.image_color as img_color

import identify_lanelines.identify_area as identify_area
import identify_lanelines.identify_radius as identify_radius

import toolbox.multiple_image_out as mio

#import imageio
#imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

MTX=None
DIST=None

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

def detect_lane_lines():
    return 0

def process_image(raw_image, cvtColor='RGB'):
    # Correct Distortion with calculated Camera Calibration (If not present calibrate)
    global MTX, DIST
    mtx, dist, raw_image = correctDistortion.correct_distortion(raw_image, mtx=MTX, dist=DIST)
    if (MTX == None) | (DIST == None):
        MTX = mtx
        DIST = dist

    # Preprocess Image to filter for LanePixels
    if cvtColor == 'RGB':
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

    src, dst = img_transform.perform_initial_sourcepoints()
    warped_image = img_transform.warper(raw_image, src, dst)

    warped_image_cont = img_color.add_contrast(warped_image)
    warped_combined = combined_threshold.combined_thresholds_complete(warped_image_cont, verbose=True)

    # Find Lane Pixels
    if (left_line.detected == False) | (right_line.detected == False):
        histogram1 = identify_area.create_histogram(warped_combined, y_area_of_image=[.66, 1])
        lanes, left_lane, right_lane = identify_area.identify_lane_position(histogram1)

        identified_left_curve_area, identified_right_curve_area = img_position.perform_lane_position(warped_combined,
                                                                                                     left_lane_fit=[0,0,left_lane],
                                                                                                     right_lane_fit=[0,0,right_lane],
                                                                                                     area_percentage=.5)
        left_line.detected = True
        right_line.detected = True
        fitted_lane_img, left_lane_fit, right_lane_fit, left_curverad, right_curverad, car_2_centerline = identify_radius.create_fitted_area_1(identified_left_curve_area, identified_right_curve_area)

        left_line.best_fit = left_lane_fit
        right_line.best_fit = right_lane_fit

    identified_left_curve_area, identified_right_curve_area = img_position.perform_lane_position(warped_combined,
                                                                                                    left_lane_fit=left_line.best_fit,
                                                                                                    right_lane_fit=right_line.best_fit,
                                                                                                    area_percentage=1)

    # abs_left_lane = left_lane
    # abs_right_lane = right_lane

    # identified_curve_area[((identified_curve_area == 1) & (warped_combined == 1))] = 1

    # identified_left_curve_area, identified_right_curve_area = identify_area.identify_curve_area(warped_combined, abs_left_lane, abs_right_lane, verbose=False)


    fitted_lane_img, left_lane_fit, right_lane_fit, left_curverad, right_curverad, car_2_centerline = identify_radius.create_fitted_area_1(identified_left_curve_area, identified_right_curve_area)

    left_line.best_fit = left_lane_fit
    right_line.best_fit = right_lane_fit
    warped_fitted_lane_img = img_transform.warper(fitted_lane_img, src, dst, direction='backward')

    blue_image = np.zeros((warped_fitted_lane_img.shape[0], warped_fitted_lane_img.shape[1], 3), np.uint8)
    blue_image[((warped_fitted_lane_img == 1))] = (255, 0, 0)
    combo = weighted_img(blue_image, raw_image, α=0.8, β=1., λ=0.)

    curve_radius_mean = (left_curverad + right_curverad) / 2
    add_text = 'Radius of Curvature = %0.1f(Meter)' %curve_radius_mean
    cv2.putText(combo, add_text, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    if car_2_centerline > 0:
        add_text = 'Car drives %0.2f Meter right of Center' %car_2_centerline
        cv2.putText(combo, add_text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    elif car_2_centerline == 0:
        add_text = 'Car drives in the Center' %car_2_centerline
        cv2.putText(combo, add_text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    elif car_2_centerline < 0:
        add_text = 'Car drives %0.2f Meter left of Center' %(-car_2_centerline)
        cv2.putText(combo, add_text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow('Window3', combo)
    cv2.waitKey(1)
    combo = cv2.cvtColor(combo, cv2.COLOR_BGR2RGB)

    return combo


if __name__ == "__main__":
    left_line = line_class.Line()
    right_line = line_class.Line()

    #image = cv2.imread('../test_images/test5.jpg')
    #combo = process_image(image)
    #cv2.imshow('Window', combo)
    #cv2.waitKey(1000)

    video_output = '../challenge_video_calc.mp4'
    #clip1 = VideoFileClip('../project_video.mp4')
    clip1 = VideoFileClip('../challenge_video.mp4')
    #clip1 = VideoFileClip('../harder_challenge_video.mp4')


    white_clip_1 = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip_1.write_videofile(video_output, audio=False)
