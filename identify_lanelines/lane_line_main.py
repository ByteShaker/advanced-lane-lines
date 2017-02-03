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

from moviepy.editor import VideoFileClip

MTX=None
DIST=None

verbose_glob=False

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def detect_lane_lines():
    return 0

def process_image(raw_image, cvtColor='RGB', verbose=False):
    img_shape = raw_image.shape

    look_at_image_area_percentage = min(left_line.image_area_percentage, right_line.image_area_percentage)

    # Correct Distortion with calculated Camera Calibration (If not present calibrate)
    global MTX, DIST, verbose_glob
    mtx, dist, raw_image = correctDistortion.correct_distortion(raw_image, mtx=MTX, dist=DIST, verbose=verbose_glob)
    if (MTX == None) | (DIST == None):
        MTX = mtx
        DIST = dist

    # Preprocess Image to filter for LanePixels
    if cvtColor == 'RGB':
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)

    # Dynamic Transform of Image Image
    lane_width_bottom, lane_width_top, bottom_angle = img_transform.calc_correct_transform(left_line, right_line, look_at_image_area_percentage, img_shape)
    if (lane_width_bottom != None) & (lane_width_top != None):
        master_lane.add_new_lanefit(lane_width_bottom, lane_width_top, bottom_angle)
    src, dst = img_transform.calc_new_sourcepoints(master_lane.mean_lane_width_bottom, master_lane.mean_lane_width_top, master_lane.mean_bottom_angle, img_shape)
    warped_image = img_transform.warper(raw_image, src, dst)

    # Add Thresholds, Gradient and Direction
    warped_combined = combined_threshold.combined_thresholds_complete(warped_image, verbose=True)

    # Find Lane Pixels
    if (left_line.detected == False) | (right_line.detected == False):
        y_area_of_image = [[.8,1],[.7,9],[.6,8],[.5,7]]
        histogram_bottom = [0,0,0,0]
        left_line_points = [0,0,0,0]
        right_lane_points = [0,0,0,0]
        for i in range(len(y_area_of_image)):
            histogram_bottom[i] = identify_area.create_histogram(warped_combined, y_area_of_image=y_area_of_image[i])
            lanes, left_line_points[i], right_lane_points[i] = identify_area.identify_lane_position(histogram_bottom[i])

        inital_left_line_fit = np.polyfit([.9*img_shape[0],.8*img_shape[0],.7*img_shape[0],.6*img_shape[0]], left_line_points, 2)
        inital_right_line_fit = np.polyfit([.9*img_shape[0],.8*img_shape[0],.7*img_shape[0],.6*img_shape[0]], right_lane_points, 2)
        identified_left_curve_area, identified_right_curve_area = img_position.perform_lane_position(warped_combined,
                                                                                                     left_lane_fit=inital_left_line_fit,
                                                                                                     right_lane_fit=inital_right_line_fit,
                                                                                                     area_percentage=look_at_image_area_percentage)

        if (np.max(identified_right_curve_area) > 0) & (np.max(identified_left_curve_area) > 0):
            left_lane_fit = identify_radius.fit_lane_line(identified_left_curve_area)
            right_lane_fit = identify_radius.fit_lane_line(identified_right_curve_area)

            yvals = np.array(range(img_shape[0]))

            left_line.add_new_linefit(left_lane_fit, yvals)
            right_line.add_new_linefit(right_lane_fit, yvals)


    identified_left_curve_area, identified_right_curve_area = img_position.perform_lane_position(warped_combined,
                                                                                                left_lane_fit=left_line.best_fit,
                                                                                                right_lane_fit=right_line.best_fit,
                                                                                                area_percentage=look_at_image_area_percentage)

    color_image = np.zeros((warped_combined.shape[0], warped_combined.shape[1], 3), np.uint8)
    color_image[((identified_left_curve_area >= 1))] = (255, 0, 0)
    color_image[((identified_right_curve_area >= 1))] = (0, 0, 255)
    color_image = weighted_img(color_image, warped_image, α=0.8, β=1., λ=0.)

    search_Area = mio.image_cluster([color_image],['Identifyed Laneline Pixels'], font_size=2, y_position=80)
    cv2.imshow('Identified line AREA', search_Area)


    if (np.max(identified_right_curve_area) > 0) & (np.max(identified_left_curve_area) > 0):

        left_lane_fit = identify_radius.fit_lane_line(identified_left_curve_area)
        right_lane_fit = identify_radius.fit_lane_line(identified_right_curve_area)

        yvals = np.array(range(img_shape[0]))

        left_line.add_new_linefit(left_lane_fit, yvals)
        right_line.add_new_linefit(right_lane_fit, yvals)


    fitted_lane_img = identify_radius.create_fitted_lane_img(left_line.current_fit, right_line.current_fit, left_line.ally)

    warped_fitted_lane_img = img_transform.warper(fitted_lane_img, src, dst, direction='backward')


    blue_image = np.zeros((warped_fitted_lane_img.shape[0], warped_fitted_lane_img.shape[1], 3), np.uint8)
    blue_image[((warped_fitted_lane_img == 1))] = (255, 0, 0)
    combo = weighted_img(blue_image, raw_image, α=1, β=1., λ=0.)

    car_2_centerline = (left_line.line_base_pos/ 2.) - (right_line.line_base_pos/ 2.)
    curve_radius_mean = (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2

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
    master_lane = line_class.Lane()

    verbose_glob = False

    #image = cv2.imread('../test_images/test2.jpg')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #combo = process_image(image)
    #combo = cv2.cvtColor(combo,cv2.COLOR_RGB2BGR)
    #cv2.imshow('Window', combo)

    #cv2.waitKey()

    #cv2.imwrite('../output_images/test2_applied_lane_lines.jpg', combo)

    video_output = '../project_video_calc2.mp4'
    clip1 = VideoFileClip('../project_video.mp4')
    #clip1 = VideoFileClip('../challenge_video.mp4')
    #clip1 = VideoFileClip('../harder_challenge_video.mp4')

    white_clip_1 = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip_1.write_videofile(video_output, audio=False)
