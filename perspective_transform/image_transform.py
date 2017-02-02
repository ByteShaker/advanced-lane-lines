import numpy as np
import cv2
import matplotlib.pyplot as plt

import identify_lanelines.line_class as line_class

import glob
import pickle

import toolbox.multiple_plots_out as mpo

def perform_initial_sourcepoints():
    # src = np.float32(
    #         [[602, 444],
    #          [220, 705],
    #          [1083, 705],
    #          [678, 444]])
    src = np.float32(
        [[600, 444],
         [210, 705],
         [1070, 705],
         [680, 444]])


    dst = np.float32(
            [[540, 0],
             [540, 720],
             [740, 720],
             [740, 0]])

    return (src, dst)

def calc_cross_parallel_point(base_fit, second_fit):

    inner_factor = np.sqrt(np.power((second_fit[1] + (1/base_fit[1])), 2) - (4 * second_fit[0] * (second_fit[2]-base_fit[2])))

    parallel_point_y1 = ((-second_fit[1] - (1/base_fit[1])) + inner_factor) / (2 * second_fit[0])
    parallel_point_y2 = ((-second_fit[1] - (1 / base_fit[1])) - inner_factor) / (2 * second_fit[0])

    if abs(parallel_point_y1) < abs(parallel_point_y2):
        cross_parallel_point = parallel_point_y1
    else:
        cross_parallel_point = parallel_point_y2

    #if cross_parallel_point < 0:
     #   cross_parallel_point = 0

    return cross_parallel_point

def calc_correct_transform(left_line, right_line, look_at_image_area_percentage=1, img_shape=(720,1280), trust_detection=True):
    image_middle = int(img_shape[1]/2)

    if (left_line.detected == False) | (right_line.detected == False):
        lane_width_bottom = None
        lane_width_top = None
        bottom_angle = 0.

    elif trust_detection:
        #Correct Angle Offset
        left_bottom_angle = 2 * left_line.current_fit[0] * max(left_line.ally) + left_line.current_fit[1]
        right_bottom_angle = 2 * right_line.current_fit[0] * max(right_line.ally) + right_line.current_fit[1]
        bottom_angle = (left_bottom_angle + right_bottom_angle) / 2.

        #Correct Trapez
        curve_type = np.mean([left_line.current_fit[0], left_line.current_fit[0]])
        lane_width_bottom = (right_line.bestx[-1]) - (left_line.bestx[-1])

        if curve_type >= 0: # right curve
            base_fit = left_line.current_fit
            second_fit = right_line.current_fit
            parallel_point_y = calc_cross_parallel_point(base_fit, second_fit)
            parallel_point_x = second_fit[0] * parallel_point_y ** 2 + second_fit[1] * parallel_point_y + second_fit[2]

            lane_width_top = np.sqrt((np.power(parallel_point_y, 2) + np.power(parallel_point_x - left_line.bestx[0], 2)))

        elif curve_type < 0: # left curve
            base_fit = right_line.current_fit
            second_fit = left_line.current_fit
            parallel_point_y = calc_cross_parallel_point(base_fit, second_fit)
            parallel_point_x = second_fit[0] * parallel_point_y ** 2 + second_fit[1] * parallel_point_y + second_fit[2]

            lane_width_top = np.sqrt((np.power(parallel_point_y, 2) + np.power(parallel_point_x - right_line.bestx[0], 2)))

    else:
        lane_width_bottom = (right_line.bestx[-1]) - (left_line.bestx[-1])
        lane_width_top = (right_line.bestx[0]) - (left_line.bestx[0])
        bottom_angle = 0

    return lane_width_bottom, lane_width_top, bottom_angle

def calc_new_sourcepoints(lane_width_bottom=None, lane_width_top=None, bottom_angle=0.):
    inital_px_top = 70
    px_shift = 720 * bottom_angle
    image_middle = 640 - px_shift
    if (lane_width_bottom==None) | (lane_width_top==None):
        left_pos_top = image_middle - int(inital_px_top / 2)
        right_pos_top = image_middle + int(inital_px_top / 2)
    else:
        new_px_top = (lane_width_top / lane_width_bottom) * inital_px_top
        left_pos_top = image_middle - int(new_px_top / 2)
        right_pos_top = image_middle + int(new_px_top / 2)

    src = np.float32(
        [[left_pos_top, 440],
         [210, 705],
         [1070, 705],
         [right_pos_top, 440]])

    dst = np.float32(
        [[540, 0],
         [540, 720],
         [740, 720],
         [740, 0]])

    return (src, dst)


def warper(img, src, dst, direction='forward'):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    if direction == 'forward':
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    elif direction == 'backward':
        Minv = cv2.getPerspectiveTransform(dst, src)
        warped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/test5.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    src, dst = perform_initial_sourcepoints()
    warped_image = warper(image, src, dst)

    mpo.plot_cluster([image, warped_image], img_text=['Original Image', 'Warped Image'])