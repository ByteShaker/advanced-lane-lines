import numpy as np
import cv2
import matplotlib.pyplot as plt

import pandas as pd

import calibration.correctDistortion as correctDistortion
import image_preprocessing.combined_threshold as combined_threshold

import perspective_transform.image_transform as img_transform

import image_preprocessing.image_position as img_position

import identify_lanelines.identify_radius as identify_radius

import toolbox.multiple_plots_out as mpo

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

def border_control(approx_middle, window, low, high):
    left_border = approx_middle - (window / 2)
    left_border = int(left_border) if left_border > low else int(low)
    right_border = approx_middle + (window / 2)
    right_border = int(right_border) if right_border < high else int(high)

    return left_border, right_border

def kernel_density_estimation(dataframe, rolling_window=1000, std=40):

    outer_side = int(rolling_window/2)
    dataframe_len = len(dataframe)
    df1 = pd.DataFrame(np.zeros(outer_side))

    extended_dataframe = df1.append(dataframe).append(df1).reset_index()
    extended_dataframe = extended_dataframe.drop('index', axis=1)
    kde_dataframe = extended_dataframe.rolling(window=rolling_window, axis=0, center=True, win_type='gaussian').mean(std=std)
    kde_dataframe = kde_dataframe[outer_side:outer_side+dataframe_len].reset_index().drop('index', axis=1)
    return kde_dataframe

def gradient_change(args):
    if (args[0]*args[1]) <= 0:
        return 1
    else:
        return 0

def create_histogram(img, y_area_of_image=[0, 1]):
    bottom_value = int(img.shape[0] * y_area_of_image[1])
    top_value = int(img.shape[0] * y_area_of_image[0])
    histogram = np.sum(img[top_value:bottom_value, :], axis=0)

    return histogram

def identify_line_position(histogram, rolling_window=1000, std=10.0, approx_pos=540, peak_filter_threshold=10, peak_filter_dot_dot_threshold=0, lane_window=180, verbose=False):
    #Todo: Complete function
    histogram_df = pd.DataFrame(histogram)
    histogram_df['Peak_Filter'] = kernel_density_estimation(histogram_df, rolling_window=rolling_window, std=std)
    histogram_df['Peak_Filter_dot'] = histogram_df['Peak_Filter'].diff()
    histogram_df['Peak_Filter_dot_zero'] = histogram_df['Peak_Filter_dot'].rolling(window=2, axis=0, center=True).apply(gradient_change)
    histogram_df['Peak_Filter_dot_dot'] = histogram_df['Peak_Filter_dot'].diff()

    histogram_df['Lanes'] = np.where((histogram_df['Peak_Filter'] > peak_filter_threshold)
                                     & (histogram_df['Peak_Filter_dot_zero'] == 1)
                                     & (histogram_df['Peak_Filter_dot_dot'] < peak_filter_dot_dot_threshold),
                                     histogram_df['Peak_Filter_dot_dot'].abs(), 0)

    histogram_len = len(histogram_df['Lanes'])

    if approx_left_lane==None:
        half_dataframe_length = int(histogram_len / 2)
        left_area = histogram_df['Lanes'][:half_dataframe_length]
    else:
        left_border, right_border = border_control(approx_left_lane, lane_window, 0, histogram_len)
        left_area = histogram_df['Lanes'][left_border:right_border]

    if approx_right_lane==None:
        half_dataframe_length = int(histogram_len / 2)
        right_area = histogram_df['Lanes'][half_dataframe_length:histogram_len]
    else:
        left_border, right_border = border_control(approx_right_lane, lane_window, 0, histogram_len)
        right_area = histogram_df['Lanes'][left_border:right_border]

    left_lane = left_area.idxmax() if left_area.max() > 0 else approx_left_lane
    right_lane = right_area.idxmax() if right_area.max() > 0 else approx_right_lane

    if verbose==True:
        print(left_lane, right_lane)
        mpo.plot_cluster([pd.DataFrame(histogram), histogram_df['Peak_Filter'],histogram_df['Peak_Filter_dot'],histogram_df['Peak_Filter_dot_zero'],histogram_df['Lanes'],histogram_df['Peak_Filter_dot_Nullstellen']])

    return histogram_df['Lanes'], left_lane, right_lane

def identify_lane_position(histogram, rolling_window=1000, std=10.0, approx_left_lane=540, approx_right_lane=740, peak_filter_threshold=10, peak_filter_dot_threshold=20., peak_filter_dot_dot_threshold=0, lane_window=100, verbose=False):
    histogram_df = pd.DataFrame(histogram)
    histogram_df['Peak_Filter'] = kernel_density_estimation(histogram_df, rolling_window=rolling_window, std=std)
    histogram_df['Peak_Filter_dot'] = histogram_df['Peak_Filter'].diff()
    histogram_df['Peak_Filter_dot_zero'] = histogram_df['Peak_Filter_dot'].rolling(window=2, axis=0, center=True).apply(gradient_change)
    histogram_df['Peak_Filter_dot_dot'] = histogram_df['Peak_Filter_dot'].diff()

    histogram_df['Lanes'] = np.where((histogram_df['Peak_Filter'] > peak_filter_threshold)
                                     & (histogram_df['Peak_Filter_dot_zero'] == 1)
                                     & (histogram_df['Peak_Filter_dot_dot'] < peak_filter_dot_dot_threshold),
                                     histogram_df['Peak_Filter_dot_dot'].abs(), 0)

    histogram_len = len(histogram_df['Lanes'])

    if approx_left_lane==None:
        half_dataframe_length = int(histogram_len / 2)
        left_area = histogram_df['Lanes'][:half_dataframe_length]
    else:
        left_border, right_border = border_control(approx_left_lane, lane_window, 0, histogram_len)
        left_area = histogram_df['Lanes'][left_border:right_border]

    if approx_right_lane==None:
        half_dataframe_length = int(histogram_len / 2)
        right_area = histogram_df['Lanes'][half_dataframe_length:histogram_len]
    else:
        left_border, right_border = border_control(approx_right_lane, lane_window, 0, histogram_len)
        right_area = histogram_df['Lanes'][left_border:right_border]

    left_lane = left_area.idxmax() if left_area.max() > 0 else approx_left_lane
    right_lane = right_area.idxmax() if right_area.max() > 0 else approx_right_lane

    if verbose==True:
        print(left_lane, right_lane)
        mpo.plot_cluster([pd.DataFrame(histogram), histogram_df['Peak_Filter'],histogram_df['Peak_Filter_dot'],histogram_df['Peak_Filter_dot_zero'],histogram_df['Lanes'],histogram_df['Peak_Filter_dot_Nullstellen']])

    return histogram_df['Lanes'], left_lane, right_lane

def identify_curve_area(pixel_image, left_lane, right_lane, verbose=False):
    identified_left_curve_area = np.zeros_like(pixel_image, dtype=np.uint8)
    identified_right_curve_area = np.zeros_like(pixel_image, dtype=np.uint8)
    img_shape = pixel_image.shape

    img_areas = 10
    for i in range(img_areas):
        histogram = create_histogram(pixel_image, y_area_of_image=[(img_areas - (i + 1)) / img_areas,
                                                                                     (img_areas - i) / img_areas])
        temp_lane_window = 20 + ((i / img_areas) * 20)
        lanes, left_lane, right_lane = identify_lane_position(histogram,
                                                              std=50,
                                                              approx_left_lane=left_lane,
                                                              approx_right_lane=right_lane,
                                                              peak_filter_threshold=10,
                                                              lane_window=temp_lane_window)
        # print(left_lane, right_lane)

        left_left_border, left_right_border = border_control(left_lane, temp_lane_window, 0, img_shape[1])
        temp_left_area = np.array([[(left_left_border, img_shape[0] * ((img_areas - i) / img_areas)),
                                    (left_left_border, img_shape[0] * ((img_areas - (i + 1)) / img_areas)),
                                    (left_right_border, img_shape[0] * ((img_areas - (i + 1)) / img_areas)),
                                    (left_right_border, img_shape[0] * ((img_areas - i) / img_areas))]], dtype=np.int32)
        temp_left_img_position = img_position.position_select(pixel_image, temp_left_area)

        identified_left_curve_area[((identified_left_curve_area >= 1) | (temp_left_img_position >= 1))] = 1

        right_left_border, right_right_border = border_control(right_lane, temp_lane_window, 0,
                                                                             img_shape[1])
        temp_right_area = np.array([[(right_left_border, img_shape[0] * ((img_areas - i) / img_areas)),
                                     (right_left_border, img_shape[0] * ((img_areas - (i + 1)) / img_areas)),
                                     (right_right_border, img_shape[0] * ((img_areas - (i + 1)) / img_areas)),
                                     (right_right_border, img_shape[0] * ((img_areas - i) / img_areas))]],
                                   dtype=np.int32)
        temp_right_img_position = img_position.position_select(pixel_image, temp_right_area)

        identified_right_curve_area[((identified_right_curve_area >= 1) | (temp_right_img_position >= 1))] = 1

        if verbose == True:
            mpo.plot_cluster([pixel_image, identified_left_curve_area, identified_right_curve_area])

    return identified_left_curve_area, identified_right_curve_area

if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/test2.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mtx, dist, image = correctDistortion.correct_distortion(image)

    combined = combined_threshold.combined_thresholds_complete(image)

    src, dst = img_transform.perform_initial_sourcepoints()
    warped_combined = img_transform.warper(combined, src, dst)

    histogram = create_histogram(warped_combined, y_area_of_image=[2/3, 1])

    lanes, left_lane, right_lane = identify_lane_position(histogram)

    abs_left_lane = left_lane
    abs_right_lane = right_lane

    #identified_curve_area[((identified_curve_area == 1) & (warped_combined == 1))] = 1

    identified_left_curve_area, identified_right_curve_area = identify_curve_area(warped_combined, abs_left_lane, abs_right_lane, verbose=True)

    #cv2.imshow('test', identified_left_curve_area)
    #cv2.waitKey(0)

    fitted_lane_img = identify_radius.create_fitted_area(identified_left_curve_area, identified_right_curve_area, abs_left_lane, abs_right_lane)
    warped_fitted_lane_img = img_transform.warper(fitted_lane_img, src, dst, direction='backward')

    red_image = np.zeros((warped_fitted_lane_img.shape[0], warped_fitted_lane_img.shape[1], 3), np.uint8)
    #red_image[:,:] = (0, 0, 255)
    red_image[((warped_fitted_lane_img == 1))] = (0, 0, 255)
    combo = weighted_img(red_image, image, α=0.8, β=1., λ=0.)

    #histogram_df['left_Lane'].plot()

    mpo.plot_cluster([image, combo, warped_combined, lanes], img_text=['Original Image', 'Cobined Threshold', 'Warped', 'Lane Position'], fontsize=18)