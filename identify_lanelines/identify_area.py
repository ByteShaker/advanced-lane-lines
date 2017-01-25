import numpy as np
import cv2
import matplotlib.pyplot as plt

import pandas as pd

import calibration.correctDistortion as correctDistortion
import image_preprocessing.combined_threshold as combined_threshold

import perspective_transform.image_transform as img_transform

import image_preprocessing.image_position as img_position

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

def create_histogram(img, y_area_of_image=[0, 1]):
    histogram = np.sum(img[img.shape[0] * y_area_of_image[0]:img.shape[0] * y_area_of_image[1], :], axis=0)

    return histogram

def identify_lane_position(histogram, rolling_window=1000, std=10.0, approx_left_lane=None, approx_right_lane=None, peak_filter_threshold=.1, peak_filter_dot_threshold=0.1, peak_filter_dot_dot_threshold=0, lane_window=100):
    histogram_df = pd.DataFrame(histogram)
    histogram_df['Peak_Filter'] = kernel_density_estimation(histogram_df, rolling_window=rolling_window, std=std)
    #histogram_df['Peak_Filter'] = histogram_df[0].rolling(window=rolling_window, axis=0, center=True, win_type='gaussian').sum(std=std)
    histogram_df['Peak_Filter_dot'] = histogram_df['Peak_Filter'].diff()
    histogram_df['Peak_Filter_dot_dot'] = histogram_df['Peak_Filter_dot'].diff()

    histogram_df['Lanes'] = np.where((histogram_df['Peak_Filter'] > peak_filter_threshold)
                                     & (histogram_df['Peak_Filter_dot'].abs() <= peak_filter_dot_threshold)
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

    return histogram_df['Lanes'], left_lane, right_lane

if __name__ == "__main__":

    # Read in an image and grayscale it
    image = cv2.imread('../test_images/test4.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = correctDistortion.correct_distortion(image)

    combined = combined_threshold.combine_mag_dir_pos(image, type = 'hls_S')

    src, dst = img_transform.perform_initial_sourcepoints()
    warped_combined = img_transform.warper(combined, src, dst)

    histogram = create_histogram(warped_combined, y_area_of_image=[2/3, 1])

    lanes, left_lane, right_lane = identify_lane_position(histogram)

    print(left_lane, right_lane)

    #Todo: Laufe entlang eines Bildes und identifiziere alle Lanepunkte ->
    identified_left_curve_area = np.zeros_like(warped_combined)
    identified_right_curve_area = np.zeros_like(warped_combined)
    img_shape = warped_combined.shape

    img_areas = 10
    for i in range(img_areas):
        histogram = create_histogram(warped_combined, y_area_of_image=[(img_areas-(i+1))/img_areas, (img_areas-i)/img_areas])
        #histogram_df = pd.DataFrame(histogram)
        #histogram_df['Peak_Filter'] = histogram_df[0].rolling(window=200, axis=0, center=True, win_type='gaussian').mean(std=10.0)
        #histogram_df['Peak_Filter'].plot()
        temp_lane_window = 100 + ((i / img_areas) * 100)
        lanes, left_lane, right_lane = identify_lane_position(histogram, std=10, approx_left_lane=left_lane, approx_right_lane=right_lane, peak_filter_threshold=0, lane_window=temp_lane_window)
        print(left_lane, right_lane)

        left_left_border, left_right_border = border_control(left_lane, temp_lane_window, 0, img_shape[1])
        temp_left_area = np.array([[(left_left_border, img_shape[0]*((img_areas-i)/img_areas)),
                                    (left_left_border, img_shape[0]*((img_areas-(i+1))/img_areas)),
                                    (left_right_border, img_shape[0]*((img_areas-(i+1))/img_areas)),
                                    (left_right_border, img_shape[0]*((img_areas-i)/img_areas))]], dtype=np.int32)
        temp_left_img_position = img_position.position_select(warped_combined, temp_left_area)

        identified_left_curve_area[((identified_left_curve_area == 1) | (temp_left_img_position == 1))] = 1


        right_left_border, right_right_border = border_control(right_lane, temp_lane_window, 0, img_shape[1])
        temp_right_area = np.array([[(right_left_border, img_shape[0] * ((img_areas - i) / img_areas)),
                                    (right_left_border, img_shape[0] * ((img_areas - (i + 1)) / img_areas)),
                                    (right_right_border, img_shape[0] * ((img_areas - (i + 1)) / img_areas)),
                                    (right_right_border, img_shape[0] * ((img_areas - i) / img_areas))]], dtype=np.int32)
        temp_right_img_position = img_position.position_select(warped_combined, temp_right_area)

        identified_right_curve_area[((identified_right_curve_area == 1) | (temp_right_img_position == 1))] = 1

        #f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 9))
        #y_area_of_image = [(img_areas - (i + 1)) / img_areas, (img_areas - i) / img_areas]
        #ax1.imshow(warped_combined[warped_combined.shape[0] * y_area_of_image[0]:warped_combined.shape[0] * y_area_of_image[1], :], cmap='gray')
        #ax1.set_title('Original Image', fontsize=20)
        #pd.DataFrame(histogram).plot(ax=ax2)
        #ax2.set_title('Cobined Threshold', fontsize=20)
        #lanes.plot(ax=ax3)
        #ax3.set_title('Cobined Threshold', fontsize=20)
        #plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
        #plt.show()

    #identified_curve_area[((identified_curve_area == 1) & (warped_combined == 1))] = 1


    #histogram_df['left_Lane'].plot()

    # Plot the result
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
    #f.tight_layout()
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(identified_right_curve_area, cmap='gray')
    ax2.set_title('Cobined Threshold', fontsize=20)
    ax3.imshow(identified_left_curve_area, cmap='gray')
    ax3.set_title('Transformed', fontsize=20)
    lanes.plot.area(ax=ax4, stacked=True)
    ax4.set_title('Lane Position', fontsize=20)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)

    plt.show()
