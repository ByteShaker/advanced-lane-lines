##Writeup Friedrich Schweizer / Advanced Lane Lines

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_calibration1.jpg "Undistorted"
[image2]: ./output_images/original_undistorted.jpg "Road Transformed"
[image3]: ./output_images/transformed_image.jpg "Binary Example"
[image4]: ./output_images/process_2_binary.jpg "Warp Example"
[image5]: ./output_images/Identified_line_AREA.jpg "Fit Visual"
[image6]: ./output_images/test6_applied_lane_lines.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained the file called `./calibration/correctDistortion.py`).  

In first instance I have a look, if I already calibrated the camera in a frame ahead or in a run with the same camera ahead. If so I just take the distortion coefficients, if not, I go on and calibrate the camera using the function `cal_mtx_dist() in the file called `./calibration/calibrateCamera.py` 

Here I first have a look for all prepared calibration images the folders. For every single Image I prepare the "object points" and "image points" referencing to the individual number of points in the function `define_points(images, nx_ny_list)`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

If this worked, I save the camera calibration and distortion coefficients to a pickle file.

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

####2. I transformed my Image first. Therefore I wrote a dynamic transform function, which includes a detection for parallelity in the resulting lane lines.

The code for my perspective transform includes a function called `warper()`, which appears in lines 50 through 55 in the file `lane_line_main.py` (output_images/examples/example.py).  
The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. These points get dynamicly calculated, depending on the width of the detected lane in the bottom and top. Also the angle in the bottom calculates a correction, so the bottom lanes are almost vertical in the warped image.

```
# Dynamic Transform of Image Image
lane_width_bottom, lane_width_top, bottom_angle = img_transform.calc_correct_transform(left_line, right_line, look_at_image_area_percentage, img_shape)
if (lane_width_bottom != None) & (lane_width_top != None):
    master_lane.add_new_lanefit(lane_width_bottom, lane_width_top, bottom_angle)
src, dst = img_transform.calc_new_sourcepoints(master_lane.mean_lane_width_bottom, master_lane.mean_lane_width_top, master_lane.mean_bottom_angle)
warped_image = img_transform.warper(raw_image, src, dst)
 
 ------------------------------
def calc_new_sourcepoints(lane_width_bottom=None, lane_width_top=None, bottom_angle=0., img_shape=(720,1280)):
    inital_px_top = 70
    px_shift = img_shape[0] * bottom_angle
    image_middle = int(img_shape[1]/2) - px_shift
    if (lane_width_bottom==None) | (lane_width_top==None):
        left_pos_top = image_middle - int(inital_px_top / 2)
        right_pos_top = image_middle + int(inital_px_top / 2)
    else:
        new_px_top = (lane_width_top / lane_width_bottom) * inital_px_top
        left_pos_top = image_middle - int(new_px_top / 2)
        right_pos_top = image_middle + int(new_px_top / 2)

    src = np.float32(
        [[left_pos_top, img_shape[0]*(440/720)],
         [img_shape[1]*(210/1280), img_shape[0]*(705/720)],
         [img_shape[1]*(1070/1280), img_shape[0]*(705/720)],
         [right_pos_top, img_shape[0]*(440/720)]])

    dst = np.float32(
        [[img_shape[1]*(540/1280), 0],
         [img_shape[1]*(540/1280), img_shape[0]],
         [img_shape[1]*(740/1280), img_shape[0]],
         [img_shape[1]*(740/1280), 0]])

    return (src, dst)

```
This resulted in the following initial source and destination points for the project_video:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 210, 440      | 540, 0        | 
| 210, 705      | 540, 720      |
| 1070, 705     | 740, 720      |
| 1070, 440      | 740, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

####3. Thresholded binary image.

In the file `combined_threshold.py` the function `ombined_thresholds_complete(image, verbose=False)` applies all transforms to create the binary image.First 
First I converted the Image to HSV Colorspace. Here I used the V layer to multiple apply gamma correction and histogram equalization. This higlights all areas with light colors.
With the new V layer I merge the Image back to HSV and convert it to BGR again.
I am using Magnitude and Direction thresholding on the new V_layer to identify laneline segments.
```
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_h, hsv_s, hsv_v = cv2.split(hsv)

    hsv_gamma_equal_dark = gamma_equalize(hsv_v, 3, 8)
    #hsv_gamma_equal_light = gamma_equalize(hsv_v, 3, .8)

    hsv = cv2.merge((hsv_h, hsv_s, hsv_gamma_equal_dark))
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    mag_binary_V = img_gradient.mag_thresh(hsv_gamma_equal_dark, 7, (40, 255))
    dir_binary_V = img_gradient.dir_threshold(hsv_gamma_equal_dark, 5, (0 * np.pi / 180, 55 * np.pi / 180))
    combo_complete_dark = cv2.bitwise_and(dir_binary_V, mag_binary_V)
```
Following you see the process in pictures:

![alt text][image4]



####4. Fitting Laneline on Binary Image

To fit the Laneline on a binary i first have a look, if I already detected a valid Laneline in last step.
If not I devide the lower part of my binary image in 5 tiles and do a histogram analysis on each tile:

```
def identify_lane_position(histogram, rolling_window=1000, std=10.0, approx_left_lane=540, approx_right_lane=740, peak_filter_threshold=10, peak_filter_dot_threshold=20., peak_filter_dot_dot_threshold=0, lane_window=120, verbose=False):
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
```

This gives me 5 valid points, which are in the shape of the Laneline. I fit a polyline trough those 5 points.

With this polyline I define the shape of the area in the image where my lines are you will find the code in file `image_position.py` in function `def perform_lane_position(img)`see lines 58 to 113
The defined shapes for left and right line are used to find all pixels, which belong to the lanelines.
These pixels are used to fit perfect polylines to the lanelines. file: `def fit_lane_line(lane_img):` function: `def fit_lane_line(lane_img)` line 7-16

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 38 through 46 in my code in `identify_radius.py`:
Here I first transform the x and y values of my fitted polyline from pixels to meter.
Then I fit a new Polyline on these parameters and us the math to calculate the radius at the bottom of th image (car position)

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `identify_radius.py` in the function `map_create_fitted_lane_img(left_line_fit, right_line_fit, yvals)` line 77 to 91.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The algorithm performs super smooth.
Here's a [link to my video result](./project_video_calc.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had no bigger issues. 
But I got lost in optimizing the algorithm to perform on every kind of distortion, angle shifts and street curvature. Spended way to much time on this project.
Therefore I used a lot of math to calculate the transform parameters and the pixel detection shape.

Pipeline is pretty robust, if I get lost of lines in 5 measures the algorithm will start over new.
Single wrong measures are dropped.
If there is rain or snow, I Think my image preprocessing will not be the best in place.

I had a lot of fun and will improve this algorithm further as soon as I am done with the next project.
