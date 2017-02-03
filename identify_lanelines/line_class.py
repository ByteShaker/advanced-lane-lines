import numpy as np

import identify_lanelines.identify_radius as identify_radius

class Lane():
    def __init__(self, number_of_fits_in_memory=100):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.x_lane_width_bottom = np.array([200,200,200])
        # x values of the last n fits of the line
        self.x_lane_width_top = np.array([200,200,200])
        # x values of the last n fits of the line
        self.x_bottom_angle = np.zeros(20)
        #Means
        self.mean_lane_width_bottom = None
        self.mean_lane_width_top = None
        self.mean_bottom_angle = 0.
        # number_of_fits_in_memory
        self.number_of_fits_in_memory = number_of_fits_in_memory

    def add_new_lanefit(self, lane_width_bottom, lane_width_top, bottom_angle):
        lane_width_bottom = np.array(lane_width_bottom, ndmin=1)
        lane_width_top = np.array(lane_width_top, ndmin=1)
        bottom_angle = np.array(bottom_angle, ndmin=1)

        if self.x_lane_width_bottom == None:
            self.x_lane_width_bottom = np.array(lane_width_bottom)
        elif self.x_lane_width_bottom.shape[0] < self.number_of_fits_in_memory:
            self.x_lane_width_bottom = np.append(self.x_lane_width_bottom, lane_width_bottom)
        else:
            self.x_lane_width_bottom = np.roll(self.x_lane_width_bottom, -1, axis=0)
            self.x_lane_width_bottom[-1] = lane_width_bottom

        if self.x_lane_width_top == None:
            self.x_lane_width_top = np.array(lane_width_top)
        elif self.x_lane_width_top.shape[0] < self.number_of_fits_in_memory:
            self.x_lane_width_top = np.append(self.x_lane_width_top, lane_width_bottom)
        else:
            self.x_lane_width_top = np.roll(self.x_lane_width_top, -1, axis=0)
            self.x_lane_width_top[-1] = lane_width_top

        if self.x_bottom_angle == None:
            self.x_bottom_angle = np.array(bottom_angle)
        elif self.x_bottom_angle.shape[0] < self.number_of_fits_in_memory:
            self.x_bottom_angle = np.append(self.x_bottom_angle, bottom_angle)
        else:
            self.x_bottom_angle = np.roll(self.x_bottom_angle, -1, axis=0)
            self.x_bottom_angle[-1] = bottom_angle

        self.mean_lane_width_bottom = np.mean(self.x_lane_width_bottom)
        self.mean_lane_width_top = np.mean(self.x_lane_width_top)
        self.mean_bottom_angle = np.mean(self.x_bottom_angle)

        self.detected = True

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self,number_of_fits_in_memory=5):
        # was the line detected in the last iteration?
        self.detected = False
        self.image_area_percentage = .6
        # x values of the last n fits of the line
        self.recent_xfitted = None
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([0, 0, 0], ndmin=1)
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
        #number_of_fits_in_memory
        self.number_of_fits_in_memory = number_of_fits_in_memory


    def add_new_linefit(self, lane_fit, yvals):

        fitx = lane_fit[0] * yvals ** 2 + lane_fit[1] * yvals + lane_fit[2]

        if self.best_fit==None:
            approved=True
        else:
            approved = self.proof_new_line_fit(lane_fit, yvals)
        #print(approved)

        if approved:
            fitx = np.array(fitx, ndmin=2)
            if self.recent_xfitted == None:
                self.recent_xfitted = fitx
            elif self.recent_xfitted.shape[0] < self.number_of_fits_in_memory:
                self.recent_xfitted = np.append(self.recent_xfitted, fitx, axis=0)
            else:
                self.recent_xfitted = np.roll(self.recent_xfitted, -1, axis=0)
                self.recent_xfitted[-1,:] = fitx

            self.bestx = np.mean(self.recent_xfitted, axis=0)

            self.best_fit = np.polyfit(yvals, self.bestx, 2)
            self.current_fit = lane_fit

            self.radius_of_curvature = identify_radius.calc_curve_radius(self.bestx, yvals, max(yvals))
            self.line_base_pos = identify_radius.calc_car_2_line(self.bestx[-1])

            #self.diffs = self.current_fit - lane_fit
            self.allx = fitx
            self.ally = yvals

            self.detected = True
            self.image_area_percentage = 1
        else:
            #self.detected = False
            #self.best_fit = None
            self.image_area_percentage = .6

    def proof_new_line_fit(self, lane_fit, yvals):
        #print(self.current_fit, lane_fit)
        self.diffs = self.best_fit - lane_fit
        #print(self.diffs)
        delta_fitx = self.diffs[0] * yvals ** 2 + self.diffs[1] * yvals + self.diffs[2]
        #print(delta_fitx)
        squared_error = np.sum(np.power(delta_fitx, 2))
        #print(squared_error)

        return squared_error < 5000000
