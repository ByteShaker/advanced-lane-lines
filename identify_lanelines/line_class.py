import numpy as np

import identify_lanelines.identify_radius as identify_radius

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self,number_of_fits_in_memory=5):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # x values of the last n fits of the line
        self.recent_xfitted_top = []
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
        #number_of_fits_in_memory
        self.number_of_fits_in_memory = number_of_fits_in_memory


    def add_new_linefit(self, fitx, lane_fit, yvals):
        if len(self.recent_xfitted) <= self.number_of_fits_in_memory
            self.recent_xfitted.append(fitx)
        else:
            np.roll(self.recent_xfitted, -1, axis=0)
            self.recent_xfitted[-1,:] = fitx
        self.bestx = np.mean(self.recent_xfitted, axis=0)
        self.best_fit = np.polyfit(yvals, bestx, 2)
        self.current_fit = lane_fit #one step back

        self.radius_of_curvature = identify_radius.calc_curve_radius(fitx, yvals, max(yvals))
        self.line_base_pos = identify_radius. calc_car_2_line(fitx[-1])

        self.diffs = self.current_fit - lane_fit
        self.allx = fitx
        self.ally = yvals

        self.detected = True