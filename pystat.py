#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
import math
import scipy as stats

######################################################################################################################


class RelativeFrequency(object):
    """
    Frequency Distribution:  shows the number of observations from the data set
            that fall into each classes.
    Relative Frequency Distribution: presents frequencies in terms of fractions
            or percentage.
    """

    def __init__(self, dataset):
        """
        Initialize variables.
        :param dataset: numpy array
                Raw dataset.

        Examples:
        ---------
        >>> numpy.random.randint(1,9,5)
        array([4, 4, 3, 1, 7])
        """
        self.dataset = dataset
        self.width_of_class_interval = None
        self.data_range = None
        self.number_of_intervals = None

    def get_data_range(self, number_of_intervals):
        """
        Data Range: is a list of intervals within which we need to take
            frequencies. System will set start point, end point and width
            of interval ny self.
            start point: min(dataset)
            end point: max(dataset)
        :param number_of_intervals: int
                How much intervals we need.
        :return: list
                List carrying all intervals.

        Example:
        -------
        >>> RelativeFrequency(numpy.random.randint(1,10,5)).get_data_range(3)
        array([2., 4., 6., 8.])
        """
        self.width_of_class_interval = self. get_width_of_class_interval(number_of_intervals)
        self.data_range = np.arange(min(self.dataset), max(self.dataset)+self.width_of_class_interval,
                                    self.width_of_class_interval).round(decimals=1)
        return self.data_range

    def get_custom_data_range(self, start_point, end_point, width_of_class_interval):
        """
        Data Range: is a list of intervals within which we need to take
            frequencies.
        :param start_point: int/float
                Start point of data range you want.
        :param end_point: int/float
                End point of data range you want.
        :param width_of_class_interval: int
                Width of interval.
        :return: list
                LIST of all intervals.

        Example:
        -------
        >>> RelativeFrequency(numpy.random.randint(1,10,5)).get_custom_data_range(1,10,3)
        array([ 1,  4,  7, 10])
        """
        self.data_range = np.arange(start_point, end_point+width_of_class_interval, width_of_class_interval)
        return self.data_range

    def get_width_of_class_interval(self, number_of_intervals):
        """
        Calculate the width of the class interval in a given dataset
        of the number of intervals.
        :param number_of_intervals: int
                Number of intervals we want in a given dataset.
        :return: int/float
                Possible width between the intervals.

        Example:
        -------
        >>> RelativeFrequency(numpy.random.randint(1,10,5)).get_width_of_class_interval(3)
        2.3
        """
        self.number_of_intervals = number_of_intervals
        self.width_of_class_interval = round(((math.ceil(max(self.dataset))-min(self.dataset))/(self.number_of_intervals)), ndigits=1)
        if self.width_of_class_interval < 0.2:
            self.get_width_of_class_interval(number_of_intervals-1)
        else:
            return self.width_of_class_interval

    def get_custom_width_of_class_interval(self, start_point, end_point, number_of_intervals):
        """
        Calculate width of any dataset apart from the dataset
        which we initialised with the class.
        :param start_point: int/float
                Start point of data range you want.
        :param end_point: int/float
                End point of data range you want.
        :param number_of_intervals: int
                Number of intervals we want in a dataset.
        :return: int/float
                Possible width between the intervals.

        Example:
        -------
        >>> RelativeFrequency(numpy.random.randint(1,10,5)).get_custom_width_of_class_interval(1,20,5)
        4
        """
        self.width_of_class_interval = round((end_point - start_point)/number_of_intervals)
        return self.width_of_class_interval

    def get_number_of_intervals(self, width_of_class_interval=0.0):
        """
        Calculate number of possible intervals in given dataset of
        required width.
        :param width_of_class_interval: int/float, default=0.0
                Width of interval.
        :return: int
                Number of possible intervals.

        Example:
        -------
        >>> RelativeFrequency(numpy.random.randint(1,10,5)).get_number_of_intervals()
        5
        >>> RelativeFrequency(numpy.random.randint(1,10,5)).get_number_of_intervals(3)
        2.3
        """
        self.width_of_class_interval = width_of_class_interval
        if self.width_of_class_interval == 0.0:
            return len(self.dataset)
        else:
            return round(((math.ceil(max(self.dataset))-min(self.dataset))/(width_of_class_interval)), ndigits=1)

    def frequency(self, data_range):
        """
        Calculate the frequency of the given dataset within
        the required data_range.
        Note: This function can calculate frequencies within
            closed range.

        :param data_range: list
                List of ranges within which frequency need
                to be calculated.
        :return: list/array
                List/array of int, carrying frequencies within
                data_range.

        Example:
        -------
        >>> RelativeFrequency(numpy.random.randint(1,10,5)).frequency([1,4,7,10])
        [1, 2, 2]
        """
        self.data_range = data_range
        result = []
        for i in range(len(self.data_range)-1):
            if i ==0:
                x1 = self.dataset[self.data_range[i]<=self.dataset]
                x2 = x1[x1<=self.data_range[i+1]]
                result.append(len(x2))
            else:
                x1 = self.dataset[self.data_range[i]<self.dataset]
                x2 = x1[x1<=self.data_range[i+1]]
                result.append(len(x2))
        return result

    def open_frequency(self, data_range):
        """
        Calculate the frequency of the given dataset within
        the required data_range.
        Note: This function can calculate frequencies within
            open range i.e. from -infinity to infinity.

        :param data_range: list
                List of ranges within which frequency need
                to be calculated.
        :return: list
                List of int, carrying frequencies within
                data_range.

        Example:
        -------
        >>> RelativeFrequency(numpy.random.randint(1,10,5)).open_frequency([1,4,7,10])
        [0, 2, 2, 2, 0]
        """
        self.data_range = data_range
        result = []
        for i in range(len(self.data_range)+1):
            if i == 0:
                x1 = self.dataset[self.data_range[i]>self.dataset]
                result.append(len(x1))
            elif i == len(self.data_range):
                x1 = self.dataset[self.data_range[i-1]<self.dataset]
                result.append(len(x1))
            else:
                x1 = self.dataset[self.data_range[i-1]<=self.dataset]
                x2 = x1[x1 <= self.data_range[i]]
                result.append(len(x2))

        return result

    def interval(self):
        """
        String form of closed intervals.
        :return: list of strings.

        Example:
        -------
        >>> rf = RelativeFrequency(numpy.random.randint(1,10,5))
        >>> rf_frequency = rf.frequency([1,4,7,10])
        >>> rf.interval()
        ['1-4', '4.1-7', '7-9']
        """
        result = []
        for i in range(len(self.data_range)-1):
            if i == 0:
                result.append(str(round(self.data_range[i], ndigits=1))+'-'+str(round(self.data_range[i+1], ndigits=1)))
            elif i == len(self.data_range)-2:
                result.append(str(round(self.data_range[i], ndigits=1))+'-'+str(round(max(self.dataset), ndigits=1)))
            else:
                result.append(str(round(self.data_range[i]+0.1, ndigits=1))+'-'+str(round(self.data_range[i+1], ndigits=1)))

        return result

    def open_interval(self):
        """
        String form of open intervals.
        :return: list of strings.

        Example:
        -------
        >>> rf = RelativeFrequency(numpy.random.randint(1,10,5))
        >>> rf_frequency = rf.frequency([1,4,7,10])
        >>> rf.open_interval()
        ['infinity-1', '1.1-4', '4.1-7', '7.1-10', '9-infinity']
        """
        result = []
        for i in range(len(self.data_range)+1):
            if i == 0:
                result.append('infinity-'+str(round(self.data_range[i], ndigits=1)))
            elif i == len(self.data_range):
                result.append(str(round(max(self.dataset), ndigits=1))+'-infinity')
            else:
                result.append(str(round(self.data_range[i-1]+0.1, ndigits=1))+'-'+str(round(self.data_range[i], ndigits=1)))

        return result

    def classification(self, data_range, close=True):
        """
        Classify the given dataset with frequency within
        the data_range and get display in tabular form.
        :param data_range: list
                List of ranges within which frequency need
                to be calculated.
        :param close: bool, default=True
                Signify open or closed interval frequencies.
                if close = True then disply close interval
                frequency and open interval frequencies
                otherwise.
        :return: pandas dataframe
                Display of intervals in string form and frequencies
                associated with it.

        Example:
        -------
        >>> RelativeFrequency(numpy.random.randint(1,10,5)).classification([1,4,7,10])
          Interval  Frequency    %
        0      1-4          4  0.8
        1    4.1-7          0  0.0
        2      7-8          1  0.2
        >>> RelativeFrequency(numpy.random.randint(1,10,5)).classification([1,4,7,10], close=False)
             Interval  Frequency    %
        0  infinity-1          0  0.0
        1       1.1-4          4  0.8
        2       4.1-7          1  0.2
        3      7.1-10          1  0.2
        4  7-infinity          0  0.0
        """
        if close is True:
            f = self.frequency(data_range)
            result = {
                "Interval": self.interval(),
                "Frequency": f,
                "%": np.array(f)/len(self.dataset),
            }
        else:
            f = self.open_frequency(data_range)
            result = {
                "Interval": self.open_interval(),
                "Frequency": f,
                "%": np.array(f)/len(self.dataset),
            }

        return pd.DataFrame(result)

    def evaluation(self, data_range, close=True):
        rep = self.classification(data_range, close) # change
        fig = plt.figure(figsize=(15.0, 6.0))
        axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
        axes1.plot(rep["Interval"], rep["Frequency"])

    def commulative_evaluation(self, data_range, more_than=True):
        self.data_range = data_range
        data_frequency = np.array(self.frequency(self.data_range))
        # more_than
        if more_than is True:
            total = data_frequency.sum()
            commulative_frequency_distribution = [total]
            for i in data_frequency:
                total -= i
                commulative_frequency_distribution.append(total)
        # less than
        else:
            start_total = 0
            commulative_frequency_distribution = [start_total]
            for i in data_frequency:
                start_total += i
                commulative_frequency_distribution.append(start_total)

        # presentation
        fig = plt.figure(figsize=(15.0, 6.0))
        axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
        axes1.plot(data_range, commulative_frequency_distribution)


#######################################################################################################################

class MeasureCentralTendency(RelativeFrequency):
    """
    Central tendency is the middle point of a distribution.
    Measure of central tendency is a measue of location of
    the middle point.
    """

    def __init__(self, dataset):
        """
        Initialize variables.
        :param dataset: numpy array
                Raw dataset.

        Examples:
        ---------
        >>> numpy.random.randint(1,9,5)
        array([4, 4, 3, 1, 7])
        """
        RelativeFrequency.__init__(self, dataset)
        self.data_range_mid_point = None
        self.weights = None

    def get_data_range_mid_point(self, data_range):
        """
        Calculate the mid point of the data range.
        :param data_range: list/array
                List of ranges of which middle points need
                to be calculated.
        :return: array
                array of middle points.

        Note:
        -----
        if class are given in a manner like
        [1-3,4-6,7-9,10-12,13-15,16-18,19-21,22-24]

        then, our datarange is
        [1,4,7,10,13,16,19,22,25]

        Example:
        --------
        >>> MeasureCentralTendency(numpy.random.randint(1, 10, 6)).get_data_range_mid_point([1,4,7,10])
        array([2, 6, 8])
        """
        self.data_range_mid_point = [round((data_range[i]+data_range[i+1])/2) for i in range(len(data_range)-1)]
        return np.array(self.data_range_mid_point)

    def arithmetic_mean_by_frequency_distribution(self, data_range):
        """
        Calculate A.M. from group data, by its frequency distribution.
        :param data_range: list/array
                List of ranges of which middle points need
                to be calculated.
        :return: float
                A.M.

        Example:
        --------
        >>> MeasureCentralTendency(numpy.random.randint(1,10,5)).arithmetic_mean_by_frequency_distribution([1,4,7,10])
        4.0
        """
        x = self.get_data_range_mid_point(data_range)
        f = np.array(self.frequency(data_range))
        sum_fx = (f*x).sum()
        # print("fx: ", sum_fx)
        sum_f = f.sum()
        # print("f: ", sum_f)

        return sum_fx/sum_f

    def arithmetic_mean_by_raw_dataset(self):
        """
        Calculate A.M. from ungroup data.
        :return: float
                A.M.
        Example:
        --------
        >>> MeasureCentralTendency(numpy.random.randint(1,10,5)).arithmetic_mean_by_raw_dataset()
        5.4
        """
        return self.dataset.mean()

    def weighted_mean(self, weights):
        """
        Calculated average/mean that take account the importance(weight)
        of each value to the overall total.
        :param weights: array
                Weights assigned to each observation.
        :return: float
                weighted mean
        Example:
        --------
        >>> MeasureCentralTendency(numpy.random.randint(1,10,5)).weighted_mean(numpy.array([1/2,1/3,1/4,1/5,1/6]))
        3.425287356321839
        """
        self.weights = weights
        return (self.dataset*self.weights).sum()/self.weights.sum()

    def geometric_mean(self, change_factors, increase=True):
        """
        Calculate G.M. from ungrouped data
        :param change_factors: array
                % increase or decrease in the data.
        :param increase: bool (default=True)
                if increase is true then growth is increasing
                and decreasing otherwise.
        :return: float
                G.M.

        NOTE:
        -----
        if direct % is given then, first multiply the array with 0.01

        Example:
        --------
        >>> MeasureCentralTendency(numpy.array([9, 7, 7, 8, 8])).geometric_mean(MeasureCentralTendency(numpy.array([9,
        7, 7, 8, 8])).get_change_factor(increase=True),increase=True)
        1.0923564863414776
        """
        if increase is True:
            change_factors = change_factors + 1

        product_change_factors = change_factors.cumprod()[-1]
        print("cf: ", product_change_factors)
        return product_change_factors**((1/len(change_factors)) - 1)

    def get_change_factor(self, increase=True):
        """
        % increase or decrease in the data.
        :param increase:bool (default=True)
                if increase is true then growth is increasing
                and decreasing otherwise.
        :return: array

        Example:
        --------
        >>> MeasureCentralTendency(numpy.random.randint(1,10,5)).get_change_factor(increase=True)
        array([ 0. , -0.5,  0. ,  0.5])
        >>> MeasureCentralTendency(numpy.random.randint(1,10,5)).get_change_factor(increase=False)
        array([0.66666667, 1.5       , 0.33333333, 0.5       ])
        """
        change_factors = []
        for i in range(len(self.dataset)-1):
            change_factors.append(self.dataset[i+1]/self.dataset[i])

        if increase is True:
            return np.array(change_factors) - 1
        else:
            return np.array(change_factors)

    def estimate_percentage_change(self, geometric_mean, time):
        """
        Calculate the estimation % change of any year from any given year
        geometric mean.
        :param geometric_mean: float
            G.M.
        :param time: int
            time interval upto which we need to calculate % change
        :return: float
            estimate % change

        Example:
        --------
        >>> gm = MeasureCentralTendency(numpy.array([9, 7, 7, 8, 8])).geometric_mean(MeasureCentralTendency(numpy.array(
        [9, 7, 7, 8, 8])).get_change_factor(increase=True),increase=True)
        >>> MeasureCentralTendency(numpy.array([9, 7, 7, 8, 8])).estimate_percentage_change(gm, 3)
        8.160243934535055
        """
        return (1 + geometric_mean)**(time) - 1

    def median_by_raw_data(self):
        """
        Calculate median
        :return: int/float
                median(central item in the data). Half of the items
                lies above it and half below it.
        Example:
        --------
        >>> MeasureCentralTendency(numpy.array([9, 7, 7, 8, 8])).median_by_raw_data()
        8.0
        """
        return np.median(self.dataset)

    def median_by_frequency_distribution(self, data_range):
        """
        Calculate median from grouped data
        :param data_range: list/array
                List of ranges of which middle points need
                to be calculated.
        :return: int/float
                median(central item in the data). Half of the items
                lies above it and half below it.

        Example:
        --------
        >>> data_range = MeasureCentralTendency(numpy.array([1, 6, 7, 1, 6, 3, 1, 7, 5, 1])).get_data_range(5)
        >>> MeasureCentralTendency(numpy.array([1, 6, 7, 1, 6, 3, 1, 7, 5, 1])).median_by_frequency_distribution(data_range)
        5.5
        """
        self.data_range = data_range

        f = np.array(self.frequency(self.data_range))
        cum_f = f.cumsum() # Cumulative sum of frequency
        n = np.array(f).sum()
        center_elemnt = (n+1)/2
        if n % 2 == 0:
            center1,center2 = math.floor(center_elemnt), math.ceil(center_elemnt)
            class_num = len(cum_f[cum_f<center_elemnt]) # Class index number of which median belong
            fnum = cum_f[class_num] # frequency num of which median belong
            data_range_lower_num, data_range_upper_num = self.data_range[class_num], self.data_range[class_num+1]
            step_width = (data_range_upper_num - data_range_lower_num)/fnum
            center1_item = (step_width * (center1 - 1)) + data_range_lower_num
            center2_item = (step_width * (center2 - 1)) + data_range_lower_num
            median = (center1_item + center2_item)/2

        else:
            center = center_elemnt
            class_num = len(cum_f[cum_f<center_elemnt]) # Class index number of which median belong
            fnum = cum_f[class_num] # frequency num of which median belong
            data_range_lower_num, data_range_upper_num = self.data_range[class_num], self.data_range[class_num+1]
            step_width = (data_range_upper_num - data_range_lower_num)/fnum
            median = (step_width * (center - 1)) + data_range_lower_num

        return median

    def mode_by_raw_data(self):
        """
        Calculate mode (if dataset takes smaller value having larger number of repeats)
        :return:tuple of 2 values.
                1. mode(the value that is repeated most in the data set).
                2. number of repetition.

        Example:
        --------
        >>> MeasureCentralTendency(numpy.array([1, 6, 7, 1, 6, 3, 1, 7, 5, 1])).mode_by_raw_data()
        (1, 4)
        """
        unique_data_set = np.unique(self.dataset, return_counts=True)
        return unique_data_set[0][unique_data_set[1].tolist().index(unique_data_set[1].max())], unique_data_set[1].max()

    def mode_by_frequency_distribution(self, data_range):
        """
        Calculate mode from grouped data.(if frequency take the larger value)
        :param data_range: list/array
                List of ranges of which middle points need
                to be calculated.
        :return:

        Example:
        --------
        >>> data_range = MeasureCentralTendency(numpy.array([5, 6, 5, 9, 9, 8, 1, 2, 4, 8])).get_data_range(5)
        >>> MeasureCentralTendency(numpy.array([5, 6, 5, 9, 9, 8, 1, 2, 4, 8])).mode_by_frequency_distribution(data_range)
        7.0
        """
        self.data_range = data_range
        f = np.array(self.frequency(self.data_range))
        # print("f: ", f)
        cum_f = f.cumsum()  # Cumulative sum of frequency
        n = np.array(f).sum()
        center_elemnt = (n+1)/2

        class_num = len(cum_f[cum_f<center_elemnt]) # Class index number of which median belong
        fnum = cum_f[class_num] # frequency num of which median belong
        data_range_lower_num, data_range_upper_num = self.data_range[class_num], self.data_range[class_num+1]
        # print("lower, upper: ", data_range_lower_num, data_range_upper_num)
        data_range_width = data_range_upper_num - data_range_lower_num
        d1 = f[class_num] - f[class_num+1]
        d2 = f[class_num] - f[class_num-1]
        d = d1/(d1+d2)
        # print("d:", d)
        mode = data_range_lower_num + (d*data_range_width)

        return mode


####################################################################################################################


# This library is under development. This project used only this
# part of the library that shown in it.
# 
# copyright @ Amit Sanger.
