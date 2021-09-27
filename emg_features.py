#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sig

def zero_crossing_rate(x): 
    reference = x[0]
    count = 0
    for current in x[1:]:
        # Check if there is a change of sign when comparing reference and current
        if  current * reference < 0:
            # update current and count
            count += 1
            reference = current
    return count


def crossing_rate(x, reference_level):
    return zero_crossing_rate(x - reference_level)


def turns_count(x, reference_level, return_peaks=False):
    dx = np.diff(x)
    detected_peaks = np.logical_and(dx[:-1] * dx[1:] < 0, x[1:-1] > reference_level)
    if return_peaks:
        # 1 is for correcting the fact that sample 0 in x[1:-1] corresponds to sample 1 in the whole signal x
        return 1 + np.where(detected_peaks)[0]
    else:
        return np.sum(detected_peaks)


class WaveRectifier:
    """WARNING: intended ONLY for real time filtering with NON-OVERLAPPING windows"""
    def __init__(self, rectification="full", fc=0.02):
        if rectification != "full" and rectification != "half":
            raise ValueError("Invalid rectification parameter. Use 'full' or 'half'")
        if not (0 < fc < 1):
            raise ValueError("Invalid cut-off frequency (should be 0 < fc < 1)")
        self.rectification = rectification
        self.fc = fc

        if rectification == "full":
            self.rectify = abs
        else:
            self.rectify = lambda x: np.clip(x, 0, np.inf)

        self._filter = sig.butter(5, fc)
        b, a = self._filter
        self._z = np.zeros(max(len(a), len(b)) - 1)

    def __call__(self, x):
        y, self._z = sig.lfilter(*self._filter, self.rectify(x), zi=self._z)
        return y


def rollapply(x, fun, window, by=1, fs=1.):
    """
    Applies a function to a signal in a moving window.
    :param x: The input signal.
    :param fun: The function to be applied in each of the windows. It should take an input signal as argument.
    :param window: A window function as returned by scipy.signal.get_window
    :param by: calculate 'fun' at every 'by'-th time point
    :param fs: sampling frequency of 'x'
    :return: a tuple consisting of time, result
    """
    result = []
    time = []
    # The original time at which the input signal x was recorded
    otime = np.arange(len(x)) / fs
    nb_windows = np.floor((len(x) - len(window)) / by).astype('int') + 1
    for window_index in range(nb_windows):  # Do not permit the loop to go out of range!!
        # The samples within the window are
        indexes = range(window_index * by, window_index * by + len(window))
        samples_within_window = x[indexes]
        time_within_window = otime[indexes]
        # COMPLETE: multiply the samples in the window by the window shape and apply the function
        # fun to the resulting vector.
        result.append(fun(samples_within_window * window))
        # We associate a single time of occurrence to each window. A usual approach
        # is to pick the central time of all the samples within the window
        time.append(np.mean(time_within_window))
    return time, np.array(result)

