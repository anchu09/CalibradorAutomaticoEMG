#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

import emg_features


def read_emg(filename, delimiter=None):
    emg = np.loadtxt(filename, delimiter=delimiter, skiprows=1)[:, -1]
    # Since Bitalino uses 10 bits, the emg goes from 0 to 1024 and it is centered around 512.
    # We center it around 0 and scale it so that it is between -1 and 1
    return (emg - 512) / 512


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Featurize and EMG')
    parser.add_argument("--filename", type=str,
                        help="CSV with the EMG samples from the BITALINO")
    parser.add_argument("--window_size", type=int, default=100,
                        help="Window size in samples for rollapplying")
    parser.add_argument("--reference_level", type=float, default=0.2,
                        help="Reference level for threshold-based featurizers")
    args = parser.parse_args()

    window = sig.get_window("hamming", args.window_size)
    full_wr = emg_features.WaveRectifier("full")
    half_wr = emg_features.WaveRectifier("half")
    featurizers = [
        np.std,
        lambda x: emg_features.crossing_rate(x, reference_level=args.reference_level),
        lambda x: emg_features.turns_count(x, reference_level=args.reference_level),
        lambda x: full_wr(x)[-1],
        lambda x: half_wr(x)[-1],
        lambda x: np.sqrt(np.sum(np.diff(x) ** 2))
    ]

    emg = read_emg(args.filename)
    features = [emg_features.rollapply(emg, featurizer, window, by=len(window))[1] for featurizer in featurizers]
    features = np.column_stack(features)
    np.savetxt(args.filename.split(".txt")[0] + "_features.txt", features)

    plt.figure(figsize=(16, 9))
    plt.subplot(3, 1, 1)
    plt.plot(emg)
    plt.plot(np.linspace(0, len(emg), len(features)), features)
    plt.subplot(3, 1, 2)
    plt.hist(np.log(features[:, -1]), bins=20)
    plt.subplot(3, 1, 3)
    plt.scatter(np.log(features[:, 0]), np.log(features[:, -1]))
    plt.show()
