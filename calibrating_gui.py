#!/usr/bin/env python

import tkinter as Tk
from collections import deque
from threading import Thread, Semaphore
from tkinter.ttk import Frame
import time

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
import scipy.signal as sig
from bitalino import BITalino
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import emg_features

SAMPLING_FREQ = 1000

class Window(Frame):
    def __init__(self, figure, master):
        Frame.__init__(self, master)
        self.entry = None
        self.setPoint = None
        self.master = master
        self.init_window(figure)

    def init_window(self, figure):
        self.master.title("Real Time Plot")
        canvas = FigureCanvasTkAgg(figure, master=self.master)
        canvas.get_tk_widget().pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)


class DinoApp:
    def __init__(self, root, bitalino, max_samples=5000, n_samples=100, plt_interval=100):
        # Featurizers
        self.window = sig.get_window("hamming", n_samples)
        self.featurizers = [
            np.std,
            lambda x: np.sqrt(np.sum(np.diff(x) ** 2))
        ]
        # Bitalino and buffers
        self.bitalino = bitalino
        self.n_samples = n_samples
        self.emg = deque(maxlen=max_samples)
        self.features = deque(maxlen=max_samples // n_samples)
        self.emg.extend(np.zeros(max_samples))
        self.features.extend(np.zeros((max_samples // n_samples, len(self.featurizers))))

        self.train_data = []
        # State variables
        self.training = False
        self.playing = False
        self.has_trained = False
        self.threshold = None
        self.MAX_THRESHOLD = 20
        self.pca=0
        self.scaler = MinMaxScaler()
        self.gmm=0
        # Main app
        self.plotted_signals = []
        self.figure, self.ax = self._init_figure(max_samples, n_samples, plt_interval)
        self.app = Window(self.figure, root)
        self.animated = self._animate_figure(max_samples, n_samples, plt_interval)
        self._add_buttons()
        # Semaphores for sync
        self.unsupervised_sem = Semaphore(value=0)
        self.play_sem = Semaphore(value=0)

    def compute_features(self, emg):
        features = [featurizer(emg * self.window) for featurizer in self.featurizers]
        return np.array(features)

    def do_unsupervised(self):
        while True:
            self.unsupervised_sem.acquire()
            # TODO: UNSUPERVISED ANALYSIS GOES HERE TO SET SENSIBILITY
            train_data = np.stack(self.train_data, axis=0)
            #train_data=preprocessing.minmax_scale(train_data)
            self.scaler.fit(train_data)
            self.scaler.transform(train_data)
            train_data=np.log(0.0000001+train_data)
            self.pca= PCA(n_components=1)
            self.gmm=GaussianMixture(n_components=2, covariance_type="full", n_init=100)
            train_data = self.pca.fit_transform(train_data)
            self.gmm.fit(train_data)
            if self.gmm.means_[0,0] > self.gmm.means_[1, 0]:
                self.jumplabel = 0
            else:
                self.jumplabel = 1
            print(f"Train data shape = {train_data.shape}")
            # legacy: self.threshold_bar.set(0.5)
            self.has_trained = True
            # legacy: print('new threshold is {}'.format(self.threshold_bar.get()))

    def predict(self, features):
       test_data=features.reshape(-1, len(self.featurizers))

       self.scaler.transform(test_data)
       test_data=np.log(0.0000001+test_data)
       test_data=self.pca.transform(test_data)
       predictions=self.gmm.predict(test_data)
       return predictions 


    def do_play(self):
        while True:
            self.play_sem.acquire()
            if not self.has_trained:
                Tk.messagebox.showinfo("Error!", "Model has not been trained")
                playing = False
            else:
                playing= True
            while playing:
                time.sleep(0.1)
                # LEGACY: if self.predict(self.features[-1]) >= self.threshold.get(): QUE HAGO CON ESTO??
                # TODO: this should be changed (Right now it is assuming that 1 is the jump class)
                if self.predict(self.features[-1]) == self.jumplabel:
                        pyautogui.press('space')
                        time.sleep(0.5)

    def _add_buttons(self):
        self.start_training_button = Tk.Button(self.app, text="Start training", command=self.change_training)
        self.start_playing_button = Tk.Button(self.app, text="Start playing", command=self.change_playing)
        self.restart_button = Tk.Button(self.app, text="Restart", command=self.restart)
        self.start_training_button.pack(side=Tk.BOTTOM)
        self.start_playing_button.pack(side=Tk.BOTTOM)
        self.restart_button.pack(side=Tk.BOTTOM)

        self.threshold_bar = Tk.Scale(self.app, from_=0, to=self.MAX_THRESHOLD, orient=Tk.HORIZONTAL)
        self.threshold_bar.set(self.MAX_THRESHOLD)
        self.threshold_bar.pack()

        self.app.pack()

    def restart(self):
        self.training = False
        self.has_trained = False
        self.playing = False
        self.threshold = None
        self.threshold_bar.set(self.MAX_THRESHOLD)
        self.train_data = []

    def collect(self):
        while True:
            emg_chunk = (self.bitalino.read(self.n_samples)[:, -1] - 512) / 512
            features = self.compute_features(emg_chunk)
            self.features.append(features)
            self.emg.extend(emg_chunk)
            if self.training:
                self.train_data.append(features)

    def change_playing(self):
        if self.start_playing_button.config('text')[-1] == "Start playing":
            self.playing = True
            self.start_playing_button.config(text="Stop playing")
        else:
            self.playing = False
            self.start_playing_button.config(text="Start playing")
        if self.playing:
            print('playing now!')
            self.play_sem.release()

    def change_training(self):
        if self.start_training_button.config('text')[-1] == "Start training":
            self.training = True
            self.playing = False
            self.start_training_button.config(text="Stop training")
        else:
            self.training = False
            self.start_training_button.config(text="Start training")
            self.unsupervised_sem.release()

    def _init_figure(self, max_samples, n_samples, plt_interval):
        figure = plt.figure(figsize=(10, 8))
        ax = figure.add_subplot(111)
        ax.set_xlim([0, max_samples / SAMPLING_FREQ])
        ax.set_ylim((-1, 1))
        ax.set_title('BITalino EMG')
        ax.set_xlabel("Time range (seconds)")
        ax.set_ylabel("Normalized EMG and Feature #1")
        return figure, ax

    def _animate_figure(self, max_samples, n_samples, plt_interval):
        formats = ['b-', 'r--']
        for i in range(2):
            self.plotted_signals.append(self.ax.plot([], [], formats[i])[0])

        def animate(_):
            self.plotted_signals[0].set_data(np.arange(max_samples) / SAMPLING_FREQ, self.emg)
            self.plotted_signals[1].set_data(np.arange(len(self.features)) / (SAMPLING_FREQ // n_samples),
                              [f[0] for f in self.features])
            return self.plotted_signals[0], self.plotted_signals[1]

        animated = animation.FuncAnimation(self.figure, animate, interval=plt_interval, blit=True)
        plt.legend(['nEMG', 'Energy'], loc="upper left")

        return animated


def main():
    # TODO: change BITALINO MAC here
    bitalino = BITalino('98:D3:31:FD:3B:92', None)
    bitalino.start(SAMPLING_FREQ, [0])

    tk_root = Tk.Tk()

    app = DinoApp(tk_root, bitalino, max_samples=1000, n_samples=100, plt_interval=100)

    Thread(target=app.do_unsupervised).start()
    Thread(target=app.do_play).start()
    Thread(target=app.collect).start()

    tk_root.mainloop()


if __name__ == '__main__':
    main()
