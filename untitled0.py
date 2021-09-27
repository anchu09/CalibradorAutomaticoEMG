# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 12:21:25 2021

@author: Usuario
"""

import numpy as np
x= np.loadtxt("dani1.txt", skiprows=1)
emg=x[:,-1]
print (emg)
import matplotlib.pyplot as plt
plt.plot(emg)