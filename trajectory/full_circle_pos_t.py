#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pylab

## Read data
filename = "data_final_2.csv"
data_x= np.loadtxt(filename, delimiter = ",", usecols= [0])
data_y= np.loadtxt(filename, delimiter= ",", usecols= [1])
data_z= np.loadtxt(filename, delimiter= ",", usecols= [2])

## visualize the 2D array
fig = plt.figure(1)
ax = fig.add_subplot(111, axisbg='black')
#pylab.ylim([0,520])
#pylab.xlim([0,520])
for i in range(0,259) :
    ax.scatter(data_x[i], data_y[i], c = "white")
    plt.savefig('img_%d.png'%(i,))
    fig.canvas.draw()
    #plt.show()