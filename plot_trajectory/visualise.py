#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np



## Read binary data
for Nr in range(9600,10000) :
    filename = "C:\\Documents and Settings\\star_%d.bin" %Nr 
    data = np.fromfile(filename,dtype='float32')
    
    ## Reshape the array into n by n
    data = np.reshape(data,(15,3))

    ## visualize the 2D array
    plt.scatter(data[:,0], data[:,1])
    plt.savefig('img.png')

