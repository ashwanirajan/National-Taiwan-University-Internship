#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

n=512

## Read binary data
for i in range(1,100) :
    fname = 'potential_t_%d.bin'%i
    data = np.fromfile(fname, dtype='double')  
    ## Reshape the array into n by n
    data = np.reshape(data,(n,n))

    ## visualize the 2D array

    plt.imshow(data, cmap='gnuplot')
    plt.savefig('potential_%d.png'%i)
    #plt.show()
