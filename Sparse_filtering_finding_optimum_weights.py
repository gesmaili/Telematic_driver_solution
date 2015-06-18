"""
(c) 2015

author: Ghazaleh Esamili
In this code we find the optimum W of sparse filter for unlabeled samples
    
"""

import numpy as np
import random
from sparse_filtering_m_like import *
import os
from my_paths import *

trips = 200
numDrivers = range(1,501)
numTrips = range(1,3)
segments = range(1,51)
segSize = 100

##########################

data = np.load(os.path.join(samplePath,"Usamples.npy"))
data -= data.mean(axis=0)
# Train     
L1_size = 100 # number of features
print("Calculating optimum weight, it will take some time")
L1 = sparseFiltering(L1_size, data)
    
features=feedForwardSF(L1,data)
np.save(os.path.join(samplePath,"Sparse_filter_100_weight.npy"),L1)

                
    

