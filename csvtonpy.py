"""
(c) 2015

author: this is part of a code by Janto Oellrich modified by Ghazaleh Esmaili
Converts GPS data files to .npy files and stores them in same driverfolders.

"""
import numpy as np
import os
from my_paths import *

print('\nConverting raw .csv files to numpy files...\n')

drivers = sorted([int(folderName) for folderName in os.listdir(driverFolder)])

trips = range(1, 201)

for driver in drivers:
    tripsData = []
    for trip in trips:
        tripsData.append(np.loadtxt(os.path.join(driverFolder,"{}\{}.csv".format(driver, trip)), delimiter=',', skiprows=1))

    np.save(os.path.join(driverFolder,str(driver),"trips.npy"), tripsData)
    print("Driver {} done".format(driver))

print ("Conversion completed.")

