"""
(c) 2015

author: Ghazaleh Esamili

This code collects 50 segmnets (each 100 gps points) of two randomly selected trips of 500 randomly selected drivers
for Unlabeled sample collection

"""
import numpy as np
import os
import random
from my_paths import *

drivers = sorted([int(folderName) for folderName in os.listdir(driverFolder)])

trips = 200                  #total number of trips for each driver
numDrivers = range(1,501)    # number of randomely selected drivers
numTrips = range(1,3)        # number of randomely selected trips
segments = range(1,51)       # number of segments to be collected from each trip
segSize = 100                # size of the segments


# initializing array
arraySize = 2*segSize,len(numDrivers)*len(numTrips)*len(segments);
Usamples = np.zeros(arraySize)

cnt = -1;
for numDriver in numDrivers:   
    DriverIndex = random.randint(0,len(drivers)-1)
    driver = drivers[DriverIndex]
    print("selected driver",driver)
    tripsData = []
    temp = []
    sample=[]
    tripsData = np.load(os.path.join(driverFolder,str(driver),"trips.npy"))
    for trip in numTrips:
        tripIndex = random.randint(0,trips-1)
        temp = tripsData[tripIndex]
        for seg in segments:
            cnt = cnt+1;
            pos = random.randint(0,len(temp)-segSize-1)
            sample = temp[pos:pos+segSize,:]
            Usamples[:,cnt]=np.reshape(sample,(segSize*2,)).copy()

np.save(os.path.join(samplePath,"Usamples.npy"),Usamples)
