"""
(c) 2015

author: Ghazaleh Esmaili


content:
        for each trip of each driver:
         - We select segments of size 100 starting from the first gps point of each of these trips. It is like a moving window of size 100 starting from the
           begining of the trip to len(trip)-segSize+1
         - For each of the above segments, we obtain the sparse features by applying the sparse filter calculated by unlabeled sample
         - We have (len(trip)-segSize+1)*number_of_features, we will reduce these numbers of features by deviding each trip to 10 parts and calculating the
           average features over each part so we have 10*number_of_features features at the end
         - we save pooled features of zero labeled samples in folder of each driver
"""

from modules import *
import random
from sparse_filtering_m_like import *
from my_paths import *

drivers = sorted([int(folderName) for folderName in os.listdir(driverFolder)])

trips = 200
numTrips = range(0,200)
segSize = 100
num_features = 100
desired_pool = 10
num_pool = range(0,desired_pool-1)

arraySize = desired_pool*num_features+1,trips;
feature_vector_max_final = 2*np.ones(arraySize)
feature_vector_mean_final = 2*np.ones(arraySize)


arraySize = num_features,desired_pool
feature_vector_max = np.zeros(arraySize)
feature_vector_mean = np.zeros(arraySize)


maxSzie=5000

arraySize = segSize*2, maxSzie
Tsamples = np.zeros(arraySize)

#loading the optimum weight of sparse filter obtained by unlabeled samples
L1=np.load(os.path.join(samplePath,"Sparse_filter_100_weight.npy"))

print("running this code takes some time...")
for driver in drivers:
    print("driver",driver)
    cnt = -1
    tripsData = []
    temp = []
    sample=[] 
    tripsData = np.load(os.path.join(driverFolder,str(driver),"trips.npy"))
    for trip in numTrips:
        temp = tripsData[trip]
        segments = range(1,len(temp)-segSize+2)
        arraySize = num_features,len(segments),len(numTrips)
        features = np.zeros(arraySize)
        arraySize = 2*segSize,len(segments)
        Tsamples = np.zeros(arraySize)
        pos = -1        
        for seg in segments:
            cnt = cnt+1
            pos = pos+1
            sample = temp[pos:pos+segSize,:]
            Tsamples[:,pos]=np.reshape(sample,(segSize*2,))
        # for each segment we calculate the features using sparse filter 
        features=feedForwardSF(L1,Tsamples[:,0:len(segments)+1])
        num_element = int(len(segments)/desired_pool)
        start_pos = 0
        end_pos = num_element
        #we implement pool_max and pool_mean to reduce features
        for pool in num_pool:
            feature_vector_temp_max = features[:,start_pos:end_pos]
            feature_vector_temp_mean = features[:,start_pos:end_pos]
            feature_vector_max[:,pool] = feature_vector_temp_max.max(axis=1)
            feature_vector_mean[:,pool] = feature_vector_temp_mean.mean(axis=1)
            start_pos = end_pos
            end_pos = start_pos + num_element

        feature_vector_temp_max = features[:,start_pos:len(segments)]
        feature_vector_temp_mean = features[:,start_pos:len(segments)]
        feature_vector_max[:,pool+1] = feature_vector_temp_max.max(axis=1)
        feature_vector_mean[:,pool+1] = feature_vector_temp_mean.mean(axis=1)
        feature_vector_max_final[0:desired_pool*num_features,trip] = np.reshape(feature_vector_max,(desired_pool*num_features,))
        feature_vector_mean_final[0:desired_pool*num_features,trip] = np.reshape(feature_vector_mean,(desired_pool*num_features,))
        np.save(os.path.join(driverFolder,str(driver),"one_pooled_features_of_trip_max_100.npy"),feature_vector_max_final)
        np.save(os.path.join(driverFolder,str(driver),"one_pooled_features_of_trip_mean_100.npy"),feature_vector_mean_final)      

     

