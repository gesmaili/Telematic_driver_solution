# Driver Telematics 

Author:    *Ghazaleh Esmaili**


This repository contains my solution to Driver Telematics challenge https://www.kaggle.com/c/axa-driver-telematics-analysis) 


The goal of this challenge was to infer from unlabelled GPS data whether a trip was driven by a given driver or not.
The dataset was around 200*2736 trips by 2736 drivers.

***

# Summary

I used deep learning technique to solve this problem and here is my approach:

# Unsupervised learning:

1. Collecting unlabelled sample:
   - 2 trips (5 and 20) of 500 randomly selected drivers are chosen.
   - 50 segments (each segment is 100 consecutive gps points) of each of the above trips are randomly selected. 
     This gives us 50 arrays of 100*2 for each trip.
   - The above two steps provide 2*500*50 = 50000 data points where each data point is a vector of 200 (100(segment_size)*2(x and y)) elements. 
2. Applying sparse filtering to get unsupervised features
   - Sparse filtering is applied on 50000 unlabelled collected data points to get the optimum weights.
   - Number of features is set to 100 in my solution
  
#Supervised learning:


10 fold cross validation:

- Randomly choosing 20 trips of a given driver (with label one) and 20 trips of other drivers (with label 0) as test set (with no replacement)
- removing the above trips and using the rest of data points as training set (180 of a given driver (label 1) and 180 of other drivers (label 0))
- The above two steps are repeated 10 times so all data points can be used for both training and testing.
- the true error is estimated as the average error (93.7% accuracy was achieved)

# Extracting features:
   - For each trip, a sliding window of size 100 starts moving from the beginning of the gps points. It moves one step at a time. 
   - At each stop of moving window, 100 gps points are collected as a segment. For a trip of n points we will have n-100+1 overlapping segments. 
   - For each segment we apply learned feature mapping by sparse filtering to calculate features. This gives us 100 features for each segment.
   - 100 features per segments results in (n-100+1)*(100) features per trip. To reduce features, we pool features together over 10 parts of trips.
   - We divide each trip to 10 non overlapping parts. If it is not dividable we put the remaining points in the last part.
   - For each part we average over obtained features. This results in 10*100 features for each trip.
   - These features are used for supervised training.
   
Classification:

- Once features corresponding to label 1 and 0 data points are selected, they will be use for classification.
- Logistic regression (softmax regression with number of class = 2) is used for classification



# How to generate the solution

Steps to replicate my solution:

1) In mypaths.py 

   - samplePath="....\\samples" should be the destination that you want to save your unlabelled samples
   - driverFolder =  should be changed to where you saved your raw gps files "...\\driver-telematic\\drivers\\drivers"
   - resultFolder = where you want to save the final result (driver model accuracy and probability of each trip of each driver) in csv format
   
2) Run "csvtonpy.py" file

   * converts csv files to numpy array 
     
3) Run "get_unlabeled_sampling.py"

   * collects 50000 unlabelled data points (200*50000 array)		
    
4) Run "Sparse_filtering_finding_optimum_weights.py"
   
   * learns feature map (optimum weights of sparse filter) by using unlabelled data set
   
5) Run "pooled_sparse_features_one_labeled_samples"

   * collects samples of trips of each driver for supervised learning
   
6) Run "pooled_sparse_features_zero_labeled_samples"

 * collects samples of trips of other drivers for supervised learning
 
 7) Run "ClassificationStep_cross_validation"
 
 * classification using softmax regression
