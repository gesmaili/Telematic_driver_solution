"""
(c) 2015

author: Ghazaleh Esmaili

content: classification using 10 fold cross validation and softmax regression
on pooled features
 
"""
from My_softmax_regression import *
import math
import random
from  writetocsv import *
from my_paths import *

###########################################################################################

drivers = sorted([int(folderName) for folderName in os.listdir(driverFolder)])

    
""" Initialize parameters of the Regressor """
    
input_size     = 1000    # input vector size
num_classes    = 2       # number of classes
lamda          = 0.00008 # Regularization
max_iterations = 400     # number of optimization iterations
perc = 0.9               # percentage of the data that is used for training
test_perc = 0.1          # percentage of the data that is used for test for 10 fold cross validation
feature_numbers = 1000   # number of pooled features for each trip
max_number_trip = 200    # total number of trips
trip_range = range(1,201)

list_results =[]         # final list that will be used to write submission format csv file for all drivers
list_results.append(['driver','prob'])

list_results_accuracy =[]         # final list that will be used to write submission format csv file for all drivers
list_results_accuracy.append(['driver','score'])

driver = 1
# loading the pooled features of each trip just to get the size of training and test sets
feature_vector_max_final1=numpy.load(os.path.join(driverFolder,str(driver),"one_pooled_features_of_trip_mean_100.npy"))
feature_vector_max_final2=numpy.load(os.path.join(driverFolder,str(driver),"zero_pooled_features_of_trip_mean_100.npy"))

max_number_training = len(feature_vector_max_final1[0])
max_number_test = len(feature_vector_max_final2[0])

# calculating training and test numbers
training_number = math.ceil(max_number_training*perc)
test_number = math.ceil(max_number_test*test_perc)

# initializing arrays 
arraySize = feature_numbers,2*training_number  
training_data = numpy.zeros(arraySize)
arraySize = 2*training_number 
training_labels = numpy.zeros(arraySize)

arraySize = feature_numbers,2*test_number   
test_data = numpy.zeros(arraySize)
arraySize = 2*test_number
test_labels = numpy.zeros(arraySize)

# number of folds for cross validation
trial_range = range(0,10)
trial_number = 10


test_accuracy = numpy.zeros(trial_number)
training_accuracy = numpy.zeros(trial_number)
probabiliy= numpy.zeros([200,trial_number])
arraySize = 2000,trial_number
vopt_theta = numpy.zeros(arraySize)

arraySize = len(drivers)
bestofDriver = numpy.zeros(arraySize)

#initiating empty list for cross_validation results
cross_validation_of_driver=[]
cnt = -1
rcnt = -1

for driver in drivers:
    print("driver", driver)
    cnt = cnt+1
    # pooled features of all trips of the given driver is loaded
    feature_vector_max_final1=numpy.load(os.path.join(driverFolder,str(driver),"one_pooled_features_of_trip_mean_100.npy"))
    feature_vector_max_final1_orig = feature_vector_max_final1.copy()
    driver_trips = feature_vector_max_final1[0:feature_numbers,:]
    # pooled features of all trips of the non given driver is loaded
    feature_vector_max_final2=numpy.load(os.path.join(driverFolder,str(driver),"zero_pooled_features_of_trip_mean_100.npy"))
    feature_vector_max_final2_orig = feature_vector_max_final2.copy()
    

    ##################### in each try 10 percent of data is selected as test and will be removed from data to have non overlapping folds ##########
    pos_orig = numpy.arange(200)
    pos_orig_new =[]
    location =[]
    
    for trys in trial_range:
        if trys == 0:
            rpos1 = random.sample(range(0, max_number_trip), test_number)
            for i in rpos1:
                location.append(i)
        else:
            pos_orig_new.append(numpy.delete(pos_orig,location,0))
            index_temp = random.sample(range(0, len(pos_orig_new[trys-1])), test_number)
            element = pos_orig_new[trys-1]
            rpos1 = [element[i] for i in index_temp]
            for i in rpos1:
                location.append(i)

             
        test_data[0:feature_numbers,0:math.ceil(test_number/1)] = feature_vector_max_final1[0:feature_numbers,rpos1]
        test_data[0:feature_numbers,math.ceil(test_number/1):test_number*2] = feature_vector_max_final2[0:feature_numbers,rpos1]
        test_labels[0:math.ceil(test_number/1)] = feature_vector_max_final1[feature_numbers,rpos1]
        test_labels[math.ceil(test_number/1):2*test_number] = feature_vector_max_final2[feature_numbers,rpos1]

        test_labels = test_labels-1
        removed_test = numpy.delete(feature_vector_max_final1,rpos1,1)
        rpos2 = random.sample(range(0, len(removed_test[0])), training_number)
        

        training_data[0:feature_numbers,0:math.ceil(training_number/1)] = feature_vector_max_final1[0:feature_numbers,rpos2]
        training_data[0:feature_numbers,math.ceil(training_number/1):training_number*2] = feature_vector_max_final2[0:feature_numbers,rpos2]
        training_labels[0:math.ceil(training_number/1)] = feature_vector_max_final1[feature_numbers,rpos2]
        training_labels[math.ceil(training_number/1):2*training_number] = feature_vector_max_final2[feature_numbers,rpos2]

        training_labels = training_labels-1

        ###### shuffling data order in training set ##############
        index = numpy.random.permutation(training_data.shape[1])
        training_data = training_data[:, index]
        training_labels = training_labels[index]
        ######## shuffling data order in test set ##############
        index = numpy.random.permutation(test_data.shape[1])
        test_data = test_data[:, index]
        test_labels = test_labels[index]
        
           
        """ Initialize Softmax Regressor """
            
        regressor = SoftmaxRegression(input_size, num_classes, lamda)
           
        """ Run the L-BFGS algorithm to get the optimal parameter values for training data """
            
        opt_solution  = scipy.optimize.minimize(regressor.softmaxCost, regressor.theta, 
                                                    args = (training_data, training_labels,), method = 'L-BFGS-B', 
                                                    jac = True, options = {'maxiter': max_iterations})
        vopt_theta[:,trys]     = opt_solution.x
               
        """ Obtain predictions for test data """
            
        [predictions, probabilities] = regressor.softmaxPredict(vopt_theta[:,trys] , test_data)
            
        """ calculating accuracy of the test data """
            
        
        predictions1=numpy.reshape(predictions,(test_number*2,))
        correct = test_labels == predictions1
        test_accuracy[trys] = numpy.mean(correct)

        [all_predictions, all_probabilities] = regressor.softmaxPredict(vopt_theta[:,trys], driver_trips)
        all_predictions = numpy.reshape(all_predictions,(200,))
        probabiliy[:,trys] = all_probabilities[1]
        
    ############# adding the result to the result list #######################
        
    cross_validation_of_driver.append(numpy.mean(test_accuracy))
    
    ############### saving result of each driver in her folder ################
    
    numpy.save(os.path.join(driverFolder,str(driver),"cross_validation_results.npy"),test_accuracy)
    numpy.save(os.path.join(driverFolder,str(driver),"mean_cross_validation_results.npy"),numpy.mean(test_accuracy))
    print("test_accuracy",numpy.mean(test_accuracy))
    
    #######################################
    

    average_trip_prob = numpy.mean(probabiliy,axis=1)    
    for Dtrip in trip_range:
        rcnt = rcnt+1
        driver_name = str(driver)+'_'+str(Dtrip)
        list_results.append([driver_name,str(average_trip_prob[Dtrip-1])])
    list_results_accuracy.append([str(driver),str(numpy.mean(test_accuracy))])
            
    ######## saving and writing  the result of all drivers in csv file ############
        
write_results_to_csv_file(list_results,"C:\\Python34\\results\\Telematic_results_cross_validation.csv")
write_results_to_csv_file(list_results_accuracy,"C:\\Python34\\results\\Telematic_results_cross_validation_accuracy.csv")   
numpy.save(os.path.join(samplePath,"cross_validation_results_ofDrivers.npy"),cross_validation_of_driver)    

