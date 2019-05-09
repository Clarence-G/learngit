# -*- coding: utf-8 -*-
 
import numpy as np
 
def unpickle(file):
    #import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
 
#load dataset of cifar10
def load_CIFAR10(cifar10_dir):
    #get the training data
    X_train = []
    y_train = []
    for i in range(1,6):
        dic = unpickle(cifar10_dir+"\\data_batch_"+str(i))
        for item in dic["data"]:
            X_train.append(item)
        for item in dic["labels"]:
            y_train.append(item)
            
    #get test data
    X_test = []
    y_test = []
    #do not know why the path is not just right as above,add a extra\
    dic = unpickle(cifar10_dir+"\\test_batch")
    for item in dic["data"]:
       X_test.append(item)
    for item in dic["labels"]:
       y_test.append(item)
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test
 
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):  
    """ 
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare 
    it for the linear classifier. These are the same steps as we used for the SVM, 
    but condensed to a single function.  
    """  
    # Load the raw CIFAR-10 data 
    cifar10_dir = 'E:\python\cs231n\cifar-10-batches-py'   # make a change
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)  
    # Subsample the data    
    mask = range(num_training, num_training + num_validation)    
    X_val = X_train[mask]                   # (1000,32,32,3)    
    y_val = y_train[mask]                   # (1000L,)   
    mask = range(num_training)    
    X_train = X_train[mask]                 # (49000,32,32,3)    
    y_train = y_train[mask]                 # (49000L,)    
    mask = range(num_test)   
    X_test = X_test[mask]                   # (1000,32,32,3)    
    y_test = y_test[mask]                   # (1000L,)    
 
    # preprocessing: subtract the mean image    
    mean_image = np.mean(X_train, axis=0)    
    X_train -= mean_image   
    X_val -= mean_image    
    X_test -= mean_image    
 
    # Reshape data to rows    
    X_train = X_train.reshape(num_training, -1)      # (49000,3072)    
    X_val = X_val.reshape(num_validation, -1)        # (1000,3072)    
    X_test = X_test.reshape(num_test, -1)            # (1000,3072)   
    
    data = {}
    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_val'] = X_val
    data['y_val'] = y_val
    data['X_test'] = X_test
    data['y_test'] = y_test
 
    return data