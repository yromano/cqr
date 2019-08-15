#!/usr/bin/env python
# coding: utf-8

import os
import torch
import random
import numpy as np
np.warnings.filterwarnings('ignore')

from datasets import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd

# for MEPS
def condition(x, y=None):
    return int(x[0][-1]>0)

from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.nc import SignErrorErrFunc
from nonconformist.nc import QuantileRegAsymmetricErrFunc


def append_statistics(coverage_sample,
                      length_sample,
                      method_name,
                      dataset_name_vec,
                      method_vec,
                      coverage_vec,
                      length_vec,
                      seed_vec,
                      test_ratio_vec,
                      seed,
                      test_ratio,
                      dataset_name_group_0,
                      dataset_name_group_1):
    
    dataset_name_group = [dataset_name_group_0, dataset_name_group_1]
    
    for group_id in range(len(dataset_name_group)):
        
        coverage = (coverage_sample[group_id]).astype(np.float)
        length = length_sample[group_id]
        
        for i in range(len(coverage)):
            dataset_name_vec.append(dataset_name_group[group_id])
            method_vec.append(method_name)
            coverage_vec.append(coverage[i])
            length_vec.append(length[i])
            seed_vec.append(seed)
            test_ratio_vec.append(test_ratio)
        

def run_equalized_coverage_experiment(dataset_name, method, seed, save_to_csv=True, test_ratio = 0.2):

    random_state_train_test = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if os.path.isdir('/scratch'):
        local_machine = 0
    else:
        local_machine = 1
        
    if local_machine:
        dataset_base_path = '/Users/romano/mydata/regression_data/'
    else:
        dataset_base_path = '/scratch/users/yromano/data/regression_data/'
        
    # desired miscoverage error
    alpha = 0.1
    
    # desired quanitile levels
    quantiles = [0.05, 0.95]
    
    # name of dataset
    dataset_name_group_0 = dataset_name + "_non_white"
    dataset_name_group_1 = dataset_name + "_white"
    
    
    # load the dataset
    X, y = datasets.GetDataset(dataset_name, dataset_base_path)
    
    # divide the dataset into test and train based on the test_ratio parameter
    x_train, x_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_ratio,
                                                        random_state=random_state_train_test)
    

    # In[2]:
    
    # compute input dimensions
    n_train = x_train.shape[0]
    in_shape = x_train.shape[1]
    
    # divide the data into proper training set and calibration set
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]
    
    # zero mean and unit variance scaling 
    scalerX = StandardScaler()
    scalerX = scalerX.fit(x_train[idx_train])
    
    # scale
    x_train = scalerX.transform(x_train)
    x_test = scalerX.transform(x_test)
    
    y_train = np.log(1.0 + y_train)
    y_test = np.log(1.0 + y_test)
    
    # reshape the data
    x_train = np.asarray(x_train)
    y_train = np.squeeze(np.asarray(y_train))
    x_test = np.asarray(x_test)
    y_test = np.squeeze(np.asarray(y_test))
    
    # display basic information
    print("Dataset: %s" % (dataset_name))
    print("Dimensions: train set (n=%d, p=%d) ; test set (n=%d, p=%d)" % 
          (x_train.shape[0], x_train.shape[1], x_test.shape[0], x_test.shape[1]))
    
    
    # In[3]:
    dataset_name_vec = []
    method_vec = []
    coverage_vec = []
    length_vec = []
    seed_vec = []
    test_ratio_vec = []
    
        
    if method == "net":
        
        # pytorch's optimizer object
        nn_learn_func = torch.optim.Adam
        
        # number of epochs
        epochs = 1000
        
        # learning rate
        lr = 0.0005
        
        # mini-batch size
        batch_size = 64
        
        # hidden dimension of the network
        hidden_size = 64
        
        # dropout regularization rate
        dropout = 0.1
        
        # weight decay regularization
        wd = 1e-6
            
        # ratio of held-out data, used in cross-validation
        cv_test_ratio = 0.1
        
        # seed for splitting the data in cross-validation.
        # Also used as the seed in quantile random forests function
        cv_random_state = 1
        
        
        # In[4]:
        
        model = helper.MSENet_RegressorAdapter(model=None,
                                   fit_params=None,
                                   in_shape = in_shape,
                                   hidden_size = hidden_size,
                                   learn_func = nn_learn_func,
                                   epochs = epochs,
                                   batch_size=batch_size,
                                   dropout=dropout,
                                   lr=lr,
                                   wd=wd,
                                   test_ratio=cv_test_ratio,
                                   random_state=cv_random_state)
        
        nc = RegressorNc(model, SignErrorErrFunc())

        y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_test, idx_train, idx_cal, alpha)
        
        method_name = "Marginal Conformal Neural Network"
        
        # compute and print average coverage and average length
        coverage_sample, length_sample = helper.compute_coverage_per_sample(y_test,
                                                                            y_lower,
                                                                            y_upper,
                                                                            alpha,
                                                                            method_name,
                                                                            x_test,
                                                                            condition)
        
        
        append_statistics(coverage_sample,
                          length_sample,
                          method_name,
                          dataset_name_vec,
                          method_vec,
                          coverage_vec,
                          length_vec,
                          seed_vec,
                          test_ratio_vec,
                          seed,
                          test_ratio,
                          dataset_name_group_0,
                          dataset_name_group_1)
                
        
        # In[]
        
        model = helper.MSENet_RegressorAdapter(model=None,
                                               fit_params=None,
                                               in_shape = in_shape,
                                               hidden_size = hidden_size,
                                               learn_func = nn_learn_func,
                                               epochs = epochs,
                                               batch_size=batch_size,
                                               dropout=dropout,
                                               lr=lr,
                                               wd=wd,
                                               test_ratio=cv_test_ratio,
                                               random_state=cv_random_state)
        nc = RegressorNc(model, SignErrorErrFunc())
        
        y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_test, idx_train, idx_cal, alpha, condition)
        
        method_name = "Conditional Conformal Neural Network (joint)"
        
        # compute and print average coverage and average length
        coverage_sample, length_sample = helper.compute_coverage_per_sample(y_test,
                                                                            y_lower,
                                                                            y_upper,
                                                                            alpha,
                                                                            method_name,
                                                                            x_test,
                                                                            condition)
        
        
        append_statistics(coverage_sample,
                          length_sample,
                          method_name,
                          dataset_name_vec,
                          method_vec,
                          coverage_vec,
                          length_vec,
                          seed_vec,
                          test_ratio_vec,
                          seed,
                          test_ratio,
                          dataset_name_group_0,
                          dataset_name_group_1)
        
        # In[6]
        
        category_map = np.array([condition((x_train[i, :], None)) for i in range(x_train.shape[0])])
        categories = np.unique(category_map)
        
        estimator_list = []
        nc_list = []
        
        for i in range(len(categories)):
            
            # define a QRF model per group
            estimator_list.append(helper.MSENet_RegressorAdapter(model=None,
                                                                 fit_params=None,
                                                                 in_shape = in_shape,
                                                                 hidden_size = hidden_size,
                                                                 learn_func = nn_learn_func,
                                                                 epochs = epochs,
                                                                 batch_size=batch_size,
                                                                 dropout=dropout,
                                                                 lr=lr,
                                                                 wd=wd,
                                                                 test_ratio=cv_test_ratio,
                                                                 random_state=cv_random_state))
            
            # define the CQR object
            nc_list.append(RegressorNc(estimator_list[i], SignErrorErrFunc()))
        
        # run CQR procedure
        y_lower, y_upper = helper.run_icp_sep(nc_list, x_train, y_train, x_test, idx_train, idx_cal, alpha, condition)
        
        method_name = "Conditional Conformal Neural Network (groupwise)"
        
        # compute and print average coverage and average length
        coverage_sample, length_sample = helper.compute_coverage_per_sample(y_test,
                                                                            y_lower,
                                                                            y_upper,
                                                                            alpha,
                                                                            method_name,
                                                                            x_test,
                                                                            condition)
        
        
        append_statistics(coverage_sample,
                          length_sample,
                          method_name,
                          dataset_name_vec,
                          method_vec,
                          coverage_vec,
                          length_vec,
                          seed_vec,
                          test_ratio_vec,
                          seed,
                          test_ratio,
                          dataset_name_group_0,
                          dataset_name_group_1)
    
    # In[]
    
    if method == "qnet":
        
        # pytorch's optimizer object
        nn_learn_func = torch.optim.Adam
        
        # number of epochs
        epochs = 1000
        
        # learning rate
        lr = 0.0005
        
        # mini-batch size
        batch_size = 64
        
        # hidden dimension of the network
        hidden_size = 64
        
        # dropout regularization rate
        dropout = 0.1
        
        # weight decay regularization
        wd = 1e-6
        
        # desired quantiles
        quantiles_net = [0.05, 0.95]
        
        # ratio of held-out data, used in cross-validation
        cv_test_ratio = 0.1
        
        # seed for splitting the data in cross-validation.
        # Also used as the seed in quantile random forests function
        cv_random_state = 1
                
        # In[7]:
        
        
        # define quantile neural network model
        quantile_estimator = helper.AllQNet_RegressorAdapter(model=None,
                                                             fit_params=None,
                                                             in_shape=in_shape,
                                                             hidden_size=hidden_size,
                                                             quantiles=quantiles_net,
                                                             learn_func=nn_learn_func,
                                                             epochs=epochs,
                                                             batch_size=batch_size,
                                                             dropout=dropout,
                                                             lr=lr,
                                                             wd=wd,
                                                             test_ratio=cv_test_ratio,
                                                             random_state=cv_random_state,
                                                             use_rearrangement=False)
        
        # define the CQR object, computing the absolute residual error of points 
        # located outside the estimated quantile neural network band 
        nc = RegressorNc(quantile_estimator, QuantileRegAsymmetricErrFunc())
        
        # run CQR procedure
        y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_test, idx_train, idx_cal, alpha)
        
        method_name = "Marginal CQR Neural Network"
        
        # compute and print average coverage and average length
        coverage_sample, length_sample = helper.compute_coverage_per_sample(y_test,
                                                                            y_lower,
                                                                            y_upper,
                                                                            alpha,
                                                                            method_name,
                                                                            x_test,
                                                                            condition)
        
        
        append_statistics(coverage_sample,
                          length_sample,
                          method_name,
                          dataset_name_vec,
                          method_vec,
                          coverage_vec,
                          length_vec,
                          seed_vec,
                          test_ratio_vec,
                          seed,
                          test_ratio,
                          dataset_name_group_0,
                          dataset_name_group_1)
        
        
        # In[]
        
        # define qnet model
        quantile_estimator = helper.AllQNet_RegressorAdapter(model=None,
                                                             fit_params=None,
                                                             in_shape=in_shape,
                                                             hidden_size=hidden_size,
                                                             quantiles=quantiles_net,
                                                             learn_func=nn_learn_func,
                                                             epochs=epochs,
                                                             batch_size=batch_size,
                                                             dropout=dropout,
                                                             lr=lr,
                                                             wd=wd,
                                                             test_ratio=cv_test_ratio,
                                                             random_state=cv_random_state,
                                                             use_rearrangement=False)
                
        # define the CQR object
        nc = RegressorNc(quantile_estimator, QuantileRegAsymmetricErrFunc())
        
        # run CQR procedure
        y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_test, idx_train, idx_cal, alpha, condition)
        
        method_name = "Conditional CQR Neural Network (joint)"
        
        # compute and print average coverage and average length
        coverage_sample, length_sample = helper.compute_coverage_per_sample(y_test,
                                                                            y_lower,
                                                                            y_upper,
                                                                            alpha,
                                                                            method_name,
                                                                            x_test,
                                                                            condition)
        
        
        append_statistics(coverage_sample,
                          length_sample,
                          method_name,
                          dataset_name_vec,
                          method_vec,
                          coverage_vec,
                          length_vec,
                          seed_vec,
                          test_ratio_vec,
                          seed,
                          test_ratio,
                          dataset_name_group_0,
                          dataset_name_group_1)
        
        # In[6]
        
        category_map = np.array([condition((x_train[i, :], None)) for i in range(x_train.shape[0])])
        categories = np.unique(category_map)
        
        quantile_estimator_list = []
        nc_list = []
        
        for i in range(len(categories)):
            
            # define a QRF model per group
            quantile_estimator_list.append(helper.AllQNet_RegressorAdapter(model=None,
                                                                           fit_params=None,
                                                                           in_shape=in_shape,
                                                                           hidden_size=hidden_size,
                                                                           quantiles=quantiles_net,
                                                                           learn_func=nn_learn_func,
                                                                           epochs=epochs,
                                                                           batch_size=batch_size,
                                                                           dropout=dropout,
                                                                           lr=lr,
                                                                           wd=wd,
                                                                           test_ratio=cv_test_ratio,
                                                                           random_state=cv_random_state,
                                                                           use_rearrangement=False))      
            
            # append a CQR object
            nc_list.append(RegressorNc(quantile_estimator_list[i], QuantileRegAsymmetricErrFunc()))
        
        # run CQR procedure
        y_lower, y_upper = helper.run_icp_sep(nc_list, x_train, y_train, x_test, idx_train, idx_cal, alpha, condition)
        
        method_name = "Conditional CQR Neural Network (groupwise)"
        
        # compute and print average coverage and average length
        coverage_sample, length_sample = helper.compute_coverage_per_sample(y_test,
                                                                            y_lower,
                                                                            y_upper,
                                                                            alpha,
                                                                            method_name,
                                                                            x_test,
                                                                            condition)
        
        
        append_statistics(coverage_sample,
                          length_sample,
                          method_name,
                          dataset_name_vec,
                          method_vec,
                          coverage_vec,
                          length_vec,
                          seed_vec,
                          test_ratio_vec,
                          seed,
                          test_ratio,
                          dataset_name_group_0,
                          dataset_name_group_1)
        
    # In[]
    
    ############### Summary

    coverage_str = 'Coverage (expected ' + str(100 - alpha*100) + '%)'

    if save_to_csv:

        outdir = './results/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        out_name = outdir + 'results.csv'


        df = pd.DataFrame({'name': dataset_name_vec,
                           'method': method_vec,
                           coverage_str : coverage_vec,
                           'Avg. Length' : length_vec,
                           'seed' : seed_vec,
                           'train test ratio' : test_ratio_vec})

        if os.path.isfile(out_name):
            df2 = pd.read_csv(out_name)
            df = pd.concat([df2, df], ignore_index=True)

        df.to_csv(out_name, index=False)