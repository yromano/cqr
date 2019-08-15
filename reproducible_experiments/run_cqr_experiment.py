import os
import sys
import torch
import random
import numpy as np
import pandas as pd
from cqr import helper
from datasets import datasets
from sklearn import linear_model
from nonconformist.nc import NcFactory
from nonconformist.nc import RegressorNc
from nonconformist.nc import AbsErrorErrFunc
from nonconformist.nc import QuantileRegErrFunc
from nonconformist.nc import RegressorNormalizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from nonconformist.nc import QuantileRegAsymmetricErrFunc


pd.set_option('precision', 3)

base_dataset_path = './datasets/'

if os.path.isdir('/scratch'):
    local_machine = 0
else:
    local_machine = 1
    
if local_machine:
    base_dataset_path = '/Users/romano/mydata/regression_data/'
else:
    base_dataset_path = '/scratch/users/yromano/data/regression_data/'
    
plot_results = False


def run_experiment(dataset_name,
                   test_method,
                   random_state_train_test,
                   save_to_csv=True):
    """ Estimate prediction intervals and print the average length and coverage

    Parameters
    ----------

    dataset_name : array of strings, list of datasets
    test_method  : string, method to be tested, estimating
                   the 90% prediction interval
    random_state_train_test : integer, random seed to be used
    save_to_csv : boolean, save average length and coverage to csv (True)
                  or not (False)

    """

    dataset_name_vec = []
    method_vec = []
    coverage_vec = []
    length_vec = []
    seed_vec = []

    seed = random_state_train_test
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    coverage_linear=0
    length_linear=0
    coverage_linear_local=0
    length_linear_local=0

    coverage_net=0
    length_net=0
    coverage_net_local=0
    length_net_local=0

    coverage_forest=0
    length_forest=0
    coverage_forest_local=0
    length_forest_local=0

    coverage_cp_qnet=0
    length_cp_qnet=0
    coverage_qnet=0
    length_qnet=0

    coverage_cp_sign_qnet=0
    length_cp_sign_qnet=0

    coverage_cp_re_qnet=0
    length_cp_re_qnet=0
    coverage_re_qnet=0
    length_re_qnet=0

    coverage_cp_sign_re_qnet=0
    length_cp_sign_re_qnet=0

    coverage_cp_qforest=0
    length_cp_qforest=0
    coverage_qforest=0
    length_qforest=0

    coverage_cp_sign_qforest=0
    length_cp_sign_qforest=0


    # determines the size of test set
    test_ratio = 0.2

    # conformal prediction miscoverage level
    significance = 0.1
    # desired quantile levels, used by the quantile regression methods
    quantiles = [0.05, 0.95]

    # Random forests parameters (shared by conditional quantile random forests
    # and conditional mean random forests regression).
    n_estimators = 1000 # usual random forests n_estimators parameter
    min_samples_leaf = 1 # default parameter of sklearn

    # Quantile random forests parameters.
    # See QuantileForestRegressorAdapter class for more details
    quantiles_forest = [5, 95]
    CV_qforest = True
    coverage_factor = 0.85
    cv_test_ratio = 0.05
    cv_random_state = 1
    cv_range_vals = 30
    cv_num_vals = 10

    # Neural network parameters  (shared by conditional quantile neural network
    # and conditional mean neural network regression)
    # See AllQNet_RegressorAdapter and MSENet_RegressorAdapter in helper.py
    nn_learn_func = torch.optim.Adam
    epochs = 1000
    lr = 0.0005
    hidden_size = 64
    batch_size = 64
    dropout = 0.1
    wd = 1e-6

    # Ask for a reduced coverage when tuning the network parameters by
    # cross-validation to avoid too conservative initial estimation of the
    # prediction interval. This estimation will be conformalized by CQR.
    quantiles_net = [0.1, 0.9]


    # local conformal prediction parameter.
    # See RegressorNc class for more details.
    beta = 1
    beta_net = 1

    # local conformal prediction parameter. The local ridge regression method
    # uses nearest neighbor regression as the MAD estimator.
    # Number of neighbors used by nearest neighbor regression.
    n_neighbors = 11

    print(dataset_name)
    sys.stdout.flush()

    try:
        # load the dataset
        X, y = datasets.GetDataset(dataset_name, base_dataset_path)
    except:
        print("CANNOT LOAD DATASET!")
        return

    # Dataset is divided into test and train data based on test_ratio parameter
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_ratio,
                                                        random_state=random_state_train_test)


    # fit a simple ridge regression model (sanity check)
    model = linear_model.RidgeCV()
    model = model.fit(X_train, np.squeeze(y_train))
    predicted_data = model.predict(X_test).astype(np.float32)

    # calculate the normalized mean squared error
    print("Ridge relative error: %f" % (np.sum((np.squeeze(y_test)-predicted_data)**2)/np.sum(np.squeeze(y_test)**2)))
    sys.stdout.flush()

    # reshape the data
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    # input dimensions
    n_train = X_train.shape[0]
    in_shape = X_train.shape[1]

    print("Size: train (%d, %d), test (%d, %d)" % (X_train.shape[0], X_train.shape[1], X_test.shape[0], X_test.shape[1]))
    sys.stdout.flush()

    # set seed for splitting the data into proper train and calibration
    np.random.seed(seed)
    idx = np.random.permutation(n_train)

    # divide the data into proper training set and calibration set
    n_half = int(np.floor(n_train/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]
    
    # zero mean and unit variance scaling of the train and test features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X_train[idx_train])
    X_train = scalerX.transform(X_train)
    X_test = scalerX.transform(X_test)
    
    # scale the labels by dividing each by the mean absolute response
    mean_ytrain = np.mean(np.abs(y_train[idx_train]))
    y_train = np.squeeze(y_train)/mean_ytrain
    y_test = np.squeeze(y_test)/mean_ytrain
    
    
    ######################## Linear

    if 'linear' == test_method:

        model = linear_model.RidgeCV()
        nc = RegressorNc(model)

        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"Ridge")
        coverage_linear, length_linear = helper.compute_coverage(y_test,y_lower,y_upper,significance,"Ridge")

        dataset_name_vec.append(dataset_name)
        method_vec.append('Ridge')
        coverage_vec.append(coverage_linear)
        length_vec.append(length_linear)
        seed_vec.append(seed)

        nc = NcFactory.create_nc(
            linear_model.RidgeCV(),
            normalizer_model=KNeighborsRegressor(n_neighbors=n_neighbors)
        )

        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"Ridge-L")
        coverage_linear_local, length_linear_local = helper.compute_coverage(y_test,y_lower,y_upper,significance,"Ridge-L")

        dataset_name_vec.append(dataset_name)
        method_vec.append('Ridge-L')
        coverage_vec.append(coverage_linear_local)
        length_vec.append(length_linear_local)
        seed_vec.append(seed)

    ######################### Neural net

    if 'neural_net' == test_method:

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
        nc = RegressorNc(model)

        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"Net")
        coverage_net, length_net = helper.compute_coverage(y_test,y_lower,y_upper,significance,"Net")

        dataset_name_vec.append(dataset_name)
        method_vec.append('Net')
        coverage_vec.append(coverage_net)
        length_vec.append(length_net)
        seed_vec.append(seed)

        normalizer_adapter = helper.MSENet_RegressorAdapter(model=None,
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
        adapter = helper.MSENet_RegressorAdapter(model=None,
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

        normalizer = RegressorNormalizer(adapter,
                                         normalizer_adapter,
                                         AbsErrorErrFunc())
        nc = RegressorNc(adapter, AbsErrorErrFunc(), normalizer, beta=beta_net)
        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"Net-L")
        coverage_net_local, length_net_local = helper.compute_coverage(y_test,y_lower,y_upper,significance,"Net-L")

        dataset_name_vec.append(dataset_name)
        method_vec.append('Net-L')
        coverage_vec.append(coverage_net_local)
        length_vec.append(length_net_local)
        seed_vec.append(seed)

    ################## Random Forest

    if 'random_forest' == test_method:

        model = RandomForestRegressor(n_estimators=n_estimators,min_samples_leaf=min_samples_leaf, random_state=0)
        nc = RegressorNc(model, AbsErrorErrFunc())

        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"RF")
        coverage_forest, length_forest = helper.compute_coverage(y_test,y_lower,y_upper,significance,"RF")

        dataset_name_vec.append(dataset_name)
        method_vec.append('RF')
        coverage_vec.append(coverage_forest)
        length_vec.append(length_forest)
        seed_vec.append(seed)

        normalizer_adapter = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=0)
        adapter = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=0)
        normalizer = RegressorNormalizer(adapter,
                                         normalizer_adapter,
                                         AbsErrorErrFunc())
        nc = RegressorNc(adapter, AbsErrorErrFunc(), normalizer, beta=beta)

        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"RF-L")
        coverage_forest_local, length_forest_local = helper.compute_coverage(y_test,y_lower,y_upper,significance,"RF-L")

        dataset_name_vec.append(dataset_name)
        method_vec.append('RF-L')
        coverage_vec.append(coverage_forest_local)
        length_vec.append(length_forest_local)
        seed_vec.append(seed)

    ################## Quantile Net

    if 'quantile_net' == test_method:

        model_full = helper.AllQNet_RegressorAdapter(model=None,
                                             fit_params=None,
                                             in_shape = in_shape,
                                             hidden_size = hidden_size,
                                             quantiles = quantiles,
                                             learn_func = nn_learn_func,
                                             epochs = epochs,
                                             batch_size=batch_size,
                                             dropout=dropout,
                                             lr=lr,
                                             wd=wd,
                                             test_ratio=cv_test_ratio,
                                             random_state=cv_random_state,
                                             use_rearrangement=False)
        model_full.fit(X_train, y_train)
        tmp = model_full.predict(X_test)
        y_lower = tmp[:,0]
        y_upper = tmp[:,1]
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"QNet")
        coverage_qnet, length_qnet = helper.compute_coverage(y_test,y_lower,y_upper,significance,"QNet")

        dataset_name_vec.append(dataset_name)
        method_vec.append('QNet')
        coverage_vec.append(coverage_qnet)
        length_vec.append(length_qnet)
        seed_vec.append(seed)

    if 'cqr_quantile_net' == test_method:

        model = helper.AllQNet_RegressorAdapter(model=None,
                                             fit_params=None,
                                             in_shape = in_shape,
                                             hidden_size = hidden_size,
                                             quantiles = quantiles_net,
                                             learn_func = nn_learn_func,
                                             epochs = epochs,
                                             batch_size=batch_size,
                                             dropout=dropout,
                                             lr=lr,
                                             wd=wd,
                                             test_ratio=cv_test_ratio,
                                             random_state=cv_random_state,
                                             use_rearrangement=False)
        nc = RegressorNc(model, QuantileRegErrFunc())

        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"CQR Net")
        coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(y_test,y_lower,y_upper,significance,"CQR Net")


        dataset_name_vec.append(dataset_name)
        method_vec.append('CQR Net')
        coverage_vec.append(coverage_cp_qnet)
        length_vec.append(length_cp_qnet)
        seed_vec.append(seed)


    if 'cqr_asymmetric_quantile_net' == test_method:

        model = helper.AllQNet_RegressorAdapter(model=None,
                                             fit_params=None,
                                             in_shape = in_shape,
                                             hidden_size = hidden_size,
                                             quantiles = quantiles_net,
                                             learn_func = nn_learn_func,
                                             epochs = epochs,
                                             batch_size=batch_size,
                                             dropout=dropout,
                                             lr=lr,
                                             wd=wd,
                                             test_ratio=cv_test_ratio,
                                             random_state=cv_random_state,
                                             use_rearrangement=False)
        nc = RegressorNc(model, QuantileRegAsymmetricErrFunc())

        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"CQR Sign Net")
        coverage_cp_sign_qnet, length_cp_sign_qnet = helper.compute_coverage(y_test,y_lower,y_upper,significance,"CQR Sign Net")


        dataset_name_vec.append(dataset_name)
        method_vec.append('CQR Sign Net')
        coverage_vec.append(coverage_cp_sign_qnet)
        length_vec.append(length_cp_sign_qnet)
        seed_vec.append(seed)


    ################### Rearrangement Quantile Net

    if 'rearrangement' == test_method:

        model_full = helper.AllQNet_RegressorAdapter(model=None,
                                             fit_params=None,
                                             in_shape = in_shape,
                                             hidden_size = hidden_size,
                                             quantiles = quantiles,
                                             learn_func = nn_learn_func,
                                             epochs = epochs,
                                             batch_size=batch_size,
                                             dropout=dropout,
                                             lr=lr,
                                             wd=wd,
                                             test_ratio=cv_test_ratio,
                                             random_state=cv_random_state,
                                             use_rearrangement=True)
        model_full.fit(X_train, y_train)
        tmp = model_full.predict(X_test)
        y_lower = tmp[:,0]
        y_upper = tmp[:,1]
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"Rearrange QNet")
        coverage_re_qnet, length_re_qnet = helper.compute_coverage(y_test,y_lower,y_upper,significance,"Rearrange QNet")

        dataset_name_vec.append(dataset_name)
        method_vec.append('Rearrange QNet')
        coverage_vec.append(coverage_re_qnet)
        length_vec.append(length_re_qnet)
        seed_vec.append(seed)

    if 'cqr_rearrangement' == test_method:

        model = helper.AllQNet_RegressorAdapter(model=None,
                                                 fit_params=None,
                                                 in_shape = in_shape,
                                                 hidden_size = hidden_size,
                                                 quantiles = quantiles_net,
                                                 learn_func = nn_learn_func,
                                                 epochs = epochs,
                                                 batch_size=batch_size,
                                                 dropout=dropout,
                                                 lr=lr,
                                                 wd=wd,
                                                 test_ratio=cv_test_ratio,
                                                 random_state=cv_random_state,
                                                 use_rearrangement=True)
        nc = RegressorNc(model, QuantileRegErrFunc())

        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"Rearrange CQR Net")
        coverage_cp_re_qnet, length_cp_re_qnet = helper.compute_coverage(y_test,y_lower,y_upper,significance,"Rearrange CQR Net")


        dataset_name_vec.append(dataset_name)
        method_vec.append('Rearrange CQR Net')
        coverage_vec.append(coverage_cp_re_qnet)
        length_vec.append(length_cp_re_qnet)
        seed_vec.append(seed)


    if 'cqr_asymmetric_rearrangement' == test_method:

        model = helper.AllQNet_RegressorAdapter(model=None,
                                                 fit_params=None,
                                                 in_shape = in_shape,
                                                 hidden_size = hidden_size,
                                                 quantiles = quantiles_net,
                                                 learn_func = nn_learn_func,
                                                 epochs = epochs,
                                                 batch_size=batch_size,
                                                 dropout=dropout,
                                                 lr=lr,
                                                 wd=wd,
                                                 test_ratio=cv_test_ratio,
                                                 random_state=cv_random_state,
                                                 use_rearrangement=True)
        nc = RegressorNc(model, QuantileRegAsymmetricErrFunc())

        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"Rearrange CQR Sign Net")
        coverage_cp_sign_re_qnet, length_cp_sign_re_qnet = helper.compute_coverage(y_test,y_lower,y_upper,significance,"Rearrange CQR Net")


        dataset_name_vec.append(dataset_name)
        method_vec.append('Rearrange CQR Sign Net')
        coverage_vec.append(coverage_cp_sign_re_qnet)
        length_vec.append(length_cp_sign_re_qnet)
        seed_vec.append(seed)

    ################### Quantile Random Forest

    if 'quantile_forest' == test_method:

        params_qforest = dict()
        params_qforest["random_state"] = 0
        params_qforest["min_samples_leaf"] = min_samples_leaf
        params_qforest["n_estimators"] = n_estimators
        params_qforest["max_features"] = X_train.shape[1]

        params_qforest["CV"]=False
        params_qforest["coverage_factor"] = coverage_factor
        params_qforest["test_ratio"]=cv_test_ratio
        params_qforest["random_state"]=cv_random_state
        params_qforest["range_vals"] = cv_range_vals
        params_qforest["num_vals"] = cv_num_vals

        model_full = helper.QuantileForestRegressorAdapter(model = None,
                                                      fit_params=None,
                                                      quantiles=np.dot(100,quantiles),
                                                      params = params_qforest)
        model_full.fit(X_train, y_train)
        tmp = model_full.predict(X_test)
        y_lower = tmp[:,0]
        y_upper = tmp[:,1]
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"QRF")
        coverage_qforest, length_qforest = helper.compute_coverage(y_test,y_lower,y_upper,significance,"QRF")

        dataset_name_vec.append(dataset_name)
        method_vec.append('QRF')
        coverage_vec.append(coverage_qforest)
        length_vec.append(length_qforest)
        seed_vec.append(seed)

    if 'cqr_quantile_forest' == test_method:

        params_qforest = dict()
        params_qforest["random_state"] = 0
        params_qforest["min_samples_leaf"] = min_samples_leaf
        params_qforest["n_estimators"] = n_estimators
        params_qforest["max_features"] = X_train.shape[1]

        params_qforest["CV"]=CV_qforest
        params_qforest["coverage_factor"] = coverage_factor
        params_qforest["test_ratio"]=cv_test_ratio
        params_qforest["random_state"]=cv_random_state
        params_qforest["range_vals"] = cv_range_vals
        params_qforest["num_vals"] = cv_num_vals


        model = helper.QuantileForestRegressorAdapter(model = None,
                                                      fit_params=None,
                                                      quantiles=quantiles_forest,
                                                      params = params_qforest)

        nc = RegressorNc(model, QuantileRegErrFunc())
        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"CQR RF")
        coverage_cp_qforest, length_cp_qforest = helper.compute_coverage(y_test,y_lower,y_upper,significance,"CQR RF")

        dataset_name_vec.append(dataset_name)
        method_vec.append('CQR RF')
        coverage_vec.append(coverage_cp_qforest)
        length_vec.append(length_cp_qforest)
        seed_vec.append(seed)

    if 'cqr_asymmetric_quantile_forest' == test_method:

        params_qforest = dict()
        params_qforest["random_state"] = 0
        params_qforest["min_samples_leaf"] = min_samples_leaf
        params_qforest["n_estimators"] = n_estimators
        params_qforest["max_features"] = X_train.shape[1]

        params_qforest["CV"]=CV_qforest
        params_qforest["coverage_factor"] = coverage_factor
        params_qforest["test_ratio"]=cv_test_ratio
        params_qforest["random_state"]=cv_random_state
        params_qforest["range_vals"] = cv_range_vals
        params_qforest["num_vals"] = cv_num_vals


        model = helper.QuantileForestRegressorAdapter(model = None,
                                                      fit_params=None,
                                                      quantiles=quantiles_forest,
                                                      params = params_qforest)

        nc = RegressorNc(model, QuantileRegAsymmetricErrFunc())
        y_lower, y_upper = helper.run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance)
        if plot_results:
            helper.plot_func_data(y_test,y_lower,y_upper,"CQR Sign RF")
        coverage_cp_sign_qforest, length_cp_sign_qforest = helper.compute_coverage(y_test,y_lower,y_upper,significance,"CQR Sign RF")

        dataset_name_vec.append(dataset_name)
        method_vec.append('CQR Sign RF')
        coverage_vec.append(coverage_cp_sign_qforest)
        length_vec.append(length_cp_sign_qforest)
        seed_vec.append(seed)


#        tmp = model.predict(X_test)
#        y_lower = tmp[:,0]
#        y_upper = tmp[:,1]
#        if plot_results:
#            helper.plot_func_data(y_test,y_lower,y_upper,"QRF")
#        coverage_qforest, length_qforest = helper.compute_coverage(y_test,y_lower,y_upper,significance,"QRF")
#
#        dataset_name_vec.append(dataset_name)
#        method_vec.append('QRF')
#        coverage_vec.append(coverage_qforest)
#        length_vec.append(length_qforest)
#        seed_vec.append(seed)



    ############### Summary

    coverage_str = 'Coverage (expected ' + str(100 - significance*100) + '%)'
    results = np.array([[dataset_name, coverage_str, 'Avg. Length', 'Seed'],
                     ['CP Linear', coverage_linear, length_linear, seed],
                     ['CP Linear Local', coverage_linear_local, length_linear_local, seed],
                     ['CP Neural Net', coverage_net, length_net, seed],
                     ['CP Neural Net Local', coverage_net_local, length_net_local, seed],
                     ['CP Random Forest', coverage_forest, length_forest, seed],
                     ['CP Random Forest Local', coverage_forest_local, length_forest_local, seed],
                     ['CP Quantile Net', coverage_cp_qnet, length_cp_qnet, seed],
                     ['CP Asymmetric Quantile Net', coverage_cp_sign_qnet, length_cp_sign_qnet, seed],
                     ['Quantile Net', coverage_qnet, length_qnet, seed],
                     ['CP Rearrange Quantile Net', coverage_cp_re_qnet, length_cp_re_qnet, seed],
                     ['CP Asymmetric Rearrange Quantile Net', coverage_cp_sign_re_qnet, length_cp_sign_re_qnet, seed],
                     ['Rearrange Quantile Net', coverage_re_qnet, length_re_qnet, seed],
                     ['CP Quantile Random Forest', coverage_cp_qforest, length_cp_qforest, seed],
                     ['CP Asymmetric Quantile Random Forest', coverage_cp_sign_qforest, length_cp_sign_qforest, seed],
                     ['Quantile Random Forest', coverage_qforest, length_qforest, seed]])

    results_ = pd.DataFrame(data=results[1:,1:],
                      index=results[1:,0],
                      columns=results[0,1:])

    print("== SUMMARY == ")
    print("dataset name: " + dataset_name)
    print(results_)
    sys.stdout.flush()

    if save_to_csv:
        results = pd.DataFrame(results)

        outdir = './results/'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        out_name = outdir + 'results.csv'

        df = pd.DataFrame({'name': dataset_name_vec,
                           'method': method_vec,
                           coverage_str : coverage_vec,
                           'Avg. Length' : length_vec,
                           'seed': seed_vec})

        if os.path.isfile(out_name):
            df2 = pd.read_csv(out_name)
            df = pd.concat([df2, df], ignore_index=True)

        df.to_csv(out_name, index=False)
