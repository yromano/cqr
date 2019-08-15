
import sys
import torch
import numpy as np
from cqr import torch_models
from functools import partial
from cqr import tune_params_cv
from nonconformist.cp import IcpRegressor
from nonconformist.base import RegressorAdapter
from skgarden import RandomForestQuantileRegressor

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"


def compute_coverage_len(y_test, y_lower, y_upper):
    """ Compute average coverage and length of prediction intervals

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    avg_length = np.mean(abs(y_upper - y_lower))
    return coverage, avg_length

def run_icp(nc, X_train, y_train, X_test, idx_train, idx_cal, significance, condition=None):
    """ Run split conformal method

    Parameters
    ----------

    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping feature vector to group id

    Returns
    -------

    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """
    icp = IcpRegressor(nc,condition=condition)

    # Fit the ICP using the proper training set
    icp.fit(X_train[idx_train,:], y_train[idx_train])

    # Calibrate the ICP using the calibration set
    icp.calibrate(X_train[idx_cal,:], y_train[idx_cal])

    # Produce predictions for the test set, with confidence 90%
    predictions = icp.predict(X_test, significance=significance)

    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    return y_lower, y_upper


def run_icp_sep(nc, X_train, y_train, X_test, idx_train, idx_cal, significance, condition):
    """ Run split conformal method, train a seperate regressor for each group

    Parameters
    ----------

    nc : class of nonconformist object
    X_train : numpy array, training features (n1Xp)
    y_train : numpy array, training labels (n1)
    X_test : numpy array, testing features (n2Xp)
    idx_train : numpy array, indices of proper training set examples
    idx_cal : numpy array, indices of calibration set examples
    significance : float, significance level (e.g. 0.1)
    condition : function, mapping a feature vector to group id

    Returns
    -------

    y_lower : numpy array, estimated lower bound for the labels (n2)
    y_upper : numpy array, estimated upper bound for the labels (n2)

    """
    
    X_proper_train = X_train[idx_train,:]
    y_proper_train = y_train[idx_train]
    X_calibration = X_train[idx_cal,:]
    y_calibration = y_train[idx_cal]
    
    category_map_proper_train = np.array([condition((X_proper_train[i, :], y_proper_train[i])) for i in range(y_proper_train.size)])
    category_map_calibration = np.array([condition((X_calibration[i, :], y_calibration[i])) for i in range(y_calibration.size)])
    category_map_test = np.array([condition((X_test[i, :], None)) for i in range(X_test.shape[0])])
    
    categories = np.unique(category_map_proper_train)

    y_lower = np.zeros(X_test.shape[0])
    y_upper = np.zeros(X_test.shape[0])
    
    cnt = 0

    for cond in categories:
        
        icp = IcpRegressor(nc[cnt])
        
        idx_proper_train_group = category_map_proper_train == cond
        # Fit the ICP using the proper training set
        icp.fit(X_proper_train[idx_proper_train_group,:], y_proper_train[idx_proper_train_group])
    
        idx_calibration_group = category_map_calibration == cond
        # Calibrate the ICP using the calibration set
        icp.calibrate(X_calibration[idx_calibration_group,:], y_calibration[idx_calibration_group])
    
        idx_test_group = category_map_test == cond
        # Produce predictions for the test set, with confidence 90%
        predictions = icp.predict(X_test[idx_test_group,:], significance=significance)
    
        y_lower[idx_test_group] = predictions[:,0]
        y_upper[idx_test_group] = predictions[:,1]
        
        cnt = cnt + 1

    return y_lower, y_upper

def compute_coverage(y_test,y_lower,y_upper,significance,name=""):
    """ Compute average coverage and length, and print results

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    significance : float, desired significance level
    name : string, optional output string (e.g. the method name)

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    print("%s: Percentage in the range (expecting %.2f): %f" % (name, 100 - significance*100, coverage))
    sys.stdout.flush()

    avg_length = abs(np.mean(y_lower - y_upper))
    print("%s: Average length: %f" % (name, avg_length))
    sys.stdout.flush()
    return coverage, avg_length

def compute_coverage_per_sample(y_test,y_lower,y_upper,significance,name="",x_test=None,condition=None):
    """ Compute average coverage and length, and print results

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    significance : float, desired significance level
    name : string, optional output string (e.g. the method name)
    x_test : numpy array, test features
    condition : function, mapping a feature vector to group id

    Returns
    -------

    coverage : float, average coverage
    avg_length : float, average length

    """
    
    if condition is not None:
        
        category_map = np.array([condition((x_test[i, :], y_test[i])) for i in range(y_test.size)])
        categories = np.unique(category_map)
        
        coverage = np.empty(len(categories), dtype=np.object)
        length = np.empty(len(categories), dtype=np.object)
        
        cnt = 0
        
        for cond in categories:
                        
            idx = category_map == cond
            
            coverage[cnt] = (y_test[idx] >= y_lower[idx]) & (y_test[idx] <= y_upper[idx])

            coverage_avg = np.sum( coverage[cnt] ) / len(y_test[idx]) * 100
            print("%s: Group %d : Percentage in the range (expecting %.2f): %f" % (name, cond, 100 - significance*100, coverage_avg))
            sys.stdout.flush()
        
            length[cnt] = abs(y_upper[idx] - y_lower[idx])
            print("%s: Group %d : Average length: %f" % (name, cond, np.mean(length[cnt])))
            sys.stdout.flush()
            cnt = cnt + 1
    
    else:        
        
        coverage = (y_test >= y_lower) & (y_test <= y_upper)
        coverage_avg = np.sum(coverage) / len(y_test) * 100
        print("%s: Percentage in the range (expecting %.2f): %f" % (name, 100 - significance*100, coverage_avg))
        sys.stdout.flush()
    
        length = abs(y_upper - y_lower)
        print("%s: Average length: %f" % (name, np.mean(length)))
        sys.stdout.flush()
    
    return coverage, length


def plot_func_data(y_test,y_lower,y_upper,name=""):
    """ Plot the test labels along with the constructed prediction band

    Parameters
    ----------

    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    name : string, optional output string (e.g. the method name)

    """

    # allowed to import graphics
    import matplotlib.pyplot as plt

    interval = y_upper - y_lower
    sort_ind = np.argsort(interval)
    y_test_sorted = y_test[sort_ind]
    upper_sorted = y_upper[sort_ind]
    lower_sorted = y_lower[sort_ind]
    mean = (upper_sorted + lower_sorted) / 2

    # Center such that the mean of the prediction interval is at 0.0
    y_test_sorted -= mean
    upper_sorted -= mean
    lower_sorted -= mean

    plt.plot(y_test_sorted, "ro")
    plt.fill_between(
        np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
        label="Pred. interval")
    plt.xlabel("Ordered samples")
    plt.ylabel("Values and prediction intervals")

    plt.title(name)
    plt.show()

    interval = y_upper - y_lower
    sort_ind = np.argsort(y_test)
    y_test_sorted = y_test[sort_ind]
    upper_sorted = y_upper[sort_ind]
    lower_sorted = y_lower[sort_ind]

    plt.plot(y_test_sorted, "ro")
    plt.fill_between(
        np.arange(len(upper_sorted)), lower_sorted, upper_sorted, alpha=0.2, color="r",
        label="Pred. interval")
    plt.xlabel("Ordered samples by response")
    plt.ylabel("Values and prediction intervals")

    plt.title(name)
    plt.show()

###############################################################################
# Deep conditional mean regression
# Minimizing MSE loss
###############################################################################

class MSENet_RegressorAdapter(RegressorAdapter):
    """ Conditional mean estimator, formulated as neural net
    """
    def __init__(self,
                 model,
                 fit_params=None,
                 in_shape=1,
                 hidden_size=1,
                 learn_func=torch.optim.Adam,
                 epochs=1000,
                 batch_size=10,
                 dropout=0.1,
                 lr=0.01,
                 wd=1e-6,
                 test_ratio=0.2,
                 random_state=0):

        """ Initialization

        Parameters
        ----------
        model : unused parameter (for compatibility with nc class)
        fit_params : unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation

        """
        super(MSENet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.model = torch_models.mse_model(in_shape=in_shape, hidden_size=hidden_size, dropout=dropout)
        self.loss_func = torch.nn.MSELoss()
        self.learner = torch_models.LearnerOptimized(self.model,
                                                     partial(learn_func, lr=lr, weight_decay=wd),
                                                     self.loss_func,
                                                     device=device,
                                                     test_ratio=self.test_ratio,
                                                     random_state=self.random_state)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(x, y, self.epochs, batch_size=self.batch_size)

    def predict(self, x):
        """ Estimate the label given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of predicted labels (n)

        """
        return self.learner.predict(x)

###############################################################################
# Deep neural network for conditional quantile regression
# Minimizing pinball loss
###############################################################################

class AllQNet_RegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, formulated as neural net
    """
    def __init__(self,
                 model,
                 fit_params=None,
                 in_shape=1,
                 hidden_size=1,
                 quantiles=[.05, .95],
                 learn_func=torch.optim.Adam,
                 epochs=1000,
                 batch_size=10,
                 dropout=0.1,
                 lr=0.01,
                 wd=1e-6,
                 test_ratio=0.2,
                 random_state=0,
                 use_rearrangement=False):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        in_shape : integer, input signal dimension
        hidden_size : integer, hidden layer dimension
        quantiles : numpy array, low and high quantile levels in range (0,1)
        learn_func : class of Pytorch's SGD optimizer
        epochs : integer, maximal number of epochs
        batch_size : integer, mini-batch size for SGD
        dropout : float, dropout rate
        lr : float, learning rate for SGD
        wd : float, weight decay
        test_ratio : float, ratio of held-out data, used in cross-validation
        random_state : integer, seed for splitting the data in cross-validation
        use_rearrangement : boolean, use the rearrangement algorithm (True)
                            of not (False). See reference [1].

        References
        ----------
        .. [1]  Chernozhukov, Victor, Iván Fernández‐Val, and Alfred Galichon.
                "Quantile and probability curves without crossing."
                Econometrica 78.3 (2010): 1093-1125.

        """
        super(AllQNet_RegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        if use_rearrangement:
            self.all_quantiles = torch.from_numpy(np.linspace(0.01,0.99,99)).float()
        else:
            self.all_quantiles = self.quantiles
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.wd = wd
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.model = torch_models.all_q_model(quantiles=self.all_quantiles,
                                              in_shape=in_shape,
                                              hidden_size=hidden_size,
                                              dropout=dropout)
        self.loss_func = torch_models.AllQuantileLoss(self.all_quantiles)
        self.learner = torch_models.LearnerOptimizedCrossing(self.model,
                                                             partial(learn_func, lr=lr, weight_decay=wd),
                                                             self.loss_func,
                                                             device=device,
                                                             test_ratio=self.test_ratio,
                                                             random_state=self.random_state,
                                                             qlow=self.quantiles[0],
                                                             qhigh=self.quantiles[1],
                                                             use_rearrangement=use_rearrangement)

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        self.learner.fit(x, y, self.epochs, self.batch_size)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        return self.learner.predict(x)


###############################################################################
# Quantile random forests model
###############################################################################

class QuantileForestRegressorAdapter(RegressorAdapter):
    """ Conditional quantile estimator, defined as quantile random forests (QRF)

    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.

    """

    def __init__(self,
                 model,
                 fit_params=None,
                 quantiles=[5, 95],
                 params=None):
        """ Initialization

        Parameters
        ----------
        model : None, unused parameter (for compatibility with nc class)
        fit_params : None, unused parameter (for compatibility with nc class)
        quantiles : numpy array, low and high quantile levels in range (0,100)
        params : dictionary of parameters
                params["random_state"] : integer, seed for splitting the data
                                         in cross-validation. Also used as the
                                         seed in quantile random forests (QRF)
                params["min_samples_leaf"] : integer, parameter of QRF
                params["n_estimators"] : integer, parameter of QRF
                params["max_features"] : integer, parameter of QRF
                params["CV"] : boolean, use cross-validation (True) or
                               not (False) to tune the two QRF quantile levels
                               to obtain the desired coverage
                params["test_ratio"] : float, ratio of held-out data, used
                                       in cross-validation
                params["coverage_factor"] : float, to avoid too conservative
                                            estimation of the prediction band,
                                            when tuning the two QRF quantile
                                            levels in cross-validation one may
                                            ask for prediction intervals with
                                            reduced average coverage, equal to
                                            coverage_factor*(q_high - q_low).
                params["range_vals"] : float, determines the lowest and highest
                                       quantile level parameters when tuning
                                       the quanitle levels bt cross-validation.
                                       The smallest value is equal to
                                       quantiles[0] - range_vals.
                                       Similarly, the largest is equal to
                                       quantiles[1] + range_vals.
                params["num_vals"] : integer, when tuning QRF's quantile
                                     parameters, sweep over a grid of length
                                     num_vals.

        """
        super(QuantileForestRegressorAdapter, self).__init__(model, fit_params)
        # Instantiate model
        self.quantiles = quantiles
        self.cv_quantiles = self.quantiles
        self.params = params
        self.rfqr = RandomForestQuantileRegressor(random_state=params["random_state"],
                                                  min_samples_leaf=params["min_samples_leaf"],
                                                  n_estimators=params["n_estimators"],
                                                  max_features=params["max_features"])

    def fit(self, x, y):
        """ Fit the model to data

        Parameters
        ----------

        x : numpy array of training features (nXp)
        y : numpy array of training labels (n)

        """
        if self.params["CV"]:
            target_coverage = self.quantiles[1] - self.quantiles[0]
            coverage_factor = self.params["coverage_factor"]
            range_vals = self.params["range_vals"]
            num_vals = self.params["num_vals"]
            grid_q_low = np.linspace(self.quantiles[0],self.quantiles[0]+range_vals,num_vals).reshape(-1,1)
            grid_q_high = np.linspace(self.quantiles[1],self.quantiles[1]-range_vals,num_vals).reshape(-1,1)
            grid_q = np.concatenate((grid_q_low,grid_q_high),1)

            self.cv_quantiles = tune_params_cv.CV_quntiles_rf(self.params,
                                                              x,
                                                              y,
                                                              target_coverage,
                                                              grid_q,
                                                              self.params["test_ratio"],
                                                              self.params["random_state"],
                                                              coverage_factor)

        self.rfqr.fit(x, y)

    def predict(self, x):
        """ Estimate the conditional low and high quantiles given the features

        Parameters
        ----------
        x : numpy array of training features (nXp)

        Returns
        -------
        ret_val : numpy array of estimated conditional quantiles (nX2)

        """
        lower = self.rfqr.predict(x, quantile=self.cv_quantiles[0])
        upper = self.rfqr.predict(x, quantile=self.cv_quantiles[1])

        ret_val = np.zeros((len(lower),2))
        ret_val[:,0] = lower
        ret_val[:,1] = upper
        return ret_val
