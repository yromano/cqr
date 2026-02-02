
from cqr import helper
from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import train_test_split


def CV_quntiles_rf(params,
                   X,
                   y,
                   target_coverage,
                   grid_q,
                   test_ratio,
                   random_state,
                   coverage_factor=0.9):
    """ Tune the low and high quantile level parameters of quantile random
        forests method, using cross-validation
    
    Parameters
    ----------
    params : dictionary of parameters
            params["random_state"] : integer, seed for splitting the data 
                                     in cross-validation. Also used as the
                                     seed in quantile random forest (QRF)
            params["min_samples_leaf"] : integer, parameter of QRF
            params["n_estimators"] : integer, parameter of QRF
            params["max_features"] : integer, parameter of QRF
    X : numpy array, containing the training features (nXp)
    y : numpy array, containing the training labels (n)
    target_coverage : desired coverage of prediction band. The output coverage
                      may be smaller if coverage_factor <= 1, in this case the
                      target will be modified to target_coverage*coverage_factor
    grid_q : numpy array, of low and high quantile levels to test
    test_ratio : float, test size of the held-out data
    random_state : integer, seed for splitting the data in cross-validation.
                   Also used as the seed in QRF.
    coverage_factor : float, when tuning the two QRF quantile levels one may
                      ask for prediction band with smaller average coverage,
                      equal to coverage_factor*(q_high - q_low) to avoid too
                      conservative estimation of the prediction band
    
    Returns
    -------
    best_q : numpy array of low and high quantile levels (length 2)
    
    References
    ----------
    .. [1]  Meinshausen, Nicolai. "Quantile regression forests."
            Journal of Machine Learning Research 7.Jun (2006): 983-999.
    
    """
    target_coverage = coverage_factor*target_coverage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio,random_state=random_state)
    best_avg_length = 1e10
    best_q = grid_q[0]

    rf = RandomForestQuantileRegressor(random_state=params["random_state"],
                                       min_samples_leaf=params["min_samples_leaf"],
                                       n_estimators=params["n_estimators"],
                                       max_features=params["max_features"])
    rf.fit(X_train, y_train)

    for q in grid_q:
        y_lower = rf.predict(X_test, quantile=q[0])
        y_upper = rf.predict(X_test, quantile=q[1])
        coverage, avg_length = helper.compute_coverage_len(y_test, y_lower, y_upper)
        if (coverage >= target_coverage) and (avg_length < best_avg_length):
            best_avg_length = avg_length
            best_q = q
        else:
            break
    return best_q
