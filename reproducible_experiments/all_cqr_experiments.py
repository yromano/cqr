
###############################################################################
# Script for reproducing the results of CQR paper
###############################################################################

import numpy as np
from reproducible_experiments.run_cqr_experiment import run_experiment
#from run_cqr_experiment import run_experiment


# list methods to test
test_methods = ['linear',
                'neural_net',
                'random_forest',
                'quantile_net',
                'cqr_quantile_net',
                'cqr_asymmetric_quantile_net',
                'rearrangement',
                'cqr_rearrangement',
                'cqr_asymmetric_rearrangement',
                'quantile_forest',
                'cqr_quantile_forest',
                'cqr_asymmetric_quantile_forest']

# list of datasets
dataset_names = ['meps_19',
                 'meps_20',
                 'meps_21',
                 'star',
                 'facebook_1',
                 'facebook_2',
                 'bio',
                 'blog_data',
                 'concrete',
                 'bike',
                 'community']

# vector of random seeds
random_state_train_test = np.arange(20)

for test_method_id in range(12):
    for dataset_name_id in range(11):
        for random_state_train_test_id in range(20):
            dataset_name = dataset_names[dataset_name_id]
            test_method = test_methods[test_method_id]
            random_state = random_state_train_test[random_state_train_test_id]

            # run an experiment and save average results to CSV file
            run_experiment(dataset_name, test_method, random_state)
