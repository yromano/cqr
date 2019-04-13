###############################################################################
# Script for reproducing the results in CQR paper
###############################################################################

import numpy as np
from run_experiment import run_experiment

# list methods to test
test_methods = ['linear_model',
                'neural_net',
                'random_forest',
                'quantile_net',
                'rearrangement',
                'quantile_forest']

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

for test_method_id in range(6):
    for dataset_name_id in range(11):
        for random_state_train_test_id in range(20):
            dataset_name = dataset_names[dataset_name_id]
            test_method = test_methods[test_method_id]
            random_state = random_state_train_test[random_state_train_test_id]

            # run an experiment and save average results to CSV file
            run_experiment(dataset_name, test_method, random_state)
