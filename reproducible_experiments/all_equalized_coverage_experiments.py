###############################################################################
# Script for reproducing the results of CQR paper
###############################################################################

import numpy as np
from reproducible_experiments.run_equalized_coverage_experiment import run_equalized_coverage_experiment
#from run_equalized_coverage_experiment import run_equalized_coverage_experiment

# list methods to test
test_methods = ['net',
                'qnet']

dataset_names = ["meps_21"]

test_ratio_vec = [0.2]
                
# vector of random seeds
random_state_train_test = np.arange(40)

for test_method_id in range(2):
    for random_state_train_test_id in range(40):
        for dataset_name_id in range(1):
            for test_ratio_id in range(1):
                test_ratio = test_ratio_vec[test_ratio_id]
                test_method = test_methods[test_method_id]
                random_state = random_state_train_test[random_state_train_test_id]
                dataset_name = dataset_names[dataset_name_id]
        
                # run an experiment and save average results to CSV file
                run_equalized_coverage_experiment(dataset_name,
                                                  test_method,
                                                  random_state,
                                                  True,
                                                  test_ratio)
