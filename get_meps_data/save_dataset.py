# Code copied from IBM's AIF360 package
# https://github.com/IBM/AIF360/blob/master/aif360/datasets/standard_dataset.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from logging import warning

import numpy as np
import pandas as pd

from regression_dataset import RegressionDataset


class SaveDataset(RegressionDataset):
    """Base class for every :obj:`RegressionDataset`. The code is similar
    to that of aif360.

    It is not strictly necessary to inherit this class when adding custom
    datasets but it may be useful.

    This class is very loosely based on code from
    https://github.com/algofairness/fairness-comparison.
    """

    def __init__(self, df, label_name, favorable_classes,
                 protected_attribute_names, privileged_classes,
                 instance_weights_name='', scores_name='',
                 categorical_features=[], features_to_keep=[],
                 features_to_drop=[], na_values=[], custom_preprocessing=None,
                 metadata=None, dataset_name='my_data'):
        """
        Subclasses of StandardDataset should perform the following before
        calling `super().__init__`:

            1. Load the dataframe from a raw file.

        Then, this class will go through a standard preprocessing routine which:

            2. (optional) Performs some dataset-specific preprocessing (e.g.
               renaming columns/values, handling missing data).

            3. Drops unrequested columns (see `features_to_keep` and
               `features_to_drop` for details).

            4. Drops rows with NA values.

            5. Creates a one-hot encoding of the categorical variables.

            6. Maps protected attributes to binary privileged/unprivileged
               values (1/0).

        Args:
            df (pandas.DataFrame): DataFrame on which to perform standard
                processing.
            label_name: Name of the label column in `df`.
            favorable_classes (list or function): Label values which are
                considered favorable or a boolean function which returns `True`
                if favorable. All others are unfavorable. Label values are
                mapped to 1 (favorable) and 0 (unfavorable) if they are not
                already binary and numerical.
            protected_attribute_names (list): List of names corresponding to
                protected attribute columns in `df`.
            privileged_classes (list(list or function)): Each element is
                a list of values which are considered privileged or a boolean
                function which return `True` if privileged for the corresponding
                column in `protected_attribute_names`. All others are
                unprivileged. Values are mapped to 1 (privileged) and 0
                (unprivileged) if they are not already numerical.
            instance_weights_name (optional): Name of the instance weights
                column in `df`.
            categorical_features (optional, list): List of column names in the
                DataFrame which are to be expanded into one-hot vectors.
            features_to_keep (optional, list): Column names to keep. All others
                are dropped except those present in `protected_attribute_names`,
                `categorical_features`, `label_name` or `instance_weights_name`.
                Defaults to all columns if not provided.
            features_to_drop (optional, list): Column names to drop. *Note: this
                overrides* `features_to_keep`.
            na_values (optional): Additional strings to recognize as NA. See
                :func:`pandas.read_csv` for details.
            custom_preprocessing (function): A function object which
                acts on and returns a DataFrame (f: DataFrame -> DataFrame). If
                `None`, no extra preprocessing is applied.
            metadata (optional): Additional metadata to append.
        """
        # 2. Perform dataset-specific preprocessing
        if custom_preprocessing:
            df = custom_preprocessing(df)

        # 3. Drop unrequested columns
        features_to_keep = features_to_keep or df.columns.tolist()
        keep = (set(features_to_keep) | set(protected_attribute_names)
              | set(categorical_features) | set([label_name]))
        if instance_weights_name:
            keep |= set([instance_weights_name])
        df = df[sorted(keep - set(features_to_drop), key=df.columns.get_loc)]
        categorical_features = sorted(set(categorical_features) - set(features_to_drop), key=df.columns.get_loc)

        # 4. Remove any rows that have missing data.
        dropped = df.dropna()
        count = df.shape[0] - dropped.shape[0]
        if count > 0:
            warning("Missing Data: {} rows removed from {}.".format(count,
                    type(self).__name__))
        df = dropped

        # 5. Create a one-hot encoding of the categorical variables.
        df = pd.get_dummies(df, columns=categorical_features, prefix_sep='=')

        # 6. Map protected attributes to privileged/unprivileged
        privileged_protected_attributes = []
        unprivileged_protected_attributes = []
        for attr, vals in zip(protected_attribute_names, privileged_classes):
            privileged_values = [1.]
            unprivileged_values = [0.]
            if callable(vals):
                df[attr] = df[attr].apply(vals)
            elif np.issubdtype(df[attr].dtype, np.number):
                # this attribute is numeric; no remapping needed
                privileged_values = vals
                unprivileged_values = list(set(df[attr]).difference(vals))
            else:
                # find all instances which match any of the attribute values
                priv = np.logical_or.reduce(np.equal.outer(vals, df[attr]))
                df.loc[priv, attr] = privileged_values[0]
                df.loc[~priv, attr] = unprivileged_values[0]

            privileged_protected_attributes.append(
                np.array(privileged_values, dtype=np.float64))
            unprivileged_protected_attributes.append(
                np.array(unprivileged_values, dtype=np.float64))

        full_name = dataset_name + ".csv"
        print("writing file: " + full_name)
        df.to_csv(full_name)
