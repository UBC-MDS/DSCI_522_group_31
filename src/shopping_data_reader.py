# author: Tran Doan Khanh Vu
# date: 2020-12-04
""" Functions related to reading shopping data
"""

import pandas as pd

numerical_features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
categorical_features = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]
binary_features = ["Weekend"]
drop_features = []


def read_data_as_df(data_path):
    """read train and test data files located in the data_path

    Args:
        data_path (string): path containing files for train and test data

    Returns:
        (train_df, test_def) (DataFrame, DataFrame): a tuple consisting of
        train dataset and test dataset in pandas.DataFrame format

    Examples:
        train_df, test_df = read_data_as_df("data/processed")
    """
    train_df = pd.read_csv(data_path + "/train_data.csv")
    test_df = pd.read_csv(data_path + "/test_data.csv")
    return train_df, test_df


def read_data_as_xy(data_path):
    """read train and test data files located in the data_path and split into
     X and y

    Args:
        data_path (string): path containing files for train and test data

    Returns:
        (X_train, y_train, X_test, y_test) (DataFrame, DataFrame): a tuple
        consisting of X and y train dataset and test dataset in
        pandas.DataFrame format

    Examples:
        X_train, y_train, X_test, y_test = read_data_as_xy("data/processed")
    """
    train_df, test_df = read_data_as_df(data_path)

    X_train, y_train = train_df.drop(columns=["Revenue"]), train_df["Revenue"]
    X_test, y_test = test_df.drop(columns=["Revenue"]), test_df["Revenue"]

    return X_train, y_train, X_test, y_test
