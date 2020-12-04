# author: Tran Doan Khanh Vu
# date: 2020-12-04
""" Utils function
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    plot_confusion_matrix,
)

# region categorize features into different types
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


def store_cross_val_results(score_map, model_name, scores, results_df):
    """
    Adapted from DSCI-573 lab1 solution
    Stores mean scores from cross_validate in results_dict for
    the given model model_name.

    Parameters
    ----------
    score_map:
        list of score names as column headers
    model_name :
        scikit-learn classification model
    scores : dict
        object return by `cross_validate`
    results_dict: dict
        dictionary to store results

    Returns
    ----------
        None

    Examples
    ----------
        clf_score_map = [
            "fit_time",
            "score_time",
            "test_accuracy",
            "train_accuracy",
            "test_f1",
            "train_f1",
        ]
        result_df = {}
        scoring = ["accuracy", "f1"]
        dummy = DummyClassifier(strategy="stratified")
        cv_scores = cross_validate(
            dummy, X_train, y_train, cv=10, scoring=scoring,
            return_train_score=True
        )
        store_cross_val_results(clf_score_map, "DummyClassifier", cv_scores,
            result_df)

    """
    d = dict()
    for s in score_map:
        d[s] = "{:0.4f}".format(np.mean(scores[s]))
    results_df[model_name] = d


def plot_results(model, X, y, labels):
    """generate confusion matrix and classification report

    Args:
        model (Pipeline): ML model
        X (DataFrame): features
        y (DataFrame): features
        labels (list): a list of labels for the target classes

    Returns:
        (plot_object, classification_report): a plot object and a
        classification_report object

    Examples:
        plot_results(best_model, X_test, y_test, ["No-revenue", "Revenue"])
    """
    confusion_matrix = plot_confusion_matrix(
        model,
        X,
        y,
        display_labels=labels,
        values_format="d",
        cmap=plt.cm.Blues,
    )
    return confusion_matrix, classification_report(
        y, model.predict(X), target_names=labels, output_dict=True
    )


def save_plots(filepath, plot, class_report, filenames):
    """save plot as a png figure, confusion_matrix as a feather file

    Args:
        plot (Plot): confusion matrix
        class_report (classification_report): classification report
        filepath (string): path to contain the result

    Returns:
        None
    Example:
        save_plots(plot, class_report, "img/reports")

    """
    plot.figure_.savefig(filepath + "/" + filenames[0])
    df = pd.DataFrame(class_report).T.reset_index()
    df.to_csv(filepath + "/" + filenames[1] + ".csv")
