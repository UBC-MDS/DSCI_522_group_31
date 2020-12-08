# author: Tran Doan Khanh Vu
# date: 2020-12-04
""" Utils function
"""
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    plot_confusion_matrix,
)


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
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    plot.figure_.savefig(filepath + "/" + filenames[0])
    df = pd.DataFrame(class_report).T.reset_index()
    df.to_csv(filepath + "/" + filenames[1] + ".csv")
