# author: Mai Le
# date: 2020-11-27
"""Reads the data from the data clean-up script, performs some statistical or machine learning analysis and summarizes the results as a figure(s) and a table(s).

Usage: src/04_build_model.py --data_path=<data_path> --out_report_path=<out_report_path> [--random_state=<random_state>] [--tune_params=<tune_params>]

Options:
--data_path=<data_path>                The path containing train & test dataset
--out_report_path=<out_report_path>    The path to export model scores in figures and tables
--random_state=<random_state>          The random state that we want to use for splitting. [default: 2020]
--tune_params=<tune_params>            Whether we need to tune hyperparameters or not
"""

# region import libraries
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_transformer,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    plot_confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

import scipy
from scipy.stats import randint, loguniform

import pickle

from sklearn.feature_selection import RFE, RFECV

import matplotlib.pyplot as plt
from docopt import docopt
import feather
import os

import pandas as pd
import numpy as np

# endregion


# region break-down functions
def read_data(data_path):
    """read train and test data files located in the data_path

    Args:
        data_path (string): path containing files for train and test data

    Returns:
        (train_df, test_def) (DataFrame, DataFrame): a tuple consisting of train dataset and test dataset in pandas.DataFrame format

    Examples:
        train_df, test_df = read_data("data/processed")
    """
    train_df = pd.read_csv(data_path + "/train_data.csv")
    test_df = pd.read_csv(data_path + "/test_data.csv")
    return train_df, test_df


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
            dummy, X_train, y_train, cv=10, scoring=scoring, return_train_score=True
        )
        store_cross_val_results(clf_score_map, "DummyClassifier", cv_scores, result_df)

    """
    d = dict()
    for s in score_map:
        d[s] = "{:0.4f}".format(np.mean(scores[s]))
    results_df[model_name] = d


def tune_hyperparams(preprocessor, X, y, random_state, tune_hyperparams=True):
    """tuning hyperparameters for LogisticRegression, RandomForestClassifier and SVC with preprocessor on X, y with random_state using RandomizedSearchCV

    Args:
        preprocessor (Pipeline/ColumnTransformer): a Pipeline to transform X
        X (DataFrame): features
        y (DataFrame): target
        random_state (integer): random state

    Returns:
        dict: a dictionary with key=model's name, value={"best_model":best_model, "best_params":best_params}

    Examples:
        perparams_best_model = tune_hyperparams(preprocessor, X_train, y_train, 2020)
    """
    if tune_hyperparams == True:
        classifiers = {
            "Logistic Regression": {
                "clf": LogisticRegression(
                    class_weight="balanced", random_state=random_state, max_iter=1000
                ),
                "param_dist": {"logisticregression__C": loguniform(1e-3, 1e3)},
            },
            "Random Forest": {
                "clf": RandomForestClassifier(
                    class_weight="balanced", random_state=random_state
                ),
                "param_dist": {
                    "randomforestclassifier__n_estimators": scipy.stats.randint(
                        low=10, high=300
                    ),
                    "randomforestclassifier__max_depth": scipy.stats.randint(
                        low=2, high=20
                    ),
                },
            },
            "SVC": {
                "clf": SVC(class_weight="balanced", random_state=random_state),
                "param_dist": {
                    "svc__gamma": [0.1, 1.0, 10, 100],
                    "svc__C": [0.1, 1.0, 10, 100],
                },
            },
        }
    else:
        classifiers = {
            "Logistic Regression": {
                "clf": LogisticRegression(
                    class_weight="balanced", random_state=random_state, max_iter=1000
                ),
                "param_dist": {"logisticregression__C": [0.008713608033492446]},
            },
            "Random Forest": {
                "clf": RandomForestClassifier(
                    class_weight="balanced", random_state=random_state
                ),
                "param_dist": {
                    "randomforestclassifier__n_estimators": [65],
                    "randomforestclassifier__max_depth": [12],
                },
            },
            "SVC": {
                "clf": SVC(class_weight="balanced", random_state=random_state),
                "param_dist": {
                    "svc__gamma": [0.1],
                    "svc__C": [1.0],
                },
            },
        }
    hyperparams_best_model = {}

    # find the best hyperparameters of each model
    for name, model_dict in classifiers.items():
        pipe = make_pipeline(preprocessor, model_dict["clf"])
        random_search = RandomizedSearchCV(
            pipe,
            param_distributions=model_dict["param_dist"],
            n_iter=10,
            verbose=1,
            n_jobs=-1,
            scoring="f1",
            cv=10,
            random_state=random_state,
        )
        random_search.fit(X, y)
        hyperparams_best_model[name] = {
            "best_model": random_search.best_estimator_,
            "best_params": random_search.best_params_,
        }
    return hyperparams_best_model


def find_best_model(
    hyperparams_best_model,
    X,
    y,
    random_state,
    path="../data/processed/model_selection_result.csv",
):
    """find the best model among 3 models using cross validation

    Args:
        hyperparams_best_model (dict): a dictionary with key=model's name, value={"best_model":best_model, "best_params":best_params}
        X (DataFrame): features
        y (DataFrame): target
        random_state (int): random_state

    Returns:
        (string, Pipeline, list): a tuple consisting of best model's name, best model object and its best hyperparameters

    Examples:
        best_model_name, best_model, best_params = find_best_model(hyperparams_best_model, X_train, y_train, 2020)
    """
    scoring = ["accuracy", "recall", "f1"]
    clf_score_map = [
        "fit_time",
        "score_time",
        "test_accuracy",
        "test_f1",
        "test_recall",
    ]
    results = {}
    for name, model in hyperparams_best_model.items():
        cv_scores = cross_validate(
            model["best_model"], X, y, cv=10, n_jobs=-1, scoring=scoring
        )
        store_cross_val_results(clf_score_map, name, cv_scores, results)
    results_df = pd.DataFrame(results).T
    print(results_df)
    results_df.reset_index().to_csv(path)
    best_model_name = results_df.iloc[np.argmax(results_df["test_f1"])].name
    best_model = hyperparams_best_model[best_model_name]["best_model"]
    best_params = hyperparams_best_model[best_model_name]["best_params"]
    return (best_model_name, best_model, best_params)


def retrieve_important_features(
    preprocessor,
    X_train,
    y_train,
    numerical_features,
    categorical_features,
    binary_features,
    random_state,
):
    """return a list of features that are important in predicting the target using RFE

    Args:
        preprocessor (ColumnTransformer): feature transformation
        X_train (DataFrame): train feature set
        y_train (DataFrame): train target
        random_state (int): random_state

    Returns:
        [list]: list of important features

    Examples:
        retrieve_important_features(preprocessor, X_train, y_train, 2020)
    """
    rfecv = make_pipeline(preprocessor, RFECV(Ridge(random_state=random_state), cv=10))
    rfecv.fit(X_train, y_train)
    ohe_columns = list(
        rfecv.named_steps["columntransformer"]
        .transformers_[2][1]
        .get_feature_names(categorical_features)
    )
    bin_columns = list(
        rfecv.named_steps["columntransformer"]
        .transformers_[3][1]
        .get_feature_names(binary_features)
    )
    new_columns = numerical_features + ohe_columns + bin_columns
    new_columns = np.array(new_columns)
    return new_columns[rfecv.named_steps["rfecv"].support_]


def plot_results(model, X, y, labels):
    """generate confusion matrix and classification report

    Args:
        model (Pipeline): ML model
        X (DataFrame): features
        y (DataFrame): features
        labels (list): a list of labels for the target classes

    Returns:
        (plot_object, classification_report): a plot object and a classification_report object

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


# endregion

# region unit tests
def test_read_data():
    # unit test for read_data function
    train_df, test_df = read_data("data/processed")
    assert (train_df.shape[0] != 0) and (test_df.shape[0] != 0), "Data size must be > 0"


def test_store_cross_val_results():
    # unit test for store_cross_val_results function
    result = {}
    score_map = ["f1"]
    cv_scores = {"f1": 0.2}
    store_cross_val_results(score_map, "test model", cv_scores, result)
    assert len(result) != 0, "Result table size must be > 0"


def test_tune_hyperparams():
    # unit test for tune_hyperparams function
    data_path = "../data/processed"
    train_df, _ = read_data(data_path)
    X_train, y_train = train_df.drop(columns=["Revenue"]), train_df["Revenue"]
    random_state = 2020

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

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (StandardScaler(), numerical_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
        (OneHotEncoder(handle_unknown="error", drop="if_binary"), binary_features),
    )

    # tuning hyperparameters for SVC, RandomForestClassifier and LogisticRegression
    hyperparams_best_model = tune_hyperparams(
        preprocessor, X_train, y_train, random_state
    )
    # TODO: use toy data set instead
    assert len(hyperparams_best_model) == 2, "Missing data of hyperparameter tuning"


def test_find_best_model():
    # unit test for find_best_model function

    data_path = "../data/processed"
    train_df, _ = read_data(data_path)
    X_train, y_train = train_df.drop(columns=["Revenue"]), train_df["Revenue"]
    random_state = 2020

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

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (StandardScaler(), numerical_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
        (OneHotEncoder(handle_unknown="error", drop="if_binary"), binary_features),
    )

    # tuning hyperparameters for SVC, RandomForestClassifier and LogisticRegression
    hyperparams_best_model = tune_hyperparams(
        preprocessor, X_train, y_train, random_state
    )

    _, best_model, _ = find_best_model(
        hyperparams_best_model, X_train, y_train, random_state
    )

    assert best_model is not None, "No best model is returned"


def test_save_plots():
    # unit test for save_plots function
    data_path = "../data/processed"
    best_model = pickle.load(open(data_path + "/best_model.sav", "rb"))
    train_df, _ = read_data(data_path)
    X_train, y_train = train_df.drop(columns=["Revenue"]), train_df["Revenue"]
    plot, class_report = plot_results(
        best_model, X_train, y_train, ["No-revenue", "Revenue"]
    )
    save_plots(
        "../img/reports/",
        plot,
        class_report,
        ("confusion_matrix_test", "classification_report_test"),
    )
    assert os.path.exists(
        "../img/reports/confusion_matrix_test.png"
    ), "No file was created"


def test_plot_results():
    # unit test for plot_results function
    data_path = "../data/processed"
    best_model = pickle.load(open(data_path + "/best_model.sav", "rb"))
    train_df, _ = read_data(data_path)
    X_train, y_train = train_df.drop(columns=["Revenue"]), train_df["Revenue"]
    plot, _ = plot_results(best_model, X_train, y_train, ["No-revenue", "Revenue"])
    assert plot is not None, "No plot was created"


def test_retrieve_important_features():
    # unit test for retrieve_important_features function
    data_path = "../data/processed"
    train_df, _ = read_data(data_path)
    X_train, y_train = train_df.drop(columns=["Revenue"]), train_df["Revenue"]
    random_state = 2020

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

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (StandardScaler(), numerical_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
        (OneHotEncoder(handle_unknown="error", drop="if_binary"), binary_features),
    )
    ls = retrieve_important_features(
        preprocessor,
        X_train,
        y_train,
        numerical_features,
        categorical_features,
        binary_features,
        random_state,
    )
    assert len(ls) != 0, "No feature is returned"


def run_all_tests():
    # execute all unit tests of all functions
    test_read_data()
    test_store_cross_val_results()
    test_tune_hyperparams()
    test_find_best_model()
    test_plot_results()
    test_retrieve_important_features()
    test_save_plots()


# endregion

# region main function
def main(data_path, out_report_path, random_state=2020, tune_params=True):
    # read the data files, split into X and y
    print("Start build_model script")
    print("Read the data files, split into X and y")
    random_state = int(random_state)
    tune_params = bool(tune_params)
    train_df, test_df = read_data(data_path)
    X_train, y_train = train_df.drop(columns=["Revenue"]), train_df["Revenue"]
    X_test, y_test = test_df.drop(columns=["Revenue"]), test_df["Revenue"]

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
    # endregion

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (StandardScaler(), numerical_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
        (OneHotEncoder(handle_unknown="error", drop="if_binary"), binary_features),
    )

    # tuning hyperparameters for SVC, RandomForestClassifier and LogisticRegression
    print(
        "Tuning hyperparameters for SVC, RandomForestClassifier and LogisticRegression"
    )
    hyperparams_best_model = tune_hyperparams(
        preprocessor, X_train, y_train, random_state
    )

    # find the best model
    print("Finding the best model using cross validation")
    _, best_model, _ = find_best_model(
        hyperparams_best_model,
        X_train,
        y_train,
        random_state,
        data_path + "/model_selection_result.csv",
    )

    # get result plots
    print("Creating plots and classification report")
    plot, class_report = plot_results(
        best_model, X_test, y_test, ["No-revenue", "Revenue"]
    )

    # save plots to report path
    print("Saving reports")
    save_plots(
        out_report_path,
        plot,
        class_report,
        ("confusion_matrix", "classification_report"),
    )

    # save model to disk
    print("Saving the best model for later use")
    pickle.dump(best_model, open(data_path + "/best_model.sav", "wb"))

    # try feature selection
    print("Building the model with RFE")

    fs_model = make_pipeline(
        preprocessor,
        RFECV(Ridge(random_state=random_state), cv=10),
        RandomForestClassifier(max_depth=12, n_estimators=275),
    )
    fs_model.fit(X_train, y_train)

    plot, class_report = plot_results(
        fs_model, X_test, y_test, ["No-revenue", "Revenue"]
    )

    # save plots to report path
    print("Saving reports")
    save_plots(
        out_report_path,
        plot,
        class_report,
        (
            "confusion_matrix_feature_selection",
            "classification_report_feature_selection",
        ),
    )
    print("End build_model script")
    return


if __name__ == "__main__":
    opt = docopt(__doc__)
    main(
        opt["--data_path"],
        opt["--out_report_path"],
        opt["--random_state"],
        opt["--tune_params"],
    )

# endregion
