# author: Tran Doan Khanh Vu
# date: 2020-12-04
"""Perform unit tests for methods in 04_build_model.py
"""
import pickle
import os
import sys

from sklearn.compose import (
    make_column_transformer,
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
build_model = __import__('04_build_model')
utils = __import__('utils')
sdr = __import__('shopping_data_reader')

DATA_PATH = "data/processed"


def main():
    run_all_tests()


def test_read_data_as_df():
    # unit test for read_data_as_df function
    train_df, test_df = sdr.read_data_as_df("data/processed")
    assert (train_df.shape[0] != 0) and (test_df.shape[0] != 0), \
        "Data size must be > 0"


def test_read_data_as_xy():
    # unit test for read_data_as_df function
    X_train, y_train, X_test, y_test = sdr.read_data_as_xy("data/processed")
    assert (X_train.shape[0] != 0) and (y_train.shape[0] != 0) and \
        (X_test.shape[0] != 0) and (y_test.shape[0] != 0), \
        "Data size must be > 0"


def test_store_cross_val_results():
    # unit test for store_cross_val_results function
    result = {}
    score_map = ["f1"]
    cv_scores = {"f1": 0.2}
    utils.store_cross_val_results(score_map, "test model", cv_scores, result)
    assert len(result) != 0, "Result table size must be > 0"


def test_tune_hyperparams():
    # unit test for tune_hyperparams function
    X_train, y_train, _, _ = sdr.read_data_as_xy(DATA_PATH)
    random_state = 2020

    preprocessor = make_column_transformer(
        ("drop", sdr.drop_features),
        (StandardScaler(), sdr.numerical_features),
        (OneHotEncoder(handle_unknown="ignore"), sdr.categorical_features),
        (OneHotEncoder(handle_unknown="error", drop="if_binary"),
            sdr.binary_features),
    )

    # tuning hyperparameters for SVC, RandomForestClassifier and
    # LogisticRegression
    hyperparams_best_model = build_model.tune_hyperparams(
        preprocessor, X_train, y_train, random_state, False
    )
    # TODO: use toy data set instead

    assert len(hyperparams_best_model) == 6, \
        "Missing data of hyperparameter tuning"


def test_find_best_model():
    # unit test for find_best_model function

    X_train, y_train, _, _ = sdr.read_data_as_xy(DATA_PATH)

    random_state = 2020

    preprocessor = make_column_transformer(
        ("drop", sdr.drop_features),
        (StandardScaler(), sdr.numerical_features),
        (OneHotEncoder(handle_unknown="ignore"), sdr.categorical_features),
        (OneHotEncoder(handle_unknown="error", drop="if_binary"),
            sdr.binary_features),
    )

    # tuning hyperparameters for SVC, RandomForestClassifier and
    # LogisticRegression
    hyperparams_best_model = build_model.tune_hyperparams(
        preprocessor, X_train, y_train, random_state, False
    )

    _, best_model, _ = build_model.find_best_model(
        hyperparams_best_model, X_train, y_train, random_state,
        DATA_PATH + "/model_selection_result.csv"
    )

    assert best_model is not None, "No best model is returned"


def test_save_plots():
    # unit test for save_plots function
    best_model = pickle.load(open(DATA_PATH + "/best_model.sav", "rb"))
    X_train, y_train, _, _ = sdr.read_data_as_xy(DATA_PATH)

    plot, class_report = utils.plot_results(
        best_model, X_train, y_train, ["No-revenue", "Revenue"]
    )

    utils.save_plots(
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
    best_model = pickle.load(open(DATA_PATH + "/best_model.sav", "rb"))
    X_train, y_train, _, _ = sdr.read_data_as_xy(DATA_PATH)

    plot, _ = utils.plot_results(
        best_model, X_train, y_train, ["No-revenue", "Revenue"]
    )
    assert plot is not None, "No plot was created"


def test_retrieve_important_features():
    # unit test for retrieve_important_features function
    X_train, y_train, _, _ = sdr.read_data_as_xy(DATA_PATH)
    random_state = 2020

    # region categorize features into different types
    numerical_features = sdr.numerical_features
    categorical_features = sdr.categorical_features
    binary_features = sdr.binary_features
    drop_features = sdr.drop_features
    # endregion

    preprocessor = make_column_transformer(
        ("drop", drop_features),
        (StandardScaler(), numerical_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
        (OneHotEncoder(handle_unknown="error", drop="if_binary"),
            binary_features),
    )
    ls = build_model.retrieve_important_features(
        preprocessor,
        X_train,
        y_train,
        numerical_features,
        categorical_features,
        binary_features,
        random_state,
    )
    assert len(ls) != 0, "No feature is returned"


def test_create_logistic_regression_model():
    model = build_model.create_logistic_regression_model(2000, False, True)["clf"]
    params = model.get_params()
    assert isinstance(model, LogisticRegression), \
        "Object is not a LogisticRegression instance"
    assert(params["random_state"] == 2000), "Random state is different"
    assert(params["class_weight"] == "balanced"), "Class weight is different"


def test_create_random_forest_model():
    model = build_model.create_random_forest_model(2000, False, True)["clf"]
    params = model.get_params()
    assert isinstance(model, RandomForestClassifier), \
        "Object is not a RandomForestClassifier instance"
    assert(params["random_state"] == 2000), "Random state is different"
    assert(params["class_weight"] == "balanced"), "Class weight is different"


def test_create_SVC_model():
    model = build_model.create_SVC_model(2000, False, True)["clf"]
    params = model.get_params()
    assert isinstance(model, SVC), \
        "Object is not a SVC instance"
    assert(params["random_state"] == 2000), "Random state is different"
    assert(params["class_weight"] == "balanced"), "Class weight is different"


def run_all_tests():
    # execute all unit tests of all functions
    # shopping_data_reader
    test_read_data_as_df()
    test_read_data_as_xy()

    # utils
    test_store_cross_val_results()
    test_plot_results()
    test_save_plots()

    # build models
    test_create_logistic_regression_model()
    test_create_random_forest_model()
    test_create_SVC_model()
    test_tune_hyperparams()
    test_find_best_model()
    test_retrieve_important_features()


if __name__ == "__main__":
    main()
