# author: Tran Doan Khanh Vu
# date: 2020-12-04
"""Perform unit tests for methods in 04_build_model.py
"""
import utils
import pickle
import os

from sklearn.compose import (
    make_column_transformer,
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
import shopping_data_reader as sdr

build_model = __import__('../04_build_model')

DATA_PATH = "data/processed"


def test_read_data():
    # unit test for read_data function
    train_df, test_df = sdr.read_data("data/processed")
    assert (train_df.shape[0] != 0) and (test_df.shape[0] != 0), \
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
    X_train, y_train, _, _ = sdr.read_data_xy(DATA_PATH)
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
        preprocessor, X_train, y_train, random_state
    )
    # TODO: use toy data set instead
    assert len(hyperparams_best_model) == 2, \
        "Missing data of hyperparameter tuning"


def test_find_best_model():
    # unit test for find_best_model function

    X_train, y_train, _, _ = sdr.read_data_xy(DATA_PATH)

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
    hyperparams_best_model = build_model.get_models(
        preprocessor, X_train, y_train, random_state
    )

    _, best_model, _ = build_model.find_best_model(
        hyperparams_best_model, X_train, y_train, random_state
    )

    assert best_model is not None, "No best model is returned"


def test_save_plots():
    # unit test for save_plots function
    best_model = pickle.load(open(DATA_PATH + "/best_model.sav", "rb"))
    X_train, y_train, _, _ = sdr.read_data_xy(DATA_PATH)

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
    X_train, y_train, _, _ = sdr.read_data_xy(DATA_PATH)

    plot, _ = utils.plot_results(
        best_model, X_train, y_train, ["No-revenue", "Revenue"]
    )
    assert plot is not None, "No plot was created"


def test_retrieve_important_features():
    # unit test for retrieve_important_features function
    X_train, y_train, _, _ = sdr.read_data_xy(DATA_PATH)
    random_state = 2020

    # region categorize features into different types
    numerical_features = utils.numerical_features
    categorical_features = utils.categorical_features
    binary_features = utils.binary_features
    drop_features = utils.drop_features
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
