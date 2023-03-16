from datetime import date
import itertools
import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error

FEATURE_COLS = ['INPUT1', 'INPUT2', 'INPUT3', 'INPUT4', 'INPUT5', 'INPUT6', 'INPUT7', 'INPUT8', 'INPUT9', 'INPUT10']
Y_COL = 'OUTPUT'

def _gen_possible_inputs():
    outs = []
    ins = [c for c in FEATURE_COLS if c != 'INPUT7']
    for length in range(1, len(ins)+1):
        for subset in itertools.combinations(ins, length):
            outs.append(subset)
    return outs

def _select_features(X_train, y_train, X_test, score_func):
    # configure to select all features
    fs = SelectKBest(score_func=score_func, k='all')
    # Learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def _plot_linear_correlation_scores(train_df, test_df):
    # f_regression: Univariate linear regression tests returning F-statistic and p-values.
    # Quick linear model for testing the effect of a single regressor, sequentually for many regressors.
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[Y_COL].values
    X_test = test_df[FEATURE_COLS].values

    # feature selection
    X_train_fs, X_test_fs, fs = _select_features(X_train, y_train, X_test, f_regression())
    # what are scores for the features
    print("--- Correlation ---")
    for i in range(len(fs.scores_)):
        print("Feature %d: %f" % (i+1, fs.scores_[i]))
    # plot the scores
    pyplot.bar([i for i in range(len(fs.scores_[i])))
    pyplot.show()

def load_train_data():
    return pd.read_csv("TraningData.csv")

def load_test_data():
    return pd.read_csv("TestSet.csv")


def train_model(train_df, features, degree=4):
    X_all = train_df[features].values
    y_all = train_df[[Y_COL]].values
    # fit the model
    if degree > 1:
        poly = PolynomialFeatures(degree=degree)
        X_all = poly.fit_transform(X_all)
        model = LinearRegression()
    else:
        model = HuberRegressor(epsilon=1.3)
    model.fit(X_all, y_all)

    # evaluate the model
    yhat = model.predict(X_all)
    # evaluate predictions with MSE
    mse = mean_squared_error(y_all, yhat)
    print("MSE: %.3f; %s" % (mse, str(features)))
    return model


def regression_flow(degree=4):
    train_df = load_train_data()
    test_df = load_test_data()

    # INPUT7 is a categorical (0/1) input feature, hence split the training into two models
    train_df_0 = train_df[train_df['INPUT7'] == 0]
    test_df_0 = test_df[test_df['INPUT7'] == 0]

    train_df_1 = train_df[train_df['INPUT7'] == 1]
    test_df_1 = test_df[test_df['INPUT7'] == 1]

    features_0 = ['INPUT1', 'INPUT2', 'INPUT3', 'INPUT4', 'INPUT5', 'INPUT8', 'INPUT9']
    features_1 = ['INPUT1', 'INPUT2', 'INPUT3', 'INPUT4', 'INPUT5', 'INPUT6']

    # Train per category value
    model_0 = train_model(train_df_0, features_0, degree=degree)
    model_1 = train_model(train_df_1, features_1, degree=degree)

    # Predict
    # make polynomial inputs
    poly = PolynomialFeatures(degree=degree)
    X_pred_0 = poly.fit_transform(test_df_0[features_0].values)
    X_pred_1 = poly.fit_transform(test_df_1[features_1].values)

    yhat_0 = model_0.predict(X_pred_0)
    yhat_1 = model_1.predict(X_pred_1)

    test_df_0['Y_test'] = yhat_0
    test_df_1['Y_test'] = yhat_1
    final_test_df = pd.concat([test_df_0, test_df_1])

    # Labelled outputs
    print(train_df['OUTPUT'].describe())
    # Predicted outputs
    print(final_test_df['Y_test'].describe())

    return final_test_df


if __name__ == '__main__':
    regression_flow(degree=4)
