# Copyright (c) 2020 Microsoft
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from azureml.core import Run
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


import warnings
warnings.filterwarnings("ignore", message='l1_reg="auto" is deprecated')
warnings.filterwarnings(
    "ignore", message='Using a non-tuple sequence for multidimensional indexing is deprecated')


# Create logger from current context
# If this is run in AML Compute, it will return the current run
# If this runs locally, it'll only print metrics to standard out
run = Run.get_context()


def load_train_test_dataset(dataset_name):
    # Load DF from CSV
    diabetes_df = run.input_datasets[dataset_name].to_pandas_dataframe()

    # Split out X and Y variables
    y = diabetes_df.pop('target').values
    X = diabetes_df

    column_names = X.columns

    # Split training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=0)

    print(
        f"Data contains {len(X_train)} training samples and {len(X_test)} test samples")
    return X_train, X_test, y_train, y_test, column_names


def train_elasticnet(X, y, alpha, l1_ratio):
    run.log('model_type', 'ElasticNet')

    # Create a new model object
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                       copy_X=True, random_state=40760)

    # Fit the model and return
    model.fit(X, y)

    # Save model to "outputs" folder
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/model.pkl')

    return model


def train_ridge_regression(X, y, alpha):
    run.log('model_type', "Ridge")

    # Create a new model object
    model = Ridge(alpha=alpha)

    # Fit the model and return
    model.fit(X, y)

    # Save model to "outputs" folder
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/model.pkl')

    return model


def train_gradient_boosted_regressor(X, y, alpha):
    run.log('model_type', "GradientBoostingRegressor")

    model = GradientBoostingRegressor(loss='huber', alpha=alpha)

    # Fit the model and return
    model.fit(X, y)

    # Save model to "outputs" folder
    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(value=model, filename='./outputs/model.pkl')

    return model


def predict_and_log_performance(model, X_test, y_test):
    # Get the predicted values
    preds = model.predict(X_test)

    # Log the metrics to the AML run
    run.log("rmse", np.sqrt(mean_squared_error(y_test, preds)))
    run.log('mae', mean_absolute_error(y_test, preds))
    run.log('r2', r2_score(y_test, preds))

    return preds


def plot_residuals_v_actuals(y, y_hat):
    """Residuals (y-axis) vs. Actuals (x-axis) - colored green"""
    resids = y - y_hat

    fig = plt.figure()
    sns.regplot(y, resids, color='g')

    plt.title('Residual vs. Actual')
    plt.xlabel("Actual Value")
    plt.ylabel("Residuals")

    plt.close(fig)

    run.log_image(name='residuals-v-actuals', plot=fig)
    return fig


def plot_predictions(y, y_hat):
    """Predictions (y-axis) vs. Actuals (x-axis)"""
    fig = plt.figure()

    sns.regplot(y, y_hat, color='b')

    plt.title("Prediction vs. Actual")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")

    plt.close(fig)
    run.log_image(name='prediction-v-actual', plot=fig)
    return fig


def plot_resid_histogram(y, y_hat):
    resids = y - y_hat

    fig = plt.figure()
    sns.distplot(resids, color='g')

    plt.title("Residual Histogram")

    plt.close(fig)

    run.log_image(name='residuals-histogram', plot=fig)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', '-d', type=str, default='data',
                        help="The path where the data file is located")
    parser.add_argument('--alpha', '-a', type=float,
                        help="The alpha value for training", default=0.03)
    parser.add_argument('--file-name', '-f', type=str, default='diabetes.csv',
                        help="The file name of the diabetes csv dataset")
    parser.add_argument('--l1-ratio', type=float, default=0.05,
                        help='The l1_ratio of the Scikit-Learn ElasticNet model')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001,
                        help='The learning rate for the GBT model')
    parser.add_argument('--model-name', '-n', type=str, default="ridge",
                        help="The name of the model to try. Supported values are 'ridge', 'elastic', and 'gbt'")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, column_names = load_train_test_dataset(
        args.dataset_name)

    if args.model_name == "ridge":
        # log the hyperparameters
        run.log('alpha', args.alpha)

        # set the arguments and algorithm choice
        args = {'alpha': args.alpha}
        algo = train_ridge_regression

    elif args.model_name == "elastic":
        # log the hyperparameters
        run.log('alpha', args.alpha)
        run.log('l1_ratio', args.l1_ratio)

        # set the arguments and algorithm choice
        args = {'alpha': args.alpha,
                'l1_ratio': args.l1_ratio}
        algo = train_elasticnet

    elif args.model_name == 'gbt':
        # log the hyperparameters
        run.log('alpha', args.alpha)
        run.log('l1_ratio', args.l1_ratio)

        # set the arguments and algorithm choice
        args = {'alpha': args.alpha}
        algo = train_gradient_boosted_regressor

    # Train the model
    model = algo(X=X_train,
                 y=y_train,
                 **args)

    # Generate predictions and log performance metrics
    preds = predict_and_log_performance(model=model,
                                        X_test=X_test,
                                        y_test=y_test)

    # Plot the residuals
    resid_fig = plot_residuals_v_actuals(y_test, preds)
    resid_hist = plot_resid_histogram(y_test, preds)
    pred_plt = plot_predictions(y_test, preds)
