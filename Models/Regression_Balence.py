import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_PATH = ".\dataset\Credit_Data.csv"   
RANDOM_STATE = 42
COLUMNS_TO_SCALE = ['Income', 'Limit', 'Rating']


def preprocess_credit_data(df, drop_col, scale_cols):
    df = pd.get_dummies(df, columns=['Gender', 'Student', 'Married', 'Ethnicity'], drop_first=True)
    X = df.drop(drop_col, axis=1)
    y = df[drop_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    intersect = [col for col in scale_cols if col in X.columns]
    X_train[intersect] = scaler.fit_transform(X_train[intersect])
    X_test[intersect] = scaler.transform(X_test[intersect])
    return X_train, X_test, y_train, y_test

def evaluate_and_plot(y_test, y_pred, title, xlab, ylab):
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.show()

def regressor_workflow_balance():
    print("\n--- Predicting Balance ---")
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = preprocess_credit_data(df, 'Balance', COLUMNS_TO_SCALE)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    print("\nLinear Regression:")
    evaluate_and_plot(y_test, y_pred_lr, "Actual vs Predicted Balance (Linear Regression)", "Actual Balance", "Predicted Balance")

    # Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    print("\nRidge Regression:")
    evaluate_and_plot(y_test, y_pred_ridge, "Actual vs Predicted Balance (Ridge)", "Actual Balance", "Predicted Balance")

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    print("\nRandom Forest Regression:")
    evaluate_and_plot(y_test, y_pred_rf, "Actual vs Predicted Balance (Random Forest)", "Actual Balance", "Predicted Balance")

def regressor_workflow_rating():
    print("\n--- Predicting Rating ---")
    df = pd.read_csv(DATA_PATH)
    df = pd.get_dummies(df, columns=['Gender', 'Student', 'Married', 'Ethnicity'], drop_first=True)
    X = df.drop('Rating', axis=1)
    y = df['Rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    intersect = [col for col in COLUMNS_TO_SCALE if col in X.columns]
    X_train[intersect] = scaler.fit_transform(X_train[intersect])
    X_test[intersect] = scaler.transform(X_test[intersect])

    # Lasso Regression
    lasso_model = Lasso()
    lasso_model.fit(X_train, y_train)
    y_pred_lasso = lasso_model.predict(X_test)
    print("\nLasso Regression (for Rating):")
    evaluate_and_plot(y_test, y_pred_lasso, "Actual vs Predicted Rating (Lasso)", "Actual Rating", "Predicted Rating")
    # Training R²
    train_pred = lasso_model.predict(X_train)
    train_r2 = r2_score(y_train, train_pred)
    print(f"Lasso Regression training R²: {train_r2:.4f}")


regressor_workflow_balance()
regressor_workflow_rating()
