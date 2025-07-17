import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


DATA_PATH = ".\dataset\car data.csv"    
TEST_SIZE = 0.1
RANDOM_STATE = 42
COLUMNS_TO_SCALE = ['Present_Price', 'Driven_kms']


def initial_clean(df):
    df = df.copy()
    df.drop(['Car_Name'], axis=1, inplace=True)
    df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)
    return df

def train_evaluate_models(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    train_r2 = r2_score(y_train, lr.predict(X_train))
    test_r2 = r2_score(y_test, y_pred_lr)
    test_mse = mean_squared_error(y_test, y_pred_lr)
    print(f"Linear Regression (train R^2): {train_r2:.4f}")
    print(f"Linear Regression (test R^2): {test_r2:.4f}")
    print(f"Linear Regression (test MSE): {test_mse:.4f}")
    plot_actual_vs_pred(y_test, y_pred_lr, "Linear Regression: Actual vs Predicted")

    # Lasso Regression (scaled)
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    print(f"Lasso Regression (test R^2): {r2_score(y_test, y_pred_lasso):.4f}")
    print(f"Lasso Regression (test MSE): {mean_squared_error(y_test, y_pred_lasso):.4f}")
    plot_actual_vs_pred(y_test, y_pred_lasso, "Lasso Regression: Actual vs Predicted")

    # Ridge Regression (scaled)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    print(f"Ridge Regression (test R^2): {r2_score(y_test, y_pred_ridge):.4f}")
    print(f"Ridge Regression (test MSE): {mean_squared_error(y_test, y_pred_ridge):.4f}")
    plot_actual_vs_pred(y_test, y_pred_ridge, "Ridge Regression: Actual vs Predicted")

def feature_engineer_v2(df):
    df = df.copy()
    df = pd.get_dummies(df, columns=['Car_Name','Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)
    df['Year_encoded'] = df['Year'] - df['Year'].min()
    df.drop('Year', axis=1, inplace=True)
    return df

def plot_actual_vs_pred(y_true, y_pred, title):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.7, color='blue', edgecolors='k')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(title)
    plt.grid(True)
    plt.show()

def train_model():
    df = pd.read_csv(DATA_PATH)
    df1 = initial_clean(df)
    X = df1[['Year', 'Present_Price', 'Driven_kms', 'Owner',
            'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Selling_type_Individual',
            'Transmission_Manual']]
    y = df1['Selling_Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Scale all features for penalties
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- MODEL PERFORMANCE ON CLASSICAL FEATURE SET ---")
    train_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

    # 2. Workflow with more one-hot and re-encoding of year (robust feature engineering)
    df2 = pd.read_csv(DATA_PATH)
    df2 = feature_engineer_v2(df2)
    X2 = df2.drop('Selling_Price', axis=1)
    y2 = df2['Selling_Price']

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Only scale numerical columns, keep dummies unchanged
    scaler2 = StandardScaler()
    columns_to_scale = [col for col in COLUMNS_TO_SCALE if col in X2.columns]
    X_train2[columns_to_scale] = scaler2.fit_transform(X_train2[columns_to_scale])
    X_test2[columns_to_scale] = scaler2.transform(X_test2[columns_to_scale])

    print("\n--- MODEL PERFORMANCE WITH FULL ONE-HOT & FEATURE ENCODING ---")
    # Linear Regression
    model2 = LinearRegression()
    model2.fit(X_train2, y_train2)
    y_pred2 = model2.predict(X_test2)
    print(f"Full Linear Regression (test R^2): {r2_score(y_test2, y_pred2):.4f}")
    print(f"Full Linear Regression (test MSE): {mean_squared_error(y_test2, y_pred2):.4f}")
    plot_actual_vs_pred(y_test2, y_pred2, "Extended Features: Linear Regression")

    # Lasso Regression
    lasso2 = Lasso(alpha=0.1)
    lasso2.fit(X_train2, y_train2)
    y_pred_lasso2 = lasso2.predict(X_test2)
    print(f"Full Lasso Regression (test R^2): {r2_score(y_test2, y_pred_lasso2):.4f}")
    print(f"Full Lasso Regression (test MSE): {mean_squared_error(y_test2, y_pred_lasso2):.4f}")
    plot_actual_vs_pred(y_test2, y_pred_lasso2, "Extended Features: Lasso Regression")

train_model()
