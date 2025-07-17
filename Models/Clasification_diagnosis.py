import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE


DATA_PATH = ".\dataset\me_cfs_vs_depression_dataset.csv" 
TEST_SIZE = 0.2
RANDOM_STATE = 42
OBJECT_COLS = [
    "gender",
    "work_status",
    "social_activity_level",
    "exercise_frequency",
    "meditation_or_mindfulness",
]
TARGET_COL = "diagnosis"


def preprocess_data(df):
    df = df.copy()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df

def encode_features(df, target_col, object_cols):
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col])
    feature_encoder = LabelEncoder()
    for col in object_cols:
        df[col] = feature_encoder.fit_transform(df[col])
    return df, target_encoder

def plot_class_distribution(df, target_col):
    class_counts = df[target_col].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='Set2')
    plt.title('Diagnosis Class Distribution')
    plt.xlabel('Diagnosis Class')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def model_train():
    df = pd.read_csv(DATA_PATH)
    print("\nFirst 5 rows:\n", df.head())
    print("\nMissing values in each column:\n", df.isna().sum())
    plot_class_distribution(df, TARGET_COL)
    df = preprocess_data(df)
    print("\nMissing values after imputation:\n", df.isna().sum())
    df, target_encoder = encode_features(df, TARGET_COL, OBJECT_COLS)
    print("\nFirst 5 rows after encoding:\n", df.head())

    # Preparation
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"\nOriginal X_train shape: {X_train_scaled.shape}")
    print(f"After SMOTE: {X_resampled.shape}")

    # Model
    model = LogisticRegression()
    model.fit(X_resampled, y_resampled)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    print("\nModel Score (Test Set): {:.4f}".format(model.score(X_test_scaled, y_test)))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Accuracies
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

model_train()
