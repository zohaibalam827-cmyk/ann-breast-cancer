"""Load the Breast Cancer Wisconsin dataset (saved as a CSV in data/)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "breast_cancer.csv"
LABEL = "target"  # 0 = malignant, 1 = benign


def save_csv(path=DATA_PATH):
    """Write the bundled dataset to a CSV file."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df[LABEL] = data.target
    path.parent.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_data(path=DATA_PATH):
    """Read the CSV (creating it first if needed). Returns X, y, feature names."""
    if not path.exists():
        save_csv(path)
    df = pd.read_csv(path)
    features = [c for c in df.columns if c != LABEL]
    return df[features].values, df[LABEL].values, features


def standardize(X_train, X_test):
    """Scale each feature to mean 0 and std 1, using the training data only.

    Neural networks train badly when features are on very different scales."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std
