import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append('..')
from preprocessing import preprocess_features


def train(df):
    """Train a Decision Tree model and return the results"""

    df = preprocess_features(df)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    feature_names = list(X_train.columns)

    return model, feature_names, X_train, X_test, y_train, y_test
