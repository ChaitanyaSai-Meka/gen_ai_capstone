import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import sys
sys.path.append('..')
from preprocessing import preprocess_features


def train(df):
    df = preprocess_features(df)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    feature_names = list(X_train.columns)

    return model, feature_names, X_train, X_test, y_train, y_test
