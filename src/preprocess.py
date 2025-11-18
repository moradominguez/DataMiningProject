# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
except:
    RandomOverSampler = None
    SMOTE = None

TARGET_COL = "f_FPro_class"
ID_COLS = ["original_ID"]

RAW_CATEGORICAL = ["name", "store", "food category", "brand"]

NUMERIC_CANDIDATES = [
    "price", "price per cal", "package_weight",
    "Protein", "Total Fat", "Carbohydrate", "Sugars, total",
    "Fiber, total dietary", "Calcium", "Iron", "Sodium",
    "Cholesterol", "Fatty acids, total saturated"
]


def build_binary_target(y):
    return y.map(lambda x: 0 if x == 3 else 1)


def select_columns(df):
    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    categorical = [c for c in RAW_CATEGORICAL if c in df.columns]
    return numeric, categorical


def make_preprocess_pipeline(numeric_cols, categorical_cols, scale=True):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric_cols),
            ("cat", Pipeline(cat_steps), categorical_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )


def apply_imbalance(X, y, method=None):
    if method is None:
        return X, y
    if method == "smote" and SMOTE is not None:
        return SMOTE().fit_resample(X, y)
    if method == "ros" and RandomOverSampler is not None:
        return RandomOverSampler().fit_resample(X, y)
    return X, y


def preprocess_dataframe(df, imbalance="smote", scale=True):
    y = build_binary_target(df[TARGET_COL])

    numeric_cols, categorical_cols = select_columns(df)

    X_df = df.drop(columns=ID_COLS + [TARGET_COL])

    pre = make_preprocess_pipeline(numeric_cols, categorical_cols, scale=scale)
    X = pre.fit_transform(X_df)

    X, y = apply_imbalance(X, y, imbalance)

    feat_names = pre.get_feature_names_out()

    return X, y.to_numpy(), pre, feat_names
