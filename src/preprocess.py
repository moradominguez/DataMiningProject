# src/preprocess.py
"""
Preprocessing utilities:
- build_binary_target: f_FPro_class -> binary
- select_columns: numeric & categorical feature sets
- make_preprocess_pipeline: imputing + scaling + one-hot
"""

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
    "Cholesterol", "Fatty acids, total saturated",
]


def build_binary_target(y: pd.Series) -> pd.Series:
    """Binary target: 1 = non-UPF (0,1,2), 0 = UPF (3)."""
    return y.map(lambda x: 0 if x == 3 else 1)


def select_columns(df: pd.DataFrame):
    """Return lists of numeric and categorical feature columns (present in df)."""
    numeric = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    categorical = [c for c in RAW_CATEGORICAL if c in df.columns]
    return numeric, categorical


def make_preprocess_pipeline(numeric_cols, categorical_cols, scale=True):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric_cols),
            ("cat", Pipeline(cat_steps), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
