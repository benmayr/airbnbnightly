from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib


def load_clean_data(path: str) -> pd.DataFrame:
    """Load listings data and apply schema/cleaning.

    - Keeps columns: id, neighbourhood_group, neighbourhood, room_type,
      minimum_nights, price, availability_365, latitude, longitude
    - Cleans price by stripping '$' and ',' and converts to float
    - Drops rows with price == 0 or price > 2000
    """
    required_cols = [
        "id",
        "neighbourhood_group",
        "neighbourhood",
        "room_type",
        "minimum_nights",
        "price",
        "availability_365",
        "latitude",
        "longitude",
    ]

    df = pd.read_csv(path, compression="infer", low_memory=False)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    use = df[required_cols].copy()

    # Price cleaning
    use["price"] = (
        use["price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    ).astype(float)

    # Filter by price bounds
    use = use[(use["price"] > 0) & (use["price"] <= 2000)].copy()

    # Reset index for cleanliness
    return use.reset_index(drop=True)


def _bucket_minimum_nights(min_nights: pd.Series) -> pd.Series:
    """Bucket minimum nights into: 1, 2-6, 7+"""
    bins = [-np.inf, 1, 6, np.inf]
    labels = ["min1", "min2_6", "min7p"]
    return pd.cut(min_nights.astype(float), bins=bins, labels=labels)


def make_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct model features and target from cleaned dataframe.

    Features:
      - one-hot: neighbourhood_group, room_type
      - bucketed minimum_nights (1, 2-6, 7+) as one-hot
      - availability_365
      - latitude, longitude
    Target:
      - log_price = log1p(price)
    """
    if not {"price", "neighbourhood_group", "room_type", "minimum_nights", "availability_365", "latitude", "longitude"}.issubset(
        df.columns
    ):
        raise ValueError("Input dataframe missing required columns for feature building")

    # Target
    y = np.log1p(df["price"].astype(float))

    # Base numeric
    feats = pd.DataFrame(
        {
            "availability_365": df["availability_365"].astype(float),
            "latitude": df["latitude"].astype(float),
            "longitude": df["longitude"].astype(float),
        }
    )

    # Bucketed minimum nights
    feats["min_nights_bucket"] = _bucket_minimum_nights(df["minimum_nights"])

    # Categorical one-hots
    cat_df = pd.DataFrame(
        {
            "neighbourhood_group": df["neighbourhood_group"].astype("category"),
            "room_type": df["room_type"].astype("category"),
            "min_nights_bucket": feats["min_nights_bucket"].astype("category"),
        }
    )

    dummies = pd.get_dummies(cat_df, columns=["neighbourhood_group", "room_type", "min_nights_bucket"], drop_first=False)

    # Combine numeric + dummies
    X = pd.concat([feats.drop(columns=["min_nights_bucket"]), dummies], axis=1)
    X = X.astype(float)

    return X, y


def train_random_forest(X: pd.DataFrame, y: pd.Series, *, random_state: int = 42) -> Pipeline:
    """Train a RandomForest wrapped in a sklearn Pipeline."""
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )

    pipe = Pipeline([
        ("model", rf),
    ])

    pipe.fit(X, y)
    return pipe


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate model, returning MAE and SMAPE computed on price scale."""
    log_pred = model.predict(X_test)
    price_true = np.expm1(y_test.to_numpy())
    price_pred = np.expm1(log_pred)

    mae = mean_absolute_error(price_true, price_pred)
    smape = _smape(price_true, price_pred)
    return {"mae": float(mae), "smape": float(smape)}


def save(model: Pipeline, path: str = "models/rf_price.pkl") -> str:
    """Persist trained model to disk using joblib. Creates parent directories."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path_obj)
    return str(path_obj)


if __name__ == "__main__":
    # Simple train script using default data path
    import argparse

    parser = argparse.ArgumentParser(description="Train RandomForest price model")
    parser.add_argument("--data", default="notebooks/data/clean/listings.csv", help="Path to listings CSV or CSV.GZ")
    parser.add_argument("--model_out", default="models/rf_price.pkl", help="Where to save the trained model")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df = load_clean_data(args.data)
    X, y = make_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=args.random_state)

    model = train_random_forest(X_train, y_train, random_state=args.random_state)
    metrics = evaluate(model, X_test, y_test)

    out_path = save(model, args.model_out)

    print("Metrics:", metrics)
    print("Saved model to:", out_path)
