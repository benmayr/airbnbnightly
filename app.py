from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# -----------------------------
# Configuration & Constants
# -----------------------------
DATA_CANDIDATES = [
    Path("notebooks/data/clean/listings.parquet"),
    Path("notebooks/data/clean/listings.parquet.gzip"),
    Path("notebooks/data/clean/listings.csv"),
    Path("data/clean/listings.parquet"),
    Path("data/clean/listings.csv"),
]
MODEL_PATH = Path("models/rf_price.pkl")
NYC_CENTER = {"lat": 40.7128, "lon": -74.0060}


# -----------------------------
# Cached Loaders
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_clean_dataset() -> pd.DataFrame:
    """Load the cleaned dataset, preferring parquet if present.
    If a CSV exists but no parquet, load CSV and write parquet for faster future loads.
    """
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "Could not find a clean dataset. Expected one of: "
            + ", ".join(str(p) for p in DATA_CANDIDATES)
        )

    if data_path.suffix.lower() in {".parquet", ".gzip"}:
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)
        # If numeric price is stored as string, clean it (defensive)
        if df["price"].dtype == object:
            df["price"] = (
                df["price"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
            ).astype(float)
        # Save parquet next to CSV for faster reloads
        try:
            parquet_target = data_path.with_suffix(".parquet")
            parquet_target.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_target, index=False)
        except Exception:
            pass

    # Minimal column presence check
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
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset missing required columns: {missing}")

    return df


@st.cache_resource(show_spinner=True)
def load_model():
    if not MODEL_PATH.exists():
        # Try to download model from a URL provided via secrets or environment
        model_url = None
        try:
            model_url = st.secrets.get("MODEL_URL", None)
        except Exception:
            model_url = None
        if not model_url:
            model_url = os.environ.get("MODEL_URL")
        if model_url:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            try:
                import urllib.request
                with urllib.request.urlopen(model_url, timeout=60) as resp, open(MODEL_PATH, "wb") as out:
                    out.write(resp.read())
            except Exception as e:
                raise FileNotFoundError(
                    f"Failed to download model from MODEL_URL: {model_url}. Error: {e}"
                )
        else:
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH} and no MODEL_URL provided in secrets/env.")
    model = joblib.load(MODEL_PATH)
    # Try to access feature names for alignment later
    feature_names = None
    try:
        feature_names = model.named_steps["model"].feature_names_in_
    except Exception:
        try:
            feature_names = model.feature_names_in_
        except Exception:
            feature_names = None
    return model, feature_names


# -----------------------------
# Feature Engineering for Inference
# -----------------------------
def bucket_min_nights(min_nights: float) -> str:
    if min_nights <= 1:
        return "min1"
    if min_nights <= 6:
        return "min2_6"
    return "min7p"


def build_feature_row(
    df: pd.DataFrame,
    neighbourhood_group: str,
    room_type: str,
    minimum_nights: int,
) -> pd.DataFrame:
    """Construct a single-row feature frame to match the training schema.
    Uses group medians for latitude/longitude/availability.
    """
    group_df = df[df["neighbourhood_group"] == neighbourhood_group]
    if group_df.empty:
        group_df = df

    lat = float(group_df["latitude"].median())
    lon = float(group_df["longitude"].median())
    avail = float(group_df["availability_365"].median())

    # Start numeric features
    feat = pd.DataFrame(
        {
            "availability_365": [avail],
            "latitude": [lat],
            "longitude": [lon],
        }
    )

    # Bucket and categorical one-hots
    bucket = bucket_min_nights(float(minimum_nights))

    cat = pd.DataFrame(
        {
            "neighbourhood_group": pd.Categorical([neighbourhood_group]),
            "room_type": pd.Categorical([room_type]),
            "min_nights_bucket": pd.Categorical([bucket]),
        }
    )

    dummies = pd.get_dummies(
        cat,
        columns=["neighbourhood_group", "room_type", "min_nights_bucket"],
        drop_first=False,
    )
    X = pd.concat([feat, dummies], axis=1).astype(float)
    return X


def align_features_to_model(X: pd.DataFrame, feature_names_in) -> pd.DataFrame:
    if feature_names_in is None:
        return X
    # ensure all required columns exist; fill missing with 0; drop extras; order columns
    X_aligned = X.reindex(columns=list(feature_names_in), fill_value=0.0)
    return X_aligned


# -----------------------------
# Demo Model Fallback (small, trained at startup if needed)
# -----------------------------
def build_training_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = np.log1p(df["price"].astype(float))
    base = pd.DataFrame(
        {
            "availability_365": df["availability_365"].astype(float),
            "latitude": df["latitude"].astype(float),
            "longitude": df["longitude"].astype(float),
        }
    )
    buckets = pd.Series(
        [bucket_min_nights(float(v)) for v in df["minimum_nights"]], index=df.index, name="min_nights_bucket"
    )
    cat = pd.DataFrame(
        {
            "neighbourhood_group": df["neighbourhood_group"].astype("category"),
            "room_type": df["room_type"].astype("category"),
            "min_nights_bucket": buckets.astype("category"),
        }
    )
    dummies = pd.get_dummies(cat, columns=["neighbourhood_group", "room_type", "min_nights_bucket"], drop_first=False)
    X = pd.concat([base, dummies], axis=1).astype(float)
    return X, y


def train_demo_model(df: pd.DataFrame) -> Tuple[Pipeline, np.ndarray]:
    # Keep it small for fast startup
    X, y = build_training_matrix(df)
    model = Pipeline([
        ("model", RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=42)),
    ])
    model.fit(X, y)
    # Persist for subsequent runs (best effort)
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    except Exception:
        pass
    # Extract feature names
    feature_names = None
    try:
        feature_names = model.named_steps["model"].feature_names_in_
    except Exception:
        feature_names = np.array(X.columns)
    return model, feature_names


# -----------------------------
# UI Helpers
# -----------------------------
def center_title(text: str):
    st.markdown(
        f"""
        <h2 style="text-align:center; font-weight:600;">{text}</h2>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="NYC Airbnb Price Forecaster", layout="wide")

center_title("NYC Airbnb Price Forecaster")

with st.spinner("Loading data and model..."):
    df = load_clean_dataset()
    try:
        model, feature_names_in = load_model()
    except FileNotFoundError:
        st.info("No saved model found. Training a small demo model now…")
        model, feature_names_in = train_demo_model(df)

# Sidebar controls
with st.sidebar:
    st.subheader("Filters")
    ng_options = sorted(df["neighbourhood_group"].dropna().astype(str).unique().tolist())
    rt_options = sorted(df["room_type"].dropna().astype(str).unique().tolist())

    selected_ng = st.selectbox("Neighbourhood Group", ng_options, index=0)
    selected_rt = st.selectbox("Room Type", rt_options, index=0)
    selected_min_nights = st.slider("Minimum Nights", min_value=1, max_value=30, value=3, step=1)

# Build a single feature row and predict
X_row = build_feature_row(df, selected_ng, selected_rt, selected_min_nights)
X_row = align_features_to_model(X_row, feature_names_in)

# Model predicts log-price; convert back to price with expm1
log_pred = model.predict(X_row)[0]
pred_price = float(np.expm1(log_pred))

# Main layout
col1, col2 = st.columns([1, 2])
with col1:
    st.metric(label="Predicted nightly price (USD)", value=f"$ {pred_price:,.0f}")

with col2:
    # Map: filter by selected neighbourhood_group for zoom center only
    focus_df = df[df["neighbourhood_group"] == selected_ng]
    if focus_df.empty:
        focus_df = df
    center_lat = float(focus_df["latitude"].median()) if not focus_df.empty else NYC_CENTER["lat"]
    center_lon = float(focus_df["longitude"].median()) if not focus_df.empty else NYC_CENTER["lon"]

    map_df = df.dropna(subset=["latitude", "longitude", "price"]).copy()
    map_df["log_price"] = np.log(map_df["price"].astype(float))

    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="log_price",
        size="availability_365",
        size_max=12,
        color_continuous_scale="Viridis",
        zoom=10,
        height=520,
        opacity=0.5,
        hover_data={"price": True, "availability_365": True, "neighbourhood": True, "room_type": True},
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
    )
    fig.update_layout(mapbox_center={"lat": center_lat, "lon": center_lon})

    st.plotly_chart(fig, use_container_width=True)

# Naïve 7-day forecast plot
multipliers = np.array([1.00, 1.02, 1.03, 1.01, 0.98, 0.97, 1.00], dtype=float)
base = pred_price
future_prices = base * multipliers

forecast_index = [datetime.today().date() + timedelta(days=i) for i in range(7)]
forecast_df = pd.DataFrame({"date": forecast_index, "predicted_price": future_prices})
forecast_df = forecast_df.set_index("date")

st.line_chart(forecast_df, height=260)

