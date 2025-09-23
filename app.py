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
import altair as alt
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
# Uncertainty Quantification
# -----------------------------
@st.cache_data(show_spinner=False)
def get_rf_quantiles(model_hash: str, X_row_values: tuple) -> Tuple[float, float, float]:
    """Compute P10, P50, P90 quantiles from RandomForest per-tree predictions.
    
    Args:
        model_hash: Hashable string identifier for the model
        X_row_values: Tuple of feature values (hashable)
        
    Returns:
        Tuple of (p10, p50, p90) in price space (not log space)
    """
    # Get the model from the global cache (it's already cached with @st.cache_resource)
    # We'll access it through the model loading function
    model, _ = load_model()
    
    # Extract the actual RandomForest from Pipeline if needed
    rf = model
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        rf = model.named_steps['model']
    
    # Check if it's a tree ensemble
    if not hasattr(rf, 'estimators_'):
        raise ValueError("Model is not a tree ensemble (no estimators_ attribute)")
    
    # Reconstruct X_row from the hashable values
    X_row = pd.DataFrame([X_row_values])
    
    # Get per-tree predictions (in log space)
    tree_predictions = []
    for tree in rf.estimators_:
        tree_pred = tree.predict(X_row)[0]
        tree_predictions.append(tree_pred)
    
    # Convert to price space
    price_predictions = np.expm1(tree_predictions)
    
    # Compute quantiles
    p10 = float(np.percentile(price_predictions, 10))
    p50 = float(np.percentile(price_predictions, 50))
    p90 = float(np.percentile(price_predictions, 90))
    
    return p10, p50, p90


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

# Compute uncertainty intervals
try:
    # Create hashable parameters for caching
    model_hash = f"model_{hash(str(MODEL_PATH))}"
    X_row_values = tuple(X_row.iloc[0].values)
    
    p10, p50, p90 = get_rf_quantiles(model_hash, X_row_values)
    uncertainty_available = True
except ValueError as e:
    # Model is not a tree ensemble, fall back to point prediction
    p10 = p50 = p90 = pred_price
    uncertainty_available = False

# Main layout
col1, col2 = st.columns([1, 2])
with col1:
    st.metric(label="Predicted nightly price (USD)", value=f"$ {pred_price:,.0f}")
    
    # Display uncertainty summary
    if uncertainty_available:
        st.caption(f"Predicted = ${p50:,.0f} (P10–P90: ${p10:,.0f}–${p90:,.0f})")
    else:
        st.caption("Uncertainty intervals not available for this model type")

with col2:
    # Map: filter by selected sidebar filters
    filtered_df = df[
        (df["neighbourhood_group"] == selected_ng) & 
        (df["room_type"] == selected_rt)
    ].dropna(subset=["latitude", "longitude", "price"]).copy()
    
    if filtered_df.empty:
        # Fallback to just neighbourhood_group if no exact matches
        filtered_df = df[df["neighbourhood_group"] == selected_ng].dropna(subset=["latitude", "longitude", "price"]).copy()
    
    if filtered_df.empty:
        # Final fallback to all data
        filtered_df = df.dropna(subset=["latitude", "longitude", "price"]).copy()
    
    # Center map on filtered data
    center_lat = float(filtered_df["latitude"].median()) if not filtered_df.empty else NYC_CENTER["lat"]
    center_lon = float(filtered_df["longitude"].median()) if not filtered_df.empty else NYC_CENTER["lon"]
    
    filtered_df["log_price"] = np.log(filtered_df["price"].astype(float))

    fig = px.scatter_mapbox(
        filtered_df,
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

# Price distribution histogram for filtered comps
st.subheader("Price Distribution (Filtered Listings)")

if filtered_df.empty:
    st.info("No listings match the selected filters.")
else:
    # Create histogram with quantile rules
    hist = alt.Chart(filtered_df).mark_bar(
        color='#4e79a7',
        opacity=0.7
    ).add_selection(
        alt.selection_interval()
    ).encode(
        alt.X('price:Q', 
              bin=alt.Bin(maxbins=40),
              title='Nightly price (USD)',
              scale=alt.Scale(zero=False)),
        alt.Y('count():Q', title='Listings')
    ).properties(
        height=220
    )
    
    # Add vertical rules for quantiles
    rules_data = pd.DataFrame({
        'quantile': ['P10', 'P50', 'P90'],
        'value': [p10, p50, p90],
        'style': ['dashed', 'solid', 'dashed']
    })
    
    rules = alt.Chart(rules_data).mark_rule(
        color='red',
        strokeDash=alt.condition(
            alt.datum.style == 'dashed',
            alt.value([5, 5]),
            alt.value([0, 0])
        )
    ).encode(
        x='value:Q',
        size=alt.value(2)
    )
    
    # Combine histogram and rules
    chart = (hist + rules).resolve_scale(color='independent')
    
    st.altair_chart(chart, use_container_width=True)
    
    # Compute user's predicted percentile within filtered data
    if uncertainty_available:
        user_percentile = (filtered_df['price'] <= p50).mean() * 100
        st.caption(f"Your predicted price (${p50:,.0f}) is at the {user_percentile:.1f}th percentile of similar listings.")
    else:
        user_percentile = (filtered_df['price'] <= pred_price).mean() * 100
        st.caption(f"Your predicted price (${pred_price:,.0f}) is at the {user_percentile:.1f}th percentile of similar listings.")

