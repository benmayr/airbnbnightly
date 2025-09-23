# NYC Airbnb Price Forecaster

A Streamlit app that loads a cleaned Inside Airbnb dataset and a trained RandomForest model to forecast nightly prices and visualize listings on a map.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data and Model
- Clean dataset (preferred): `notebooks/data/clean/listings.parquet` (or `listings.csv`)
- Model: `models/rf_price.pkl`

You can generate the model via `notebooks/02_model_dev.ipynb` or:
```bash
python -m src.model --data notebooks/data/clean/listings.csv --model_out models/rf_price.pkl
```

## Deploy to Streamlit Cloud
1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app pointing to this repo.
3. Set the entry point to `app.py`.
4. Ensure `requirements.txt` exists (provided).
5. Provide data/model files in the repo:
   - Recommended: commit `notebooks/data/clean/listings.parquet` (or CSV) and `models/rf_price.pkl`.
   - For large files, use Git LFS or host externally and modify `DATA_CANDIDATES`/`MODEL_PATH` in `app.py` to download on startup.

### Environment variables (optional)
If hosting data/model on a private URL, add environment variables in Streamlit Cloud (Settings → Secrets) and fetch in `app.py`.

### Map tiles
The app uses OpenStreetMap tiles via Plotly's `open-street-map` style, which requires no API keys.
