# NYC Airbnb Price Forecaster

A machine learning application that predicts Airbnb nightly prices in New York City using ensemble methods and uncertainty quantification. Built with Python and Streamlit, this application demonstrates advanced data science techniques including feature engineering, model validation, and real-time prediction with confidence intervals.

## Overview

This project showcases a complete end-to-end machine learning pipeline for real estate price prediction, featuring:

- **Ensemble Learning**: RandomForest regressor with uncertainty quantification
- **Feature Engineering**: Automated categorical encoding and numerical bucketing
- **Model Validation**: Comprehensive evaluation with MAE and SMAPE metrics
- **Interactive Analytics**: Real-time price forecasting with confidence intervals
- **Market Analysis**: Comparative analysis against filtered comparable listings
- **Partial Dependence Analysis**: Feature impact visualization for model interpretability

## Technical Architecture

### Data Pipeline
- **Source**: Inside Airbnb NYC dataset (cleaned and standardized)
- **Processing**: Automated data validation, feature engineering, and outlier removal
- **Storage**: Parquet format for optimal performance with fallback to CSV

### Machine Learning Pipeline
- **Algorithm**: RandomForest Regressor with 300 estimators
- **Features**: One-hot encoded categorical variables, bucketed numerical features, and geographic coordinates
- **Target**: Log-transformed price with inverse transformation for interpretability
- **Validation**: 80/20 train-test split with cross-validation

### Uncertainty Quantification
- **Method**: Per-tree prediction aggregation for P10, P50, P90 quantiles
- **Implementation**: Custom quantile computation with caching for performance
- **Interpretation**: Confidence intervals for risk assessment and decision-making

## Key Features

### Predictive Analytics
- Real-time price prediction with uncertainty bounds
- Feature importance analysis and model interpretability
- Partial dependence plots for feature impact assessment

### Market Intelligence
- Comparative analysis against filtered market segments
- Percentile ranking within comparable listings
- Geographic price distribution visualization

### Interactive Dashboard
- Dynamic filtering by neighborhood, room type, and requirements
- Real-time model updates with parameter changes
- Comprehensive visualization suite (histograms, ECDFs, scatter plots)

## Technical Implementation

### Core Technologies
- **Backend**: Python 3.8+, scikit-learn, pandas, numpy
- **Frontend**: Streamlit for interactive web interface
- **Visualization**: Plotly for maps, Altair for statistical charts
- **Deployment**: Streamlit Cloud with automated model training fallback

### Performance Optimizations
- Cached model loading and prediction computation
- Efficient data structures and memory management
- Responsive UI with real-time updates

## Project Structure

```
├── app.py                      # Main Streamlit application
├── src/
│   └── model.py               # ML pipeline and utilities
├── notebooks/
│   ├── 01_data_explore.ipynb  # Data exploration and cleaning
│   ├── 02_model_dev.ipynb     # Model training and evaluation
│   └── data/                  # Processed datasets
├── models/                    # Trained model artifacts
└── requirements.txt           # Python dependencies
```

## Model Performance

The RandomForest model achieves competitive performance on the NYC Airbnb dataset:
- **Mean Absolute Error**: Optimized for business interpretability
- **Symmetric Mean Absolute Percentage Error**: Robust to outliers
- **Feature Importance**: Geographically and categorically meaningful rankings

## Deployment

### Local Development
```bash
git clone https://github.com/benmayr/airbnbnightly.git
cd airbnbnightly
pip install -r requirements.txt
streamlit run app.py
```

### Production Deployment
- Automated model training on first deployment
- Fallback mechanisms for missing data or models
- Environment variable configuration for external resources

## Business Applications

This application demonstrates several key competencies relevant to financial services:

- **Risk Assessment**: Uncertainty quantification for price prediction confidence
- **Market Analysis**: Comparative benchmarking and percentile ranking
- **Feature Engineering**: Automated data preprocessing and transformation
- **Model Validation**: Comprehensive evaluation and performance monitoring
- **Interactive Analytics**: Real-time decision support tools

## Technical Competencies Demonstrated

- **Machine Learning**: Ensemble methods, feature engineering, model validation
- **Software Engineering**: Clean code architecture, error handling, performance optimization
- **Data Science**: Statistical analysis, visualization, uncertainty quantification
- **Full-Stack Development**: Backend ML pipeline with frontend web interface
- **DevOps**: Automated deployment, caching, and fallback mechanisms

## Future Enhancements

- Multi-city expansion with transfer learning
- Real-time data integration and model retraining
- Advanced ensemble methods (XGBoost, neural networks)
- A/B testing framework for model comparison
- API development for programmatic access

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Inside Airbnb for providing the comprehensive NYC dataset
- The open-source Python ecosystem for ML and visualization tools