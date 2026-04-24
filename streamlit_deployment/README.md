# GENOME Visibility Predictor - Streamlit App

## 📋 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Access the App
Open your browser to: http://localhost:8501

## 📁 File Structure
```
streamlit_deployment/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── models/
│   ├── ensemble_classifier.pkl    # Stage 1 model
│   ├── stage2_regressor.pkl       # Stage 2 model
│   ├── features_config.pkl        # Feature lists
│   ├── metadata.pkl               # Model performance metrics
│   └── *.json                     # Human-readable configs
└── data/
    ├── feature_importance.csv     # Feature rankings
    ├── feature_standards.csv      # Visibility benchmarks
    └── *.pkl                      # Pickled data files
```

## 🎯 Model Performance

### Stage 1: Visibility Classification
- ROC AUC: 0.9987
- Accuracy: 97.9%
- F1 Score: 0.96

### Stage 2: PAWC Prediction
- R²: 0.27
- Spearman: 0.33
- RMSE: 16.82

## 🚀 Features

1. **Interactive Prediction**: Input source features and get instant visibility predictions
2. **Feature Analysis**: Explore which features drive visibility
3. **Model Insights**: View detailed model architecture and performance
4. **Feature Standards**: Compare your sources against visibility benchmarks

## 📊 Usage Example

```python
# In the app, adjust sliders for:
- Relevance: 0.85
- Influence: 0.75
- Uniqueness: 0.80
- Click Probability: 0.70
- etc.

# Get results:
✅ Visibility Status: VISIBLE
📊 Visibility Probability: 87.5%
🎯 Predicted PAWC Score: 45.2
```

## 🔧 Deployment

### Deploy to Streamlit Cloud
1. Push this folder to GitHub
2. Go to https://share.streamlit.io
3. Connect your repository
4. Deploy!

### Deploy to Heroku
```bash
heroku create your-app-name
git push heroku main
```

## 📝 License
MIT License - feel free to use for any purpose!

## 👤 Author
Your Name - GENOME Visibility Prediction Model
