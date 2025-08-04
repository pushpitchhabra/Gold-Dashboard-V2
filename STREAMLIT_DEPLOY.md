# Streamlit Cloud Deployment Guide

## Main Application File
**File to run:** `app.py`

## Deployment Steps for Streamlit Cloud

1. **Go to Streamlit Cloud:**
   - Visit: https://share.streamlit.io/
   - Sign in with your GitHub account

2. **Deploy New App:**
   - Click "New app"
   - Repository: `pushpitchhabra/Gold-Dashboard-V2`
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"

3. **App Configuration:**
   - **Repository URL:** https://github.com/pushpitchhabra/Gold-Dashboard-V2.git
   - **Main file:** app.py
   - **Python version:** 3.9+ (auto-detected)
   - **Dependencies:** requirements.txt (included)

## What the App Does
- AI-powered gold trading dashboard
- Real-time predictions using XGBoost
- Technical analysis with multiple indicators
- Macro-economic factors integration
- FRED API integration for economic data

## Files Structure
```
├── app.py                 # 🎯 MAIN FILE - Run this on Streamlit
├── requirements.txt       # Dependencies
├── config.yaml           # Configuration (includes FRED API key)
├── data_loader.py        # Data loading module
├── predictor.py          # AI prediction engine
├── macro_factors.py      # Economic data analysis
├── strategy_analyzer.py  # Trading strategy analysis
└── .streamlit/           # Streamlit configuration
    └── config.toml       # Streamlit settings
```

## Important Notes
- The FRED API key is configured in `config.yaml`
- All dependencies are listed in `requirements.txt`
- The app is optimized for Streamlit Cloud deployment
- No additional configuration needed - just deploy!

## Local Testing
To test locally:
```bash
git clone https://github.com/pushpitchhabra/Gold-Dashboard-V2.git
cd Gold-Dashboard-V2
pip install -r requirements.txt
streamlit run app.py
```
