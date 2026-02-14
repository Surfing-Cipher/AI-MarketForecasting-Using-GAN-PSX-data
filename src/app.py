from flask import Flask, jsonify, render_template
from data_pipeline import PSXDataPipeline
from models_engine import GANGenerator, LSTMForecaster, XGBoostForecaster
import numpy as np
import pandas as pd
import json
import os
import logging
import ta

# Reuse the shared application logger
logger = logging.getLogger('nexus_ai')

# Set template folder to parent directory's templates
template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
app = Flask(__name__, template_folder=template_dir)

# --- Initialize Modules ---
# We keep your existing Pipeline
pipeline = PSXDataPipeline(ticker="OGDC") 
gan = GANGenerator()
lstm = LSTMForecaster()
xgb = XGBoostForecaster()

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/portfolio')
def portfolio():
    # Placeholder
    return render_template('coming_soon.html')

@app.route('/gan-model')
def gan_page():
    return render_template('gan_model.html')

@app.route('/settings')
def coming_soon():
    return render_template('coming_soon.html')

@app.route('/api/metrics')
def get_metrics():
    try:
        # 1. Get Data from Your Pipeline
        df = pipeline.get_processed_data()
        
        # Safety Check: If pipeline fails or DB is empty
        if df is None or df.empty:
            return jsonify({"error": "No data available from Pipeline"}), 500

        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        
        # 2. GAN Generation (Real Synthetic Scenarios)
        gan_projection = gan.generate_synthetic_data()

        # 3. Forecasts (LSTM + XGBoost)
        # LSTM: Convert to list for compatibility
        recent_60_days = df['Close'].tail(60).values.tolist()
        lstm_pred = lstm.predict(recent_60_days)
        
        # XGBoost: Compute technical indicators and predict
        xgb_features = {
            'RSI': float(latest['RSI']),
            'SMA_20': float(ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator().iloc[-1]),
            'SMA_50': float(ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator().iloc[-1]),
            'EMA_12': float(ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator().iloc[-1]),
            'close': current_price  # <--- CRITICAL ADDITION
        }
        xgb_pred = xgb.predict(xgb_features)  # Returns 0.0 if model not ready
        
        # Weighted Ensemble (based on backtest accuracy: LSTM 68.4%, XGB 40.6%)
        W_LSTM = 0.63
        W_XGB  = 0.37
        ensemble = (lstm_pred * W_LSTM) + (xgb_pred * W_XGB)

        # 4. JSON Response
        response = {
            "current_price": round(current_price, 2),
            "rsi": round(float(latest['RSI']), 2),
            "sentiment": 0.3,  # TODO: Replace with live NLP pipeline (FYP-2)
            "predictions": {
                "lstm": round(float(lstm_pred), 2),      # <--- REAL LSTM VALUE
                "xgboost": round(float(xgb_pred), 2),
                "ensemble": round(float(ensemble), 2)
            },
            "chart_data": {
                # Ensure Date is formatted for Chart.js
                "dates": df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                "close": df['Close'].tolist(),
                "bb_upper": df['BB_High'].tolist(),
                "bb_lower": df['BB_Low'].tolist()
            },
            "gan_projection": gan_projection
        }
        return jsonify(response)

    except Exception as e:
        logger.error(f"API /api/metrics error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/backtest')
def get_backtest():
    """Serve backtesting results from model_performance.json"""
    try:
        backtest_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'model_performance.json')
        backtest_path = os.path.normpath(backtest_path)
        
        if os.path.exists(backtest_path):
            with open(backtest_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({"error": "Backtesting results not found. Run backtest.py first."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting FYP-1 Dashboard...")
    app.run(debug=True, port=5000)