import pandas as pd
import numpy as np
import json
import xgboost as xgb
import os
from db_manager import fetch_data
from models_engine import LSTMForecaster

def run_backtest():
    print("🚀 Starting Backtest...")
    
    # 1. Load Data
    df = fetch_data("OGDC")
    if df is None or df.empty:
        print("❌ Database empty."); return

    # Standardize columns
    df.columns = df.columns.str.strip().str.lower()
    
    # 2. Indicators
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df = df.dropna()

    # 3. Load Models
    lstm_engine = LSTMForecaster() 
    xgb_model = xgb.XGBRegressor()
    
    # Verify Model File - use absolute path
    _dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.normpath(os.path.join(_dir, '..', 'models', 'saved_models', 'xgb_model.json'))
    if not os.path.exists(model_path):
        print(f"❌ ERROR: {model_path} not found!")
        return
    xgb_model.load_model(model_path)

    # 4. Test (Last 90 Days)
    test_slice = df.tail(90)
    actuals = test_slice['close'].tolist()
    dates = test_slice['date'].dt.strftime('%Y-%m-%d').tolist()
    
    l_preds, x_preds = [], []

    print(f"⏳ Testing on {len(test_slice)} days...")
    
    for i in range(len(test_slice)):
        # LSTM
        idx = test_slice.index[i]
        hist = df.loc[:idx-1].tail(60)['close'].values.tolist()
        l_preds.append(lstm_engine.predict(hist))
        
        # XGBoost (5 Features)
        row = test_slice.iloc[i]
        # RSI, SMA20, SMA50, EMA12, Close
        feats = np.array([[row['RSI'], row['SMA_20'], row['SMA_50'], row['EMA_12'], row['close']]])
        x_preds.append(float(xgb_model.predict(feats)[0]))

    # 5. Calculate Ensemble (Average of LSTM + XGBoost)
    ensemble_preds = [(l + x) / 2 for l, x in zip(l_preds, x_preds)]
    
    # 6. Metrics
    l_mape = np.mean(np.abs((np.array(actuals) - np.array(l_preds)) / np.array(actuals))) * 100
    x_mape = np.mean(np.abs((np.array(actuals) - np.array(x_preds)) / np.array(actuals))) * 100
    e_mape = np.mean(np.abs((np.array(actuals) - np.array(ensemble_preds)) / np.array(actuals))) * 100
    
    results = {
        "lstm": {
            "predictions": l_preds, 
            "accuracy": round(100 - l_mape, 1),
            "actual": actuals,
            "dates": dates
        },
        "xgboost": {
            "predictions": x_preds, 
            "accuracy": round(100 - x_mape, 1),
            "actual": actuals,
            "dates": dates
        },
        "ensemble": {
            "predictions": ensemble_preds,
            "accuracy": round(100 - e_mape, 1),
            "actual": actuals,
            "dates": dates
        }
    }
    
    # 7. Save JSON
    save_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'model_performance.json')
    save_path = os.path.normpath(save_path)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ SUCCESS!")
    print(f"📊 LSTM Accuracy:     {results['lstm']['accuracy']}%")
    print(f"📊 XGBoost Accuracy:  {results['xgboost']['accuracy']}%")
    print(f"📊 Ensemble Accuracy: {results['ensemble']['accuracy']}%")
    print(f"📂 Data Saved to:     {save_path}")

if __name__ == "__main__":
    run_backtest()