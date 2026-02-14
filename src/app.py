from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from data_pipeline import PSXDataPipeline
from models_engine import GANGenerator, LSTMForecaster, XGBoostForecaster
from db_manager import (
    init_db, create_user, verify_user,
    add_to_watchlist, remove_from_watchlist, get_watchlist
)
from sentiment_engine import get_live_sentiment
import numpy as np
import pandas as pd
import json
import os
import logging
import secrets
import ta

# Reuse the shared application logger
logger = logging.getLogger('nexus_ai')

# Set template folder to parent directory's templates
template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
app = Flask(__name__, template_folder=template_dir)

# Persistent secret key — survives server restarts so sessions are preserved
_secret_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '.flask_secret'))
if os.environ.get('FLASK_SECRET_KEY'):
    app.secret_key = os.environ['FLASK_SECRET_KEY']
elif os.path.exists(_secret_path):
    with open(_secret_path, 'r') as f:
        app.secret_key = f.read().strip()
else:
    _generated = secrets.token_hex(32)
    with open(_secret_path, 'w') as f:
        f.write(_generated)
    app.secret_key = _generated
    logger.info(f'Generated new Flask secret key → .flask_secret')

# --- Initialize Database & Modules ---
init_db()
pipeline = PSXDataPipeline(ticker="OGDC") 
gan = GANGenerator()
lstm = LSTMForecaster()
xgb_model = XGBoostForecaster()


# ==========================================
# AUTH HELPER
# ==========================================
def login_required(f):
    """Decorator to protect routes that require authentication."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    """Decorator to protect routes that require admin privileges."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        if not session.get('is_admin'):
            return jsonify({"error": "Admin privileges required"}), 403
        return f(*args, **kwargs)
    return decorated


# ==========================================
# PAGE ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@app.route('/gan-model')
def gan_page():
    return render_template('gan_model.html')

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

@app.route('/admin/logs')
@admin_required
def admin_logs_page():
    return render_template('admin_logs.html')


# ==========================================
# AUTH API ROUTES
# ==========================================
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')

    if not username or not email or not password:
        return jsonify({"error": "username, email, and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    user = create_user(username, email, password)
    if user is None:
        return jsonify({"error": "Username or email already exists"}), 409

    # Auto-login after registration
    session['user_id'] = user['id']
    session['username'] = user['username']
    return jsonify({"message": "Registration successful", "user": user}), 201


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400

    user = verify_user(username, password)
    if user is None:
        return jsonify({"error": "Invalid credentials"}), 401

    session['user_id'] = user['id']
    session['username'] = user['username']
    session['is_admin'] = user.get('is_admin', False)
    return jsonify({"message": "Login successful", "user": user}), 200


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"}), 200


@app.route('/api/session')
def get_session():
    if 'user_id' in session:
        return jsonify({
            "authenticated": True,
            "user": {"id": session['user_id'], "username": session['username']}
        }), 200
    return jsonify({"authenticated": False}), 401


# ==========================================
# WATCHLIST API ROUTES
# ==========================================
@app.route('/api/watchlist', methods=['GET'])
@login_required
def api_get_watchlist():
    tickers = get_watchlist(session['user_id'])
    return jsonify({"watchlist": tickers}), 200


@app.route('/api/watchlist', methods=['POST'])
@login_required
def api_add_watchlist():
    data = request.get_json()
    if not data or not data.get('ticker_symbol'):
        return jsonify({"error": "ticker_symbol is required"}), 400

    ticker = data['ticker_symbol'].strip().upper()
    added = add_to_watchlist(session['user_id'], ticker)
    if added:
        return jsonify({"message": f"{ticker} added to watchlist"}), 201
    return jsonify({"message": f"{ticker} already in watchlist"}), 200


@app.route('/api/watchlist/<ticker>', methods=['DELETE'])
@login_required
def api_remove_watchlist(ticker):
    removed = remove_from_watchlist(session['user_id'], ticker.upper())
    if removed:
        return jsonify({"message": f"{ticker.upper()} removed from watchlist"}), 200
    return jsonify({"error": f"{ticker.upper()} not found in watchlist"}), 404


# ==========================================
# DATA API ROUTES
# ==========================================
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
        
        # 2. GAN Confidence Interval (Monte Carlo)
        gan_ci = gan.generate_confidence_interval(n_simulations=50)

        # 3. Forecasts (LSTM + XGBoost)
        recent_60_days = df['Close'].tail(60).values.tolist()
        lstm_pred = lstm.predict(recent_60_days)
        
        xgb_features = {
            'RSI': float(latest['RSI']),
            'SMA_20': float(ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator().iloc[-1]),
            'SMA_50': float(ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator().iloc[-1]),
            'EMA_12': float(ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator().iloc[-1]),
            'close': current_price
        }
        xgb_pred = xgb_model.predict(xgb_features)
        
        # 4. XGBoost Gating — exclude XGB if its prediction deviates > 50% from LSTM
        xgb_gated = False
        if lstm_pred > 0 and xgb_pred > 0:
            deviation = abs(xgb_pred - lstm_pred) / lstm_pred
            if deviation < 0.5:
                # Both models are in reasonable agreement
                ensemble = (lstm_pred * 0.63) + (xgb_pred * 0.37)
            else:
                # XGBoost is miscalibrated — use LSTM only
                ensemble = lstm_pred
                xgb_gated = True
                logger.warning(
                    f"XGBoost GATED: deviation={deviation:.1%} "
                    f"(LSTM={lstm_pred:.2f}, XGB={xgb_pred:.2f})"
                )
        else:
            ensemble = lstm_pred if lstm_pred > 0 else xgb_pred

        # 5. Sharpe Ratio (Financial Risk IQ)
        returns = df['Close'].pct_change().dropna().tail(90)
        daily_rf = 0.05 / 252  # 5% annual risk-free rate → daily
        excess = returns - daily_rf
        sharpe = float(excess.mean() / excess.std()) * (252 ** 0.5) if len(returns) > 1 else 0.0
        volatility = float(returns.std() * (252 ** 0.5) * 100) if len(returns) > 1 else 0.0

        # 6. SHAP Explainability
        shap_result = xgb_model.explain(xgb_features)

        # 7. NLP Sentiment
        sentiment_data = get_live_sentiment()

        # 8. JSON Response
        response = {
            "current_price": round(current_price, 2),
            "rsi": round(float(latest['RSI']), 2),
            "sharpe_ratio": round(sharpe, 2),
            "volatility": round(volatility, 2),
            "sentiment": sentiment_data["score"],
            "sentiment_label": sentiment_data["label"],
            "headline_count": sentiment_data["headline_count"],
            "predictions": {
                "lstm": round(float(lstm_pred), 2),
                "xgboost": round(float(xgb_pred), 2),
                "ensemble": round(float(ensemble), 2),
                "xgb_gated": xgb_gated
            },
            "shap": shap_result,
            "gan_confidence": {
                "mean": gan_ci["mean"],
                "std": gan_ci["std"],
                "ci_lower": gan_ci["ci_lower"],
                "ci_upper": gan_ci["ci_upper"],
            },
            "chart_data": {
                "dates": df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                "close": df['Close'].tolist(),
                "bb_upper": df['BB_High'].tolist(),
                "bb_lower": df['BB_Low'].tolist()
            },
            "gan_projection": gan_ci["path"]
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

# ==========================================
# ADMIN API ROUTES
# ==========================================
@app.route('/api/admin/logs')
@admin_required
def api_admin_logs():
    """Tail the last 100 lines of system.log."""
    from collections import deque
    log_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'system.log'))
    try:
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = list(deque(f, 100))
            return jsonify({"lines": [l.rstrip() for l in lines]})
        return jsonify({"lines": ["system.log not found."]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/admin/retrain', methods=['POST'])
@admin_required
def trigger_retrain():
    """Runs GAN-LSTM augmented training asynchronously."""
    import threading
    import subprocess
    import sys

    script_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), '..', 'GAN_MODEL', 'gan_lstm_augmented_training.py')
    )
    if not os.path.exists(script_path):
        return jsonify({"error": "Training script not found"}), 404

    def _run_training():
        logger.info("ADMIN: GAN-LSTM Retrain triggered.")
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=3600
            )
            if result.returncode == 0:
                logger.info("ADMIN: Retrain completed successfully.")
            else:
                logger.error(f"ADMIN: Retrain failed: {result.stderr[-500:]}")
        except Exception as e:
            logger.error(f"ADMIN: Retrain exception: {e}")

    t = threading.Thread(target=_run_training, daemon=True, name='retrain-worker')
    t.start()
    return jsonify({"status": "Retraining started in background", "thread": t.name})


if __name__ == '__main__':
    logger.info("Starting Nexus AI Dashboard on port 5000...")
    app.run(debug=True, port=5000)