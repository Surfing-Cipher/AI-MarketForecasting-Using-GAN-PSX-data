import numpy as np
import pandas as pd
import os
import logging
import joblib
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, LeakyReLU, BatchNormalization, Input, Bidirectional, Dropout
from sklearn.preprocessing import MinMaxScaler
from db_manager import fetch_data

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
_log_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
_log_file = os.path.join(_log_dir, 'system.log')

logger = logging.getLogger('nexus_ai')
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    _fh = logging.FileHandler(_log_file, encoding='utf-8')
    _fh.setLevel(logging.INFO)
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _fmt = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    _fh.setFormatter(_fmt)
    _ch.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_ch)

# ==========================================
# 1. GAN ARCHITECTURE (Local Definition)
# ==========================================
def build_gan_generator(latent_dim=100, seq_len=30, num_features=5):
    model = Sequential(name="Generator")
    model.add(Input(shape=(latent_dim,)))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(RepeatVector(seq_len))
    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization(momentum=0.8))
    model.add(TimeDistributed(Dense(num_features, activation='sigmoid')))
    return model

# ==========================================
# 2. LSTM ARCHITECTURE (Local Definition - The Crash Fix)
# ==========================================
def build_lstm_model(lookback=60):
    model = Sequential(name="LSTM_Forecaster")
    model.add(Input(shape=(lookback, 1)))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(25))
    model.add(Dense(1))
    return model

# ==========================================
# 3. CLASS IMPLEMENTATIONS
# ==========================================

class GANGenerator:
    def __init__(self, model_path=None):
        if model_path is None:
            _dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(_dir, '..', 'models', 'saved_models', 'gan_generator_ohlcv.h5'))
        self.latent_dim = 100
        self.seq_len = 30
        self.num_features = 5
        self.model = build_gan_generator(self.latent_dim, self.seq_len, self.num_features)
        
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info("GAN Generator weights loaded successfully.")
            except Exception as e:
                logger.error(f"GAN weights failed to load: {e}")
        else:
            logger.warning(f"GAN model file not found at {model_path}")
        
        # No GAN scaler .pkl exists yet; using dummy calibration.
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(np.array([[100]*5, [300]*5])) 

    def generate_synthetic_data(self):
        try:
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            gen = self.model.predict(noise, verbose=0)
            return self.scaler.inverse_transform(gen[0])[:, 3].tolist()
        except Exception as e:
            logger.error(f"GAN generation failed: {e}")
            return []

class LSTMForecaster:
    def __init__(self, model_path=None):
        _dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.normpath(os.path.join(_dir, '..', 'models', 'saved_models', 'lstm_model.h5'))
        self.lookback = 60
        # 1. Build Local Architecture (Bypasses Version Error)
        self.model = build_lstm_model(self.lookback)
        
        if os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info("LSTM Forecaster weights loaded successfully.")
                self.ready = True
            except Exception as e:
                logger.error(f"LSTM weights failed to load: {e}")
                self.ready = False
        else:
            logger.warning(f"LSTM model file not found at {model_path}")
            self.ready = False
            
        # 2. Load the REAL training scaler from .pkl
        scaler_path = os.path.normpath(os.path.join(_dir, '..', 'models', 'scalers', 'lstm_model_scaler.pkl'))
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"LSTM scaler loaded from {scaler_path}")
            except Exception as e:
                logger.error(f"Failed to load LSTM scaler from {scaler_path}: {e}. Falling back to hardcoded range.")
                self.scaler = MinMaxScaler(feature_range=(0, 1))
                self.scaler.fit(np.array([[60.0], [380.0]]))
        else:
            logger.warning(f"LSTM scaler .pkl not found at {scaler_path}. Using hardcoded fallback range [60, 380].")
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.fit(np.array([[60.0], [380.0]]))

    def predict(self, recent_data):
        if not self.ready or len(recent_data) < self.lookback:
            return 0.0
        try:
            input_slice = np.array(recent_data[-self.lookback:]).reshape(-1, 1)
            scaled = self.scaler.transform(input_slice)
            X_test = scaled.reshape(1, self.lookback, 1)
            pred_scaled = self.model.predict(X_test, verbose=0)
            return float(self.scaler.inverse_transform(pred_scaled)[0][0])
        except Exception as e:
            logger.error(f"LSTM prediction failed: {e}")
            return 0.0

class XGBoostForecaster:
    def __init__(self, model_path=None):
        if model_path is None:
            _dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(_dir, '..', 'models', 'saved_models', 'xgb_model.json'))
        self.model = xgb.XGBRegressor()
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            logger.info("XGBoost model loaded successfully.")
            self.ready = True
        else:
            logger.warning(f"XGBoost model file not found at {model_path}")
            self.ready = False

    def predict(self, features):
        if not self.ready:
            return 0.0
        
        # 5-feature input: RSI, SMA_20, SMA_50, EMA_12, Close
        try:
            arr = np.array([[
                features.get('RSI', 0), 
                features.get('SMA_20', 0), 
                features.get('SMA_50', 0), 
                features.get('EMA_12', 0), 
                features.get('close', 0)
            ]])
            return float(self.model.predict(arr)[0])
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return 0.0
