from data_pipeline import PSXDataPipeline
from models_engine import GANGenerator, LSTMForecaster, XGBoostForecaster
import traceback
import numpy as np

print("1. Initializing Modules...")
try:
    pipeline = PSXDataPipeline(ticker="OGDC")
    print("Pipeline Initialized.")
    
    gan = GANGenerator()
    print("GAN Initialized.")
    
    lstm = LSTMForecaster()
    print("LSTM Initialized.")
    
    xgb = XGBoostForecaster()
    print("XGBoost Initialized.")
except Exception as e:
    print(f"Initialization Error: {e}")
    traceback.print_exc()
    exit()

print("\n2. Simulating /api/metrics Request...")
try:
    # 1. Get Data from Pipeline
    print("Fetching processed data...")
    df = pipeline.get_processed_data()
    print(f"Data fetched. Shape: {df.shape}")
    latest = df.iloc[-1]
    print(f"Latest Date: {latest['Date']}")
    
    # 2. GAN Generation
    print("Generating GAN projection...")
    gan_projection = gan.generate_synthetic_data()
    print(f"GAN Projection Generated. Length: {len(gan_projection)}")

    # 3. Forecasts
    print("Generating Forecasts...")
    current_price = float(latest['Close'])
    lstm_pred = lstm.predict(df['Close'].tail(10).values)
    xgb_pred = xgb.mock_predict(current_price)
    
    ensemble = (lstm_pred + xgb_pred) / 2
    
    response = {
        "current_price": round(current_price, 2),
        "gan_projection": gan_projection,
        "ensemble": round(float(ensemble), 2)
    }
    print("\n✅ Simulation Successful!")
    print("Sample Response Data:", response)

except Exception as e:
    print(f"\n❌ Runtime Error during simulation: {e}")
    traceback.print_exc()
