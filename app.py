from flask import Flask, jsonify, render_template
from data_pipeline import PSXDataPipeline
from models_engine import GANGenerator, LSTMForecaster, XGBoostForecaster
import numpy as np

app = Flask(__name__)

# Initialize Modules
pipeline = PSXDataPipeline(ticker="OGDC") # Use "OGDC" for PSX Website
gan = GANGenerator()
lstm = LSTMForecaster()
xgb = XGBoostForecaster()

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/portfolio')
@app.route('/gan-model')
@app.route('/settings')
def coming_soon():
    return render_template('coming_soon.html')

@app.route('/api/metrics')
def get_metrics():
    try:
        # 1. Get Data from Pipeline
        df = pipeline.get_processed_data()
        latest = df.iloc[-1]
        
        # 2. GAN Generation (Synthetic Data)
        synthetic_moves = gan.generate_synthetic_data(n_samples=30)
        
        # Create synthetic price path for the chart
        current_price = float(latest['Close'])
        synthetic_path = []
        temp_price = current_price
        for move in synthetic_moves:
            # Scale the GAN output to represent % movement
            temp_price = temp_price * (1 + (move * 0.01)) 
            synthetic_path.append(round(temp_price, 2))

        # 3. Forecasts
        lstm_pred = lstm.predict(df['Close'].tail(10).values)
        xgb_pred = xgb.mock_predict(current_price)
        ensemble = (lstm_pred + xgb_pred) / 2

        # 4. JSON Response
        response = {
            "current_price": round(current_price, 2),
            "rsi": round(float(latest['RSI']), 2),
            "sentiment": round(np.random.uniform(-1, 1), 2), # Placeholder
            "predictions": {
                "lstm": round(float(lstm_pred), 2),
                "xgboost": round(float(xgb_pred), 2),
                "ensemble": round(float(ensemble), 2)
            },
            "chart_data": {
                "dates": df['Date'].dt.strftime('%Y-%m-%d').tolist(),
                "close": df['Close'].tolist(),
                "bb_upper": df['BB_High'].tolist(),
                "bb_lower": df['BB_Low'].tolist()
            },
            "gan_projection": synthetic_path
        }
        return jsonify(response)

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting FYP-1 Dashboard...")
    app.run(debug=True, port=5000)