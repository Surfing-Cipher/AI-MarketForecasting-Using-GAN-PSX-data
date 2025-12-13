# AI Market Forecasting (PSX) - FYP 1

![Project Output](/images/fyp1output.png)

## Overview
This project is an **AI-powered Stock Market Forecasting Dashboard** specifically tailored for the Pakistan Stock Exchange (PSX), focusing on Oil & Gas Development Company (OGDC). It leverages a hybrid approach combining **Topological Data Analysis (TDA), GANs, LSTM, and XGBoost** to predict stock trends.

The interface is designed with a **"Future Dark" aesthetic** inspired by high-end financial terminals (Bloomberg, Refinitiv), featuring a glassmorphism UI, real-time data integration, and neon accents.

## Key Features

### 🚀 Advanced AI Models
- **GAN (Generative Adversarial Network):** Generates synthetic price paths to model potential market volatility.
- **LSTM (Long Short-Term Memory):** Deep learning model for time-series forecasting.
- **XGBoost:** Gradient boosting regression for mock prediction validation.
- **Ensemble Learning:** Combines LSTM and XGBoost outputs for a robust consensus forecast.

### 🎨 Stunning UI/UX
- **Theme:** Deep Midnight Blue (`slate-950`) with Neon Cyan/Green/Purple accents.
- **Technology:** Built with **Flask**, **Tailwind CSS**, and **Chart.js**.
- **Glassmorphism:** Frosted glass effect for cards and sidebars.
- **Interactive Charts:** Real-time plotting of Close Price vs. Bollinger Bands.

### 📊 Real-Time Data Pipeline
- **Live Scraper:** Fetches real-time stock data from the PSX Data Portal.
- **Technical Indicators:** Automatically calculates RSI (14) and Bollinger Bands.
- **Robustness:** Includes fallback mechanisms (Yahoo Finance) and error handling for scraping failures.

## Project Structure
```
📂 GAN PSX Marketforecasting
├── 📄 app.py                # Main Flask Application
├── 📄 data_pipeline.py      # Scraper & Feature Engineering Logic
├── 📄 models_engine.py      # AI Models (GAN, LSTM, XGBoost) definitions
├── 📄 requirements.txt      # Python Dependencies
├── 📂 templates
│   ├── 📄 dashboard.html    # Main UI (Future Dark Theme)
│   └── 📄 coming_soon.html  # Generic Template for future modules
└── 📂 image
    └── 📄 fyp1output.png    # Screenshot of the dashboard
```

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Surfing-Cipher/AI-MarketForecasting-Using-GAN-PSX-data.git
   cd AI-MarketForecasting-Using-GAN-PSX-data
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```
   The dashboard will live at `http://127.0.0.1:5000`.

## Future Roadmap (FYP-2)
- [ ] Connect Portfolio Management module.
- [ ] Train GAN model on full 10-year dataset.
- [ ] Deploy to cloud (AWS/Heroku).
- [ ] Add User Authentication.

## License
MIT License - Copyright (c) 2025 Surfing-Cipher
