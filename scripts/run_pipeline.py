from data_pipeline import PSXDataPipeline
from db_manager import init_db, save_to_db, fetch_data

def main():
    # 1. Initialize Database
    print("--- Step 1: Initialize Database ---")
    init_db()
    
    # 2. Initialize Pipeline
    TICKER = "OGDC"
    print(f"\n--- Step 2: Running Pipeline for {TICKER} ---")
    pipeline = PSXDataPipeline(ticker=TICKER)
    
    # 3. Fetch Historical Data
    # Note: fetch_historical_data now returns OHLCV due to our recent update
    print("Fetching historical data (this may take a moment)...")
    df = pipeline.fetch_historical_data(months=60)
    
    if not df.empty:
        print(f"Fetched {len(df)} rows from source.")
        
        # 4. Save to Database
        print("\n--- Step 3: Saving to Database ---")
        save_to_db(df, TICKER)
        
        # 5. Verify Data
        print("\n--- Step 4: Verification ---")
        saved_df = fetch_data(TICKER)
        print(f"Total rows in DB for {TICKER}: {len(saved_df)}")
        print("Last 5 rows:")
        print(saved_df.tail())
    else:
        print("[ERROR] No data fetched from pipeline.")

if __name__ == "__main__":
    main()
