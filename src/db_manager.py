from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
import pandas as pd
from datetime import datetime

# Define Base
Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    
    # Unique constraint to prevent duplicate entries for same ticker + date
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='uix_ticker_date'),
    )

# Database Setup
# Use absolute path based on this file's location
import os as _os
_current_dir = _os.path.dirname(_os.path.abspath(__file__))
DB_NAME = _os.path.join(_current_dir, '..', 'data', 'database', 'nexus.db')
DB_NAME = _os.path.normpath(DB_NAME)
engine = create_engine(f"sqlite:///{DB_NAME}")
Session = sessionmaker(bind=engine)

def init_db():
    """Creates the tables in the database."""
    Base.metadata.create_all(engine)
    print(f"[INFO] Database '{DB_NAME}' initialized successfully.")

def save_to_db(df, ticker):
    """
    Upserts market data from a DataFrame into the database.
    df: Pandas DataFrame with columns [Date, Open, High, Low, Close, Volume]
    ticker: Stock ticker symbol (e.g., 'OGDC')
    """
    if df.empty:
        print("[WARN] DataFrame is empty. Nothing to save.")
        return

    session = Session()
    try:
        count = 0
        for _, row in df.iterrows():
            # Check if record exists
            exists = session.query(MarketData).filter_by(
                ticker=ticker, 
                date=row['Date']
            ).first()
            
            if not exists:
                record = MarketData(
                    ticker=ticker,
                    date=row['Date'],
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=row['Volume']
                )
                session.add(record)
                count += 1
            else:
                # Update existing record if needed
                exists.open = row['Open']
                exists.high = row['High']
                exists.low = row['Low']
                exists.close = row['Close']
                exists.volume = row['Volume']
                # count += 1 # Uncomment if you want to count updates too originall requirement implies upsert or ignore, usually we count inserts
        
        session.commit()
        print(f"[SUCCESS] Saved/Updated {count} new records for {ticker}.")
    except Exception as e:
        session.rollback()
        print(f"[ERROR] Failed to save data: {e}")
    finally:
        session.close()

def fetch_data(ticker):
    """
    Retrieves data for a specific ticker as a Pandas DataFrame.
    """
    session = Session()
    try:
        query = session.query(MarketData).filter_by(ticker=ticker).order_by(MarketData.date.asc())
        df = pd.read_sql(query.statement, session.bind)
        
        # Normalize column names to match data_pipeline.py (capital case)
        if not df.empty:
            df.rename(columns={
                'date': 'Date',
                'open': 'Open', 
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
        
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        return pd.DataFrame() # Return empty DF on error
    finally:
        session.close()

if __name__ == "__main__":
    init_db()
