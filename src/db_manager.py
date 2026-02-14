from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import os
import logging
from datetime import datetime

logger = logging.getLogger('nexus_ai')

# Define Base
Base = declarative_base()

# ==========================================
# ORM MODELS
# ==========================================

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
    
    __table_args__ = (
        UniqueConstraint('ticker', 'date', name='uix_ticker_date'),
    )


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    watchlist = relationship('Watchlist', back_populates='user', cascade='all, delete-orphan')

    def to_dict(self):
        return {"id": self.id, "username": self.username, "email": self.email}


class Watchlist(Base):
    __tablename__ = 'watchlist'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    ticker_symbol = Column(String(20), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)

    user = relationship('User', back_populates='watchlist')

    __table_args__ = (
        UniqueConstraint('user_id', 'ticker_symbol', name='uix_user_ticker'),
    )


# ==========================================
# DATABASE SETUP
# ==========================================

_current_dir = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.normpath(os.path.join(_current_dir, '..', 'data', 'database', 'nexus.db'))
engine = create_engine(f"sqlite:///{DB_NAME}")
Session = sessionmaker(bind=engine)


def init_db():
    """Creates all tables in the database."""
    Base.metadata.create_all(engine)
    logger.info(f"Database '{DB_NAME}' initialized successfully (tables: market_data, users, watchlist).")


# ==========================================
# MARKET DATA FUNCTIONS (existing)
# ==========================================

def save_to_db(df, ticker):
    """Upserts market data from a DataFrame into the database."""
    if df.empty:
        logger.warning("DataFrame is empty. Nothing to save.")
        return

    session = Session()
    try:
        count = 0
        for _, row in df.iterrows():
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
                exists.open = row['Open']
                exists.high = row['High']
                exists.low = row['Low']
                exists.close = row['Close']
                exists.volume = row['Volume']
        
        session.commit()
        logger.info(f"Saved/Updated {count} new records for {ticker}.")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to save data: {e}")
    finally:
        session.close()


def fetch_data(ticker):
    """Retrieves data for a specific ticker as a Pandas DataFrame."""
    session = Session()
    try:
        query = session.query(MarketData).filter_by(ticker=ticker).order_by(MarketData.date.asc())
        df = pd.read_sql(query.statement, session.bind)
        
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
        logger.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()
    finally:
        session.close()


# ==========================================
# USER AUTH FUNCTIONS (FYP-2)
# ==========================================

def create_user(username, email, password):
    """Creates a new user with hashed password. Returns user dict or None on failure."""
    session = Session()
    try:
        existing = session.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        if existing:
            logger.warning(f"User registration failed: '{username}' or '{email}' already exists.")
            return None

        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        session.add(user)
        session.commit()
        logger.info(f"User '{username}' registered successfully.")
        return user.to_dict()
    except Exception as e:
        session.rollback()
        logger.error(f"User creation error: {e}")
        return None
    finally:
        session.close()


def verify_user(username, password):
    """Verifies credentials. Returns user dict or None."""
    session = Session()
    try:
        user = session.query(User).filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            logger.info(f"User '{username}' authenticated successfully.")
            return user.to_dict()
        logger.warning(f"Failed login attempt for username: '{username}'.")
        return None
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None
    finally:
        session.close()


# ==========================================
# WATCHLIST FUNCTIONS (FYP-2)
# ==========================================

def add_to_watchlist(user_id, ticker_symbol):
    """Adds a ticker to the user's watchlist. Returns True on success."""
    session = Session()
    try:
        exists = session.query(Watchlist).filter_by(
            user_id=user_id, ticker_symbol=ticker_symbol.upper()
        ).first()
        if exists:
            logger.info(f"Ticker '{ticker_symbol}' already in watchlist for user {user_id}.")
            return False

        entry = Watchlist(user_id=user_id, ticker_symbol=ticker_symbol.upper())
        session.add(entry)
        session.commit()
        logger.info(f"Added '{ticker_symbol}' to watchlist for user {user_id}.")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Watchlist add error: {e}")
        return False
    finally:
        session.close()


def remove_from_watchlist(user_id, ticker_symbol):
    """Removes a ticker from the user's watchlist. Returns True on success."""
    session = Session()
    try:
        entry = session.query(Watchlist).filter_by(
            user_id=user_id, ticker_symbol=ticker_symbol.upper()
        ).first()
        if entry:
            session.delete(entry)
            session.commit()
            logger.info(f"Removed '{ticker_symbol}' from watchlist for user {user_id}.")
            return True
        return False
    except Exception as e:
        session.rollback()
        logger.error(f"Watchlist remove error: {e}")
        return False
    finally:
        session.close()


def get_watchlist(user_id):
    """Returns list of ticker symbols in user's watchlist."""
    session = Session()
    try:
        entries = session.query(Watchlist).filter_by(user_id=user_id).order_by(Watchlist.added_at.desc()).all()
        return [e.ticker_symbol for e in entries]
    except Exception as e:
        logger.error(f"Watchlist fetch error: {e}")
        return []
    finally:
        session.close()


if __name__ == "__main__":
    init_db()
