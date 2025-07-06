"""
Configuration management for the Indian Stock Market Predictor.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for managing API keys and settings."""
    
    # API Configuration
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'Q0DIW81MMLWGUY8K')
    
    # Database Configuration
    DATABASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'stocks.db')
    
    # Data Storage Configuration
    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Model Configuration
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Default Stock Symbols (NIFTY 50 stocks)
    DEFAULT_STOCKS = [
        'RELIANCE.BSE',
        'TCS.BSE',
        'HDFCBANK.BSE',
        'INFY.BSE',
        'HINDUNILVR.BSE',
        'ITC.BSE',
        'SBIN.BSE',
        'BHARTIARTL.BSE',
        'KOTAKBANK.BSE',
        'LT.BSE'
    ]
    
    # Technical Analysis Parameters
    SMA_PERIODS = [50, 200]
    EMA_PERIODS = [12, 26]
    RSI_PERIOD = 14
    MACD_PERIODS = {'fast': 12, 'slow': 26, 'signal': 9}
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    ATR_PERIOD = 14
    
    # Machine Learning Parameters
    TRAIN_TEST_SPLIT_RATIO = 0.8
    RANDOM_STATE = 42
    
    # API Rate Limiting
    API_CALL_DELAY = 12  # seconds between API calls (Alpha Vantage free tier: 5 calls per minute)
