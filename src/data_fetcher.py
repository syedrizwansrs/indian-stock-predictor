"""
Data Acquisition Module for Indian Stock Market Data.
Fetches historical OHLCV data using Alpha Vantage API and stores it locally.
"""
import os
import time
import pandas as pd
import sqlite3
import requests
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List
from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    def clear_symbol_cache(self, symbol: str):
        """
        Delete all cached data for a specific stock symbol.
        Args:
            symbol (str): Stock symbol to clear from cache
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM stocks WHERE symbol = ?", (symbol,))
            conn.commit()
            conn.close()
            logger.info(f"Cleared cache for symbol: {symbol}")
        except Exception as e:
            logger.error(f"Error clearing cache for {symbol}: {str(e)}")
    """Handles fetching and storing stock market data."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the DataFetcher.
        
        Args:
            api_key (str): Alpha Vantage API key. If None, uses config default.
        """
        self.api_key = api_key or Config.ALPHA_VANTAGE_API_KEY
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.db_path = Config.DATABASE_PATH
        self._ensure_data_directory()
        self._init_database()
    
    def _ensure_data_directory(self):
        """Create data directory if it doesn't exist."""
        os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database for storing stock data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create stocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def fetch_alpha_vantage_data(self, symbol: str, outputsize: str = 'full') -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Alpha Vantage API.
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.BSE')
            outputsize (str): 'compact' for recent 100 days, 'full' for 20+ years
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Convert Indian symbols to Alpha Vantage format
            if '.BSE' in symbol or '.NSE' in symbol:
                av_symbol = symbol.replace('.BSE', '.BSE').replace('.NSE', '.NSE')
            else:
                av_symbol = f"{symbol}.BSE"  # Default to BSE
            
            logger.info(f"Fetching data for {av_symbol} from Alpha Vantage...")
            data, meta_data = self.ts.get_daily(symbol=av_symbol, outputsize=outputsize)
            
            # Rename columns to standard format
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            data.index.name = 'date'
            data = data.sort_index()
            
            # Convert to numeric types
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from Alpha Vantage for {symbol}: {str(e)}")
            return None
    
    def fetch_yfinance_data(self, symbol: str, period: str = 'max') -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance as a fallback.
        
        Args:
            symbol (str): Stock symbol (e.g., 'RELIANCE.NS' for NSE)
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Convert to Yahoo Finance format
            if '.BSE' in symbol:
                yf_symbol = symbol.replace('.BSE', '.BO')
            elif '.NSE' in symbol:
                yf_symbol = symbol.replace('.NSE', '.NS')
            else:
                yf_symbol = f"{symbol}.NS"  # Default to NSE
            
            logger.info(f"Fetching data for {yf_symbol} from Yahoo Finance...")
            
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for {yf_symbol}")
                return None
            
            # Rename columns to standard format
            data.columns = [col.lower() for col in data.columns]
            data.index.name = 'date'
            data = data[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
            return None
    
    def save_to_database(self, symbol: str, data: pd.DataFrame):
        """
        Save stock data to SQLite database, preventing true duplicates.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Stock data to save
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Prepare data for insertion
            data_to_insert = data.copy()
            data_to_insert['symbol'] = symbol
            data_to_insert = data_to_insert.reset_index()
            # Insert each row, skipping duplicates
            inserted = 0
            for _, row in data_to_insert.iterrows():
                try:
                    # Ensure date is a string in YYYY-MM-DD format
                    date_val = row['date']
                    if pd.isnull(date_val):
                        date_str = None
                    elif hasattr(date_val, 'strftime'):
                        date_str = date_val.strftime('%Y-%m-%d')
                    else:
                        date_str = str(date_val)
                    cursor.execute(
                        """
                        INSERT INTO stocks (symbol, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row['symbol'],
                            date_str,
                            row['open'] if not pd.isnull(row['open']) else None,
                            row['high'] if not pd.isnull(row['high']) else None,
                            row['low'] if not pd.isnull(row['low']) else None,
                            row['close'] if not pd.isnull(row['close']) else None,
                            int(row['volume']) if not pd.isnull(row['volume']) else None
                        )
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    # Duplicate entry, skip
                    continue
                except Exception as e:
                    logger.error(f"Error inserting row for {symbol} on {row['date']}: {str(e)}")
            conn.commit()
            conn.close()
            logger.info(f"Saved {inserted} new records for {symbol} to database (duplicates skipped)")
        except Exception as e:
            logger.error(f"Error saving data to database for {symbol}: {str(e)}")
    
    def save_to_csv(self, symbol: str, data: pd.DataFrame):
        """
        Save stock data to CSV file.
        
        Args:
            symbol (str): Stock symbol
            data (pd.DataFrame): Stock data to save
        """
        try:
            filename = f"{symbol.replace('.', '_')}_daily.csv"
            filepath = os.path.join(Config.DATA_DIR, filename)
            data.to_csv(filepath)
            logger.info(f"Saved data for {symbol} to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data to CSV for {symbol}: {str(e)}")
    
    def load_from_database(self, symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load stock data from SQLite database.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Stock data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT date, open, high, low, close, volume FROM stocks WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            data = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not data.empty:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
                logger.info(f"Loaded {len(data)} records for {symbol} from database")
                return data
            else:
                logger.warning(f"No data found for {symbol} in database")
                return None
                
        except Exception as e:
            logger.error(f"Error loading data from database for {symbol}: {str(e)}")
            return None
    
    def fetch_and_store_data(self, symbol: str, use_fallback: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch stock data and store it locally.
        
        Args:
            symbol (str): Stock symbol
            use_fallback (bool): Whether to use Yahoo Finance as fallback
            
        Returns:
            pd.DataFrame: Fetched stock data
        """
        # Try Alpha Vantage first
        data = self.fetch_alpha_vantage_data(symbol)
        
        # If Alpha Vantage fails and fallback is enabled, try Yahoo Finance
        if data is None and use_fallback:
            logger.info("Alpha Vantage failed, trying Yahoo Finance...")
            data = self.fetch_yfinance_data(symbol)
        
        if data is not None:
            # Save to both database and CSV
            self.save_to_database(symbol, data)
            self.save_to_csv(symbol, data)
            
            # Respect API rate limits
            time.sleep(Config.API_CALL_DELAY)
            
            return data
        else:
            logger.error(f"Failed to fetch data for {symbol} from all sources")
            return None
    
    def get_latest_data_date(self, symbol: str) -> Optional[datetime]:
        """
        Get the latest date for which data is available for a symbol.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            datetime: Latest data date
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT MAX(date) FROM stocks WHERE symbol = ?",
                (symbol,)
            )
            
            result = cursor.fetchone()[0]
            conn.close()
            
            if result:
                return datetime.strptime(result, '%Y-%m-%d')
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest data date for {symbol}: {str(e)}")
            return None
    
    def update_stock_data(self, symbol: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Update stock data by fetching only missing recent data, or force refresh from API.
        
        Args:
            symbol (str): Stock symbol
            force_refresh (bool): If True, always fetch fresh data from API and update cache.
        Returns:
            pd.DataFrame: Updated stock data
        """
        if force_refresh:
            logger.info(f"Force refresh enabled for {symbol}, fetching all historical data from API.")
            return self.fetch_and_store_data(symbol)
        latest_date = self.get_latest_data_date(symbol)
        if latest_date is None:
            # No existing data, fetch all
            logger.info(f"No existing data for {symbol}, fetching all historical data")
            return self.fetch_and_store_data(symbol)
        # Check if data is recent (within last 7 days)
        days_old = (datetime.now() - latest_date).days
        if days_old <= 7:
            logger.info(f"Data for {symbol} is recent ({days_old} days old), loading from database")
            return self.load_from_database(symbol)
        else:
            logger.info(f"Data for {symbol} is {days_old} days old, fetching updates")
            return self.fetch_and_store_data(symbol)
    
    def bulk_fetch_stocks(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks.
        
        Args:
            symbols (List[str]): List of stock symbols
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            data = self.update_stock_data(symbol)
            
            if data is not None:
                results[symbol] = data
            
            # Respect rate limits between API calls
            time.sleep(Config.API_CALL_DELAY)
        
        logger.info(f"Successfully fetched data for {len(results)} out of {len(symbols)} symbols")
        return results

# Example usage
if __name__ == "__main__":
    # Initialize data fetcher
    fetcher = DataFetcher()
    
    # Fetch data for a few NIFTY 50 stocks
    test_symbols = Config.DEFAULT_STOCKS[:3]  # Test with first 3 stocks
    
    results = fetcher.bulk_fetch_stocks(test_symbols)
    
    for symbol, data in results.items():
        print(f"\n{symbol}: {len(data)} records")
        print(data.head())
        print(data.tail())
