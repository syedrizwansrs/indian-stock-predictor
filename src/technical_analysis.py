"""
Technical Analysis Module for Stock Market Data.
Calculates technical indicators and prepares features for machine learning.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from .config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import pandas_ta, but don't fail if it's not available
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas_ta not available, using manual implementations")

class TechnicalAnalyzer:
    """Handles calculation of technical indicators and feature engineering."""
    
    def __init__(self):
        """Initialize the TechnicalAnalyzer."""
        self.indicators = {}
    
    def calculate_sma(self, data: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            periods (List[int]): List of SMA periods
            
        Returns:
            pd.DataFrame: Data with SMA columns added
        """
        if periods is None:
            periods = Config.SMA_PERIODS
        
        result = data.copy()
        
        for period in periods:
            col_name = f'SMA_{period}'
            result[col_name] = data['close'].rolling(window=period).mean()
            logger.debug(f"Calculated {col_name}")
        
        return result
    
    def calculate_ema(self, data: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            periods (List[int]): List of EMA periods
            
        Returns:
            pd.DataFrame: Data with EMA columns added
        """
        if periods is None:
            periods = Config.EMA_PERIODS
        
        result = data.copy()
        
        for period in periods:
            col_name = f'EMA_{period}'
            result[col_name] = data['close'].ewm(span=period).mean()
            logger.debug(f"Calculated {col_name}")
        
        return result
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            period (int): RSI period
            
        Returns:
            pd.DataFrame: Data with RSI column added
        """
        if period is None:
            period = Config.RSI_PERIOD
        
        result = data.copy()
        
        # Manual RSI calculation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        result['RSI'] = 100 - (100 / (1 + rs))
        
        logger.debug(f"Calculated RSI with period {period}")
        
        return result
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = None, slow: int = None, signal: int = None) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            fast (int): Fast EMA period
            slow (int): Slow EMA period
            signal (int): Signal line EMA period
            
        Returns:
            pd.DataFrame: Data with MACD columns added
        """
        if fast is None:
            fast = Config.MACD_PERIODS['fast']
        if slow is None:
            slow = Config.MACD_PERIODS['slow']
        if signal is None:
            signal = Config.MACD_PERIODS['signal']
        
        result = data.copy()
        
        # Manual MACD calculation
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        
        result['MACD'] = ema_fast - ema_slow
        result['MACD_Signal'] = result['MACD'].ewm(span=signal).mean()
        result['MACD_Histogram'] = result['MACD'] - result['MACD_Signal']
        
        logger.debug(f"Calculated MACD with periods {fast}, {slow}, {signal}")
        
        return result
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = None, std_dev: float = None) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            period (int): Moving average period
            std_dev (float): Standard deviation multiplier
            
        Returns:
            pd.DataFrame: Data with Bollinger Bands columns added
        """
        if period is None:
            period = Config.BOLLINGER_PERIOD
        if std_dev is None:
            std_dev = Config.BOLLINGER_STD
        
        result = data.copy()
        
        # Manual Bollinger Bands calculation
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        
        result['BB_Upper'] = sma + (std * std_dev)
        result['BB_Middle'] = sma
        result['BB_Lower'] = sma - (std * std_dev)
        result['BB_Width'] = result['BB_Upper'] - result['BB_Lower']
        result['BB_Position'] = (data['close'] - result['BB_Lower']) / (result['BB_Upper'] - result['BB_Lower'])
        
        logger.debug(f"Calculated Bollinger Bands with period {period} and std {std_dev}")
        
        return result
    
    def calculate_atr(self, data: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        Calculate Average True Range.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            period (int): ATR period
            
        Returns:
            pd.DataFrame: Data with ATR column added
        """
        if period is None:
            period = Config.ATR_PERIOD
        
        result = data.copy()
        
        # Manual ATR calculation
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.DataFrame({
            'hl': high_low,
            'hc': high_close_prev,
            'lc': low_close_prev
        }).max(axis=1)
        
        result['ATR'] = true_range.rolling(window=period).mean()
        logger.debug(f"Calculated ATR with period {period}")
        
        return result
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On-Balance Volume.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with OBV column added
        """
        result = data.copy()
        
        # Manual OBV calculation
        price_change = data['close'].diff()
        volume_direction = np.where(price_change > 0, data['volume'], 
                                  np.where(price_change < 0, -data['volume'], 0))
        result['OBV'] = volume_direction.cumsum()
        
        logger.debug("Calculated OBV")
        
        return result
    
    def calculate_additional_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional technical indicators.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with additional indicators
        """
        result = data.copy()
        
        # Stochastic Oscillator - Manual calculation
        k_period = 14
        d_period = 3
        low_k = data['low'].rolling(window=k_period).min()
        high_k = data['high'].rolling(window=k_period).max()
        result['Stoch_K'] = 100 * ((data['close'] - low_k) / (high_k - low_k))
        result['Stoch_D'] = result['Stoch_K'].rolling(window=d_period).mean()
        
        # Williams %R - Manual calculation
        period = 14
        low_period = data['low'].rolling(window=period).min()
        high_period = data['high'].rolling(window=period).max()
        result['Williams_R'] = -100 * ((high_period - data['close']) / (high_period - low_period))
        
        # Commodity Channel Index - Manual calculation
        cci_period = 20
        tp = (data['high'] + data['low'] + data['close']) / 3  # Typical price
        sma_tp = tp.rolling(window=cci_period).mean()
        # Use mean absolute deviation manually, since pd.Series.mad() is deprecated/removed
        mad = tp.rolling(window=cci_period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        result['CCI'] = (tp - sma_tp) / (0.015 * mad)
        
        # Momentum - Manual calculation
        momentum_period = 10
        result['Momentum'] = data['close'] / data['close'].shift(momentum_period)
        
        # Rate of Change - Manual calculation
        roc_period = 10
        result['ROC'] = ((data['close'] - data['close'].shift(roc_period)) / data['close'].shift(roc_period)) * 100
        
        logger.debug("Calculated additional indicators")
        return result
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with all indicators added
        """
        logger.info("Calculating all technical indicators...")
        
        result = data.copy()
        
        # Calculate all indicators
        result = self.calculate_sma(result)
        result = self.calculate_ema(result)
        result = self.calculate_rsi(result)
        result = self.calculate_macd(result)
        result = self.calculate_bollinger_bands(result)
        result = self.calculate_atr(result)
        result = self.calculate_obv(result)
        result = self.calculate_additional_indicators(result)
        
        logger.info(f"Calculated {len(result.columns) - len(data.columns)} technical indicators")
        return result
    
    def engineer_features(self, data: pd.DataFrame, lookback_periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Engineer features for machine learning.
        
        Args:
            data (pd.DataFrame): Stock data with technical indicators
            lookback_periods (List[int]): Periods for creating lagged features
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        logger.info("Engineering features for machine learning...")
        
        result = data.copy()
        
        # Price-based features
        result['Returns'] = result['close'].pct_change()
        result['Log_Returns'] = np.log(result['close'] / result['close'].shift(1))
        result['Price_Change'] = result['close'] - result['close'].shift(1)
        result['High_Low_Ratio'] = result['high'] / result['low']
        result['Close_Open_Ratio'] = result['close'] / result['open']
        
        # Volume-based features
        result['Volume_Change'] = result['volume'].pct_change()
        result['Volume_MA_Ratio'] = result['volume'] / result['volume'].rolling(20).mean()
        result['Price_Volume'] = result['close'] * result['volume']
        
        # Volatility features
        result['Volatility'] = result['Returns'].rolling(20).std()
        result['High_Low_Pct'] = (result['high'] - result['low']) / result['close']
        
        # Moving average ratios
        if 'SMA_50' in result.columns and 'SMA_200' in result.columns:
            result['SMA_Ratio'] = result['SMA_50'] / result['SMA_200']
        
        if 'EMA_12' in result.columns and 'EMA_26' in result.columns:
            result['EMA_Ratio'] = result['EMA_12'] / result['EMA_26']
        
        # Price position relative to moving averages
        for period in Config.SMA_PERIODS:
            if f'SMA_{period}' in result.columns:
                result[f'Price_SMA_{period}_Ratio'] = result['close'] / result[f'SMA_{period}']
        
        # RSI-based features
        if 'RSI' in result.columns:
            result['RSI_Overbought'] = (result['RSI'] > 70).astype(int)
            result['RSI_Oversold'] = (result['RSI'] < 30).astype(int)
        
        # Create lagged features
        for period in lookback_periods:
            result[f'Returns_Lag_{period}'] = result['Returns'].shift(period)
            result[f'Volume_Change_Lag_{period}'] = result['Volume_Change'].shift(period)
            result[f'RSI_Lag_{period}'] = result['RSI'].shift(period) if 'RSI' in result.columns else np.nan
            result[f'Close_Lag_{period}'] = result['close'].shift(period)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            result[f'Returns_Mean_{window}'] = result['Returns'].rolling(window).mean()
            result[f'Returns_Std_{window}'] = result['Returns'].rolling(window).std()
            result[f'Volume_Mean_{window}'] = result['volume'].rolling(window).mean()
            result[f'High_Mean_{window}'] = result['high'].rolling(window).mean()
            result[f'Low_Mean_{window}'] = result['low'].rolling(window).mean()
        
        # Target variable for prediction (next day's price direction)
        result['Target'] = (result['close'].shift(-1) > result['close']).astype(int)
        
        # Additional target variables
        result['Target_Return'] = result['close'].shift(-1) / result['close'] - 1
        result['Target_Price'] = result['close'].shift(-1)
        
        logger.info(f"Engineered {len(result.columns) - len(data.columns)} features")
        return result
    
    def get_feature_importance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for feature importance analysis.
        
        Args:
            data (pd.DataFrame): Data with all indicators and features
            
        Returns:
            pd.DataFrame: Clean data ready for ML models
        """
        # Remove rows with NaN values
        clean_data = data.dropna()
        
        # Remove target variables from features
        feature_cols = [col for col in clean_data.columns 
                       if not col.startswith('Target') and col not in ['open', 'high', 'low', 'close', 'volume']]
        
        features = clean_data[feature_cols]
        target = clean_data['Target'] if 'Target' in clean_data.columns else None
        
        logger.info(f"Prepared {len(features.columns)} features for ML models")
        return features, target
    
    def process_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Complete processing pipeline for stock data.
        
        Args:
            data (pd.DataFrame): Raw stock data
            
        Returns:
            pd.DataFrame: Processed data with all indicators and features
        """
        logger.info("Starting complete stock data processing pipeline...")
        
        # Calculate technical indicators
        processed_data = self.calculate_all_indicators(data)
        
        # Engineer features
        processed_data = self.engineer_features(processed_data)
        
        logger.info("Stock data processing pipeline completed")
        return processed_data

# Example usage
if __name__ == "__main__":
    # Example with sample data
    from .data_fetcher import DataFetcher
    
    # Initialize modules
    fetcher = DataFetcher()
    analyzer = TechnicalAnalyzer()
    
    # Fetch sample data
    symbol = "RELIANCE.BSE"
    data = fetcher.load_from_database(symbol)
    
    if data is not None:
        # Process the data
        processed_data = analyzer.process_stock_data(data)
        
        print(f"Original data shape: {data.shape}")
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Added {processed_data.shape[1] - data.shape[1]} columns")
        
        # Show some results
        print("\nTechnical Indicators (last 5 rows):")
        indicator_cols = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'ATR', 'OBV']
        available_cols = [col for col in indicator_cols if col in processed_data.columns]
        print(processed_data[available_cols].tail())
        
        # Show feature engineering results
        print("\nEngineered Features (last 5 rows):")
        feature_cols = ['Returns', 'Volatility', 'SMA_Ratio', 'Target']
        available_features = [col for col in feature_cols if col in processed_data.columns]
        print(processed_data[available_features].tail())
    else:
        print(f"No data available for {symbol}. Please run data_fetcher.py first.")
