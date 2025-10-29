"""Data loading and preprocessing module for cryptocurrency data."""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple


class CryptoDataLoader:
    """Load and preprocess cryptocurrency data."""
    
    def __init__(self, symbol: str = "BTC-USD", period: str = "2y"):
        """
        Initialize data loader.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC-USD', 'ETH-USD')
            period: Data period (e.g., '1y', '2y', '5y', 'max')
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load cryptocurrency data from Yahoo Finance.
        
        Returns:
            DataFrame with cryptocurrency price data
        """
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period)
        return self.data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators as features.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        df = data.copy()
        
        # Moving averages
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_14'] = df['Close'].rolling(window=14).mean()
        df['MA_30'] = df['Close'].rolling(window=30).mean()
        
        # Exponential moving averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=14).std()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_7d'] = df['Close'].pct_change(periods=7)
        
        return df
    
    def prepare_sequences(self, data: pd.DataFrame, features: List[str], 
                         target: str = 'Close', seq_length: int = 30,
                         train_size: float = 0.8) -> Tuple:
        """
        Prepare sequences for time series prediction.
        
        Args:
            data: Input DataFrame
            features: List of feature column names
            target: Target column name
            seq_length: Sequence length for input
            train_size: Proportion of data for training
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler)
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # Remove NaN values
        df = data[features + [target]].dropna()
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Create sequences
        X, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X.append(scaled_data[i-seq_length:i, :-1])  # All features
            y.append(scaled_data[i, -1])  # Target
            
        X, y = np.array(X), np.array(y)
        
        # Split train/test
        split_idx = int(len(X) * train_size)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test, scaler
    
    def get_feature_sets(self) -> dict:
        """
        Get different combinations of input features for evaluation.
        
        Returns:
            Dictionary of feature sets with descriptive names
        """
        feature_sets = {
            'price_only': ['Close'],
            'ohlcv': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'price_ma': ['Close', 'MA_7', 'MA_14', 'MA_30'],
            'technical_basic': ['Close', 'MA_7', 'MA_14', 'RSI', 'Volume'],
            'technical_full': ['Close', 'MA_7', 'MA_14', 'MA_30', 'EMA_12', 
                              'EMA_26', 'MACD', 'RSI', 'Volatility', 'Volume'],
            'all_features': ['Open', 'High', 'Low', 'Close', 'Volume', 
                            'MA_7', 'MA_14', 'MA_30', 'EMA_12', 'EMA_26',
                            'MACD', 'MACD_signal', 'RSI', 'BB_upper', 'BB_lower',
                            'Volatility', 'Price_Change', 'Price_Change_7d']
        }
        return feature_sets
