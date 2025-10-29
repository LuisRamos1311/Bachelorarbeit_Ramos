"""Base model interface and implementations of modern prediction methods."""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class BasePredictor(ABC):
    """Base class for all prediction models."""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X_train, y_train):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    def get_name(self):
        """Return model name."""
        return self.__class__.__name__


class LSTMPredictor(BasePredictor):
    """LSTM-based prediction model."""
    
    def __init__(self, units: int = 50, dropout: float = 0.2, 
                 epochs: int = 50, batch_size: int = 32):
        """
        Initialize LSTM model.
        
        Args:
            units: Number of LSTM units
            dropout: Dropout rate
            epochs: Training epochs
            batch_size: Batch size for training
        """
        super().__init__()
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        
    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = keras.Sequential([
            layers.LSTM(self.units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(self.dropout),
            layers.LSTM(self.units, return_sequences=False),
            layers.Dropout(self.dropout),
            layers.Dense(25),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def fit(self, X_train, y_train):
        """Train the LSTM model."""
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        self.model.fit(X_train, y_train, 
                      epochs=self.epochs, 
                      batch_size=self.batch_size,
                      verbose=0)
        self.is_fitted = True
        
    def predict(self, X):
        """Make predictions with LSTM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X, verbose=0).flatten()


class GRUPredictor(BasePredictor):
    """GRU-based prediction model."""
    
    def __init__(self, units: int = 50, dropout: float = 0.2,
                 epochs: int = 50, batch_size: int = 32):
        """
        Initialize GRU model.
        
        Args:
            units: Number of GRU units
            dropout: Dropout rate
            epochs: Training epochs
            batch_size: Batch size for training
        """
        super().__init__()
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        
    def build_model(self, input_shape):
        """Build GRU model architecture."""
        model = keras.Sequential([
            layers.GRU(self.units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(self.dropout),
            layers.GRU(self.units, return_sequences=False),
            layers.Dropout(self.dropout),
            layers.Dense(25),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def fit(self, X_train, y_train):
        """Train the GRU model."""
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        self.model.fit(X_train, y_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      verbose=0)
        self.is_fitted = True
        
    def predict(self, X):
        """Make predictions with GRU."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X, verbose=0).flatten()


class SimpleRNNPredictor(BasePredictor):
    """Simple RNN-based prediction model."""
    
    def __init__(self, units: int = 50, dropout: float = 0.2,
                 epochs: int = 50, batch_size: int = 32):
        """
        Initialize Simple RNN model.
        
        Args:
            units: Number of RNN units
            dropout: Dropout rate
            epochs: Training epochs
            batch_size: Batch size for training
        """
        super().__init__()
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        
    def build_model(self, input_shape):
        """Build Simple RNN model architecture."""
        model = keras.Sequential([
            layers.SimpleRNN(self.units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(self.dropout),
            layers.SimpleRNN(self.units, return_sequences=False),
            layers.Dropout(self.dropout),
            layers.Dense(25),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def fit(self, X_train, y_train):
        """Train the Simple RNN model."""
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        self.model.fit(X_train, y_train,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      verbose=0)
        self.is_fitted = True
        
    def predict(self, X):
        """Make predictions with Simple RNN."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X, verbose=0).flatten()


class ARIMAPredictor(BasePredictor):
    """ARIMA-based prediction model."""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """
        Initialize ARIMA model.
        
        Args:
            order: ARIMA order (p, d, q)
        """
        super().__init__()
        self.order = order
        
    def fit(self, X_train, y_train):
        """Train the ARIMA model."""
        from statsmodels.tsa.arima.model import ARIMA
        
        # ARIMA works with 1D series, use last feature as input
        if len(X_train.shape) == 3:
            # Take the last timestep of last feature
            train_series = X_train[:, -1, -1]
        else:
            train_series = X_train.flatten()
            
        self.model = ARIMA(train_series, order=self.order)
        self.model = self.model.fit()
        self.is_fitted = True
        
    def predict(self, X):
        """Make predictions with ARIMA."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        for i in range(len(X)):
            pred = self.model.forecast(steps=1)[0]
            predictions.append(pred)
            
        return np.array(predictions)
