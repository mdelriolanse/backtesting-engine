"""
Simple Machine Learning layer for the backtesting engine
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import json
import random

logger = logging.getLogger(__name__)

class SimpleFeatureEngineer:
    """Simple feature engineering for price data with technical indicators"""
    
    @staticmethod
    def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators from OHLCV data"""
        df = df.copy()
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi'] = SimpleFeatureEngineer._calculate_rsi(df['close'], 14)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Price changes
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_10d'] = df['close'].pct_change(10)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Volatility
        df['volatility_5d'] = df['price_change_1d'].rolling(window=5).std()
        df['volatility_20d'] = df['price_change_1d'].rolling(window=20).std()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class SimpleMLPredictor:
    """Simple ML predictor for price movements"""
    
    def __init__(self):
        self.price_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.direction_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        # Create technical indicators
        df_with_indicators = SimpleFeatureEngineer.create_technical_indicators(df)
        
        # Select feature columns
        feature_columns = [
            'sma_5', 'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'bb_position', 'bb_width',
            'price_change_1d', 'price_change_5d', 'price_change_10d',
            'volume_ratio', 'volatility_5d', 'volatility_20d'
        ]
        
        # Create target variables
        df_with_indicators['future_price_1d'] = df_with_indicators['close'].shift(-1)
        df_with_indicators['future_return_1d'] = df_with_indicators['close'].pct_change().shift(-1)
        df_with_indicators['future_direction'] = (df_with_indicators['future_return_1d'] > 0).astype(int)
        
        # Store feature columns
        self.feature_columns = feature_columns
        
        return df_with_indicators
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train ML models on historical data"""
        try:
            # Prepare features
            df_features = self.prepare_features(df)
            
            # Remove rows with NaN values
            df_clean = df_features.dropna()
            
            if len(df_clean) < 100:
                raise ValueError("Not enough data for training (need at least 100 samples)")
            
            # Prepare features and targets
            X = df_clean[self.feature_columns]
            y_price = df_clean['future_price_1d']
            y_direction = df_clean['future_direction']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_price_train, y_price_test, y_dir_train, y_dir_test = train_test_split(
                X_scaled, y_price, y_direction, test_size=0.2, random_state=42
            )
            
            # Train models
            self.price_predictor.fit(X_train, y_price_train)
            self.direction_predictor.fit(X_train, y_dir_train)
            
            # Evaluate models
            price_pred = self.price_predictor.predict(X_test)
            direction_pred = self.direction_predictor.predict(X_test)
            
            price_mse = mean_squared_error(y_price_test, price_pred)
            direction_accuracy = accuracy_score(y_dir_test, direction_pred)
            
            self.is_trained = True
            
            return {
                "price_mse": price_mse,
                "direction_accuracy": direction_accuracy,
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            raise e
    
    def predict(self, df: pd.DataFrame) -> List[Dict]:
        """Generate predictions for the given data"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        try:
            # Prepare features
            df_features = self.prepare_features(df)
            
            # Get the latest data point
            latest_data = df_features.iloc[-1:]
            
            # Check if we have all required features
            if latest_data[self.feature_columns].isna().any().any():
                raise ValueError("Insufficient data for prediction")
            
            # Prepare features
            X = latest_data[self.feature_columns]
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            price_pred = self.price_predictor.predict(X_scaled)[0]
            direction_pred = self.direction_predictor.predict(X_scaled)[0]
            direction_proba = self.direction_predictor.predict_proba(X_scaled)[0]
            
            # Calculate confidence
            confidence = max(direction_proba)
            
            # Determine signal
            current_price = latest_data['close'].iloc[0]
            price_change_pct = (price_pred - current_price) / current_price
            
            if direction_pred == 1 and price_change_pct > 0.01:  # 1% threshold
                signal_type = "BUY"
            elif direction_pred == 0 and price_change_pct < -0.01:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"
            
            return [{
                "timestamp": latest_data.index[0].isoformat(),
                "model_name": "RandomForest",
                "prediction": price_pred,
                "confidence": confidence,
                "signal_type": signal_type,
                "price_change_pct": price_change_pct
            }]
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise e

    def predict_series(self, df: pd.DataFrame, every_n: int = 1) -> List[Dict]:
        """Generate predictions across the series, optionally sampling every_n rows"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        try:
            df_features = self.prepare_features(df)
            df_clean = df_features.dropna()
            if df_clean.empty:
                return []
            X_all = df_clean[self.feature_columns]
            X_scaled = self.scaler.transform(X_all)
            price_preds = self.price_predictor.predict(X_scaled)
            direction_preds = self.direction_predictor.predict(X_scaled)
            direction_probas = self.direction_predictor.predict_proba(X_scaled)
            results = []
            for idx, (ts, row) in enumerate(df_clean.iterrows()):
                if every_n > 1 and (idx % every_n) != 0:
                    continue
                price_pred = float(price_preds[idx])
                direction_pred = int(direction_preds[idx])
                confidence = float(max(direction_probas[idx]))
                current_price = float(row['close'])
                price_change_pct = (price_pred - current_price) / current_price if current_price else 0.0
                if direction_pred == 1 and price_change_pct > 0.01:
                    signal_type = "BUY"
                elif direction_pred == 0 and price_change_pct < -0.01:
                    signal_type = "SELL"
                else:
                    signal_type = "HOLD"
                results.append({
                    "timestamp": ts.isoformat(),
                    "model_name": "RandomForest",
                    "prediction": price_pred,
                    "confidence": confidence,
                    "signal_type": signal_type,
                    "price_change_pct": price_change_pct
                })
            return results
        except Exception as e:
            logger.error(f"Error making series predictions: {e}")
            raise e

class SimpleMLManager:
    """Simple ML manager for the backtesting engine"""
    
    def __init__(self):
        self.predictors = {}  # symbol -> SimpleMLPredictor
        self.training_results = {}  # symbol -> training results
    
    def train_models(self, symbol: str, price_data: List[Dict]) -> Dict[str, float]:
        """Train ML models for a symbol"""
        try:
            # Convert price data to DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Create predictor if it doesn't exist
            if symbol not in self.predictors:
                self.predictors[symbol] = SimpleMLPredictor()
            
            # Train models
            results = self.predictors[symbol].train(df)
            self.training_results[symbol] = results
            
            logger.info(f"Trained ML models for {symbol}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error training models for {symbol}: {e}")
            raise e
    
    def generate_signals(self, symbol: str, price_data: List[Dict], every_n: int = 1) -> List[Dict]:
        """Generate ML signals for a symbol. If every_n>1, return a sampled series."""
        try:
            if symbol not in self.predictors or not self.predictors[symbol].is_trained:
                return []
            
            # Convert price data to DataFrame
            df = pd.DataFrame(price_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Generate predictions
            if every_n and every_n > 1:
                signals = self.predictors[symbol].predict_series(df, every_n=every_n)
            else:
                signals = self.predictors[symbol].predict(df)
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {e}")
            return []
    
    def get_training_status(self, symbol: str) -> Dict:
        """Get training status for a symbol"""
        if symbol in self.training_results:
            return {
                "trained": True,
                "results": self.training_results[symbol]
            }
        else:
            return {"trained": False}

# Global ML manager instance
ml_manager = SimpleMLManager()
