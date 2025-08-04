"""
Feature Engineering Module for Gold Trading Dashboard
Adds technical indicators and creates target column for ML training
"""

import pandas as pd
import numpy as np
import yaml
import logging

# Use alternative TA library for better compatibility
import ta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldFeatureEngineer:
    def __init__(self, config_path="config.yaml"):
        """Initialize feature engineer with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Get indicator parameters from config
        self.indicators = self.config['indicators']
        self.model_config = self.config['model']
        
        # Target generation parameters
        self.target_threshold = self.model_config['target_threshold']
        self.lookforward_periods = self.model_config['lookforward_periods']
    
    def add_technical_indicators(self, df):
        """
        Add all technical indicators to the dataframe
        """
        logger.info("Adding technical indicators...")
        
        # Make a copy to avoid modifying original data
        df_features = df.copy()
        
        # RSI (Relative Strength Index)
        df_features['RSI'] = ta.momentum.RSIIndicator(df_features['Close'], window=self.indicators['rsi_period']).rsi()
        
        # EMA (Exponential Moving Averages)
        df_features['EMA_Fast'] = ta.trend.EMAIndicator(df_features['Close'], window=self.indicators['ema_fast']).ema_indicator()
        df_features['EMA_Slow'] = ta.trend.EMAIndicator(df_features['Close'], window=self.indicators['ema_slow']).ema_indicator()
        df_features['EMA_Signal'] = df_features['EMA_Fast'] - df_features['EMA_Slow']
        
        # MACD (Moving Average Convergence Divergence)
        macd_indicator = ta.trend.MACD(df_features['Close'], 
                                      window_fast=self.indicators['macd_fast'],
                                      window_slow=self.indicators['macd_slow'],
                                      window_sign=self.indicators['macd_signal'])
        df_features['MACD'] = macd_indicator.macd()
        df_features['MACD_Signal'] = macd_indicator.macd_signal()
        df_features['MACD_Histogram'] = macd_indicator.macd_diff()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df_features['Close'], 
                                                   window=self.indicators['bb_period'],
                                                   window_dev=self.indicators['bb_std'])
        df_features['BB_Upper'] = bb_indicator.bollinger_hband()
        df_features['BB_Middle'] = bb_indicator.bollinger_mavg()
        df_features['BB_Lower'] = bb_indicator.bollinger_lband()
        df_features['BB_Width'] = (df_features['BB_Upper'] - df_features['BB_Lower']) / df_features['BB_Middle']
        df_features['BB_Position'] = (df_features['Close'] - df_features['BB_Lower']) / (df_features['BB_Upper'] - df_features['BB_Lower'])
        
        # ATR (Average True Range)
        df_features['ATR'] = ta.volatility.AverageTrueRange(df_features['High'], df_features['Low'], df_features['Close'], 
                                                           window=self.indicators['atr_period']).average_true_range()
        
        # Price-based features
        df_features['Price_Change'] = df_features['Close'].pct_change()
        df_features['Price_Change_MA'] = df_features['Price_Change'].rolling(window=10).mean()
        df_features['Volatility'] = df_features['Price_Change'].rolling(window=20).std()
        
        # Volume-based features (if volume data is available)
        if 'Volume' in df_features.columns:
            df_features['Volume_MA'] = df_features['Volume'].rolling(window=20).mean()
            df_features['Volume_Ratio'] = df_features['Volume'] / df_features['Volume_MA']
        
        # High-Low spread
        df_features['HL_Spread'] = (df_features['High'] - df_features['Low']) / df_features['Close']
        
        # Support and Resistance levels (simplified)
        df_features['Resistance'] = df_features['High'].rolling(window=20).max()
        df_features['Support'] = df_features['Low'].rolling(window=20).min()
        df_features['Distance_to_Resistance'] = (df_features['Resistance'] - df_features['Close']) / df_features['Close']
        df_features['Distance_to_Support'] = (df_features['Close'] - df_features['Support']) / df_features['Close']
        
        # Price Action Features - Advanced Candlestick Analysis
        df_features = self._add_price_action_features(df_features)
        
        # Time-based features
        df_features['Hour'] = df_features.index.hour
        df_features['DayOfWeek'] = df_features.index.dayofweek
        df_features['Month'] = df_features.index.month
        
        # Cyclical encoding for time features
        df_features['Hour_sin'] = np.sin(2 * np.pi * df_features['Hour'] / 24)
        df_features['Hour_cos'] = np.cos(2 * np.pi * df_features['Hour'] / 24)
        df_features['DayOfWeek_sin'] = np.sin(2 * np.pi * df_features['DayOfWeek'] / 7)
        df_features['DayOfWeek_cos'] = np.cos(2 * np.pi * df_features['DayOfWeek'] / 7)
        
        logger.info(f"Added technical indicators. Shape: {df_features.shape}")
        return df_features
    
    def _add_price_action_features(self, df_features):
        """
        Add advanced price action features including bullish/bearish candle conditions
        """
        # Calculate prior highs, lows, and closes
        df_features['Prior_High'] = df_features['High'].shift(1)
        df_features['Prior_Low'] = df_features['Low'].shift(1)
        df_features['Prior_Close'] = df_features['Close'].shift(1)
        df_features['Prior_Open'] = df_features['Open'].shift(1)
        
        # Bullish candle conditions (as requested)
        # High above prior high, Low above prior low, Close above prior low
        df_features['Bullish_Candle'] = (
            (df_features['High'] > df_features['Prior_High']) &
            (df_features['Low'] > df_features['Prior_Low']) &
            (df_features['Close'] > df_features['Prior_Low'])
        ).astype(int)
        
        # Bearish candle conditions (as requested)
        # High below prior low, Low below prior low, Close below prior low
        df_features['Bearish_Candle'] = (
            (df_features['High'] < df_features['Prior_Low']) &
            (df_features['Low'] < df_features['Prior_Low']) &
            (df_features['Close'] < df_features['Prior_Low'])
        ).astype(int)
        
        # Additional candlestick patterns
        # Doji (indecision)
        df_features['Doji'] = (
            abs(df_features['Close'] - df_features['Open']) <= 
            (df_features['High'] - df_features['Low']) * 0.1
        ).astype(int)
        
        # Hammer (bullish reversal)
        df_features['Hammer'] = (
            (df_features['Close'] > df_features['Open']) &
            ((df_features['Low'] - df_features['Open']) >= 
             2 * (df_features['Close'] - df_features['Open'])) &
            ((df_features['High'] - df_features['Close']) <= 
             (df_features['Close'] - df_features['Open']) * 0.1)
        ).astype(int)
        
        # Shooting Star (bearish reversal)
        df_features['Shooting_Star'] = (
            (df_features['Open'] > df_features['Close']) &
            ((df_features['High'] - df_features['Open']) >= 
             2 * (df_features['Open'] - df_features['Close'])) &
            ((df_features['Close'] - df_features['Low']) <= 
             (df_features['Open'] - df_features['Close']) * 0.1)
        ).astype(int)
        
        # Engulfing patterns
        df_features['Bullish_Engulfing'] = (
            (df_features['Close'] > df_features['Open']) &
            (df_features['Prior_Close'] < df_features['Prior_Open']) &
            (df_features['Open'] < df_features['Prior_Close']) &
            (df_features['Close'] > df_features['Prior_Open'])
        ).astype(int)
        
        df_features['Bearish_Engulfing'] = (
            (df_features['Close'] < df_features['Open']) &
            (df_features['Prior_Close'] > df_features['Prior_Open']) &
            (df_features['Open'] > df_features['Prior_Close']) &
            (df_features['Close'] < df_features['Prior_Open'])
        ).astype(int)
        
        # Body and shadow ratios
        df_features['Body_Size'] = abs(df_features['Close'] - df_features['Open'])
        df_features['Upper_Shadow'] = df_features['High'] - np.maximum(df_features['Open'], df_features['Close'])
        df_features['Lower_Shadow'] = np.minimum(df_features['Open'], df_features['Close']) - df_features['Low']
        df_features['Total_Range'] = df_features['High'] - df_features['Low']
        
        # Ratios (avoid division by zero)
        df_features['Body_To_Range_Ratio'] = np.where(
            df_features['Total_Range'] > 0,
            df_features['Body_Size'] / df_features['Total_Range'],
            0
        )
        
        df_features['Upper_Shadow_Ratio'] = np.where(
            df_features['Total_Range'] > 0,
            df_features['Upper_Shadow'] / df_features['Total_Range'],
            0
        )
        
        df_features['Lower_Shadow_Ratio'] = np.where(
            df_features['Total_Range'] > 0,
            df_features['Lower_Shadow'] / df_features['Total_Range'],
            0
        )
        
        # Price action momentum
        df_features['Price_Action_Momentum'] = (
            df_features['Bullish_Candle'] * 2 + 
            df_features['Bullish_Engulfing'] * 1.5 +
            df_features['Hammer'] * 1 -
            df_features['Bearish_Candle'] * 2 -
            df_features['Bearish_Engulfing'] * 1.5 -
            df_features['Shooting_Star'] * 1
        )
        
        # Rolling price action strength (last 5 candles)
        df_features['Price_Action_Strength_5'] = df_features['Price_Action_Momentum'].rolling(window=5).mean()
        
        return df_features
    
    def create_target_column(self, df):
        """
        Create target column for classification
        Target = 1 if price rises by more than threshold in next N periods, else 0
        """
        logger.info("Creating target column...")
        
        df_target = df.copy()
        
        # Calculate future returns
        future_close = df_target['Close'].shift(-self.lookforward_periods)
        current_close = df_target['Close']
        
        # Calculate percentage change
        future_return = (future_close - current_close) / current_close
        
        # Create binary target
        df_target['Target'] = (future_return > self.target_threshold).astype(int)
        
        # Remove rows where we can't calculate future returns
        df_target = df_target.iloc[:-self.lookforward_periods]
        
        logger.info(f"Target distribution: {df_target['Target'].value_counts().to_dict()}")
        return df_target
    
    def prepare_features_for_training(self, df):
        """
        Prepare features for machine learning training
        """
        logger.info("Preparing features for training...")
        
        # Add technical indicators
        df_features = self.add_technical_indicators(df)
        
        # Create target column
        df_with_target = self.create_target_column(df_features)
        
        # Select feature columns (exclude OHLCV and target)
        feature_columns = [col for col in df_with_target.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
        
        # Remove any columns with all NaN values
        feature_columns = [col for col in feature_columns 
                          if not df_with_target[col].isna().all()]
        
        # Remove rows with NaN values
        clean_data = df_with_target.dropna()
        
        logger.info(f"Feature columns: {len(feature_columns)}")
        logger.info(f"Clean data shape: {clean_data.shape}")
        
        # Separate features and target
        X = clean_data[feature_columns]
        y = clean_data['Target']
        
        # Ensure X and y have the same length
        min_length = min(len(X), len(y))
        X = X.iloc[:min_length]
        y = y.iloc[:min_length]
        
        logger.info(f"Final X shape: {X.shape}, y length: {len(y)}")
        
        return X, y
    
    def prepare_features_for_prediction(self, df):
        """
        Prepare features for prediction (without target column)
        """
        logger.info("Preparing features for prediction...")
        
        # Add technical indicators
        df_features = self.add_technical_indicators(df)
        
        # Select feature columns (exclude OHLCV)
        feature_columns = [col for col in df_features.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Remove any columns with all NaN values
        feature_columns = [col for col in feature_columns 
                          if not df_features[col].isna().all()]
        
        logger.info(f"Prediction feature columns: {len(feature_columns)}")
        
        return df_features, feature_columns
    
    def get_latest_features(self, df):
        """
        Get features for the most recent data point
        """
        df_features, feature_columns = self.prepare_features_for_prediction(df)
        
        # Get the last row with complete data
        latest_features = df_features[feature_columns].dropna().iloc[-1:]
        
        if latest_features.empty:
            logger.warning("No complete feature data available for latest prediction")
            return None, feature_columns
        
        return latest_features, feature_columns
    
    def get_indicator_summary(self, df):
        """
        Get a summary of current technical indicators for display
        """
        df_features = self.add_technical_indicators(df)
        latest = df_features.iloc[-1]
        
        summary = {
            'RSI': latest['RSI'],
            'MACD': latest['MACD'],
            'MACD_Signal': latest['MACD_Signal'],
            'BB_Position': latest['BB_Position'],
            'ATR': latest['ATR'],
            'EMA_Signal': latest['EMA_Signal'],
            'Price_Change': latest['Price_Change'],
            'Volatility': df_features['Price_Change'].rolling(window=20).std().iloc[-1]
        }
        
        return summary

def main():
    """Test the feature engineer"""
    from data_loader import GoldDataLoader
    
    # Load data
    loader = GoldDataLoader()
    df = loader.load_historical_data()
    
    # Initialize feature engineer
    engineer = GoldFeatureEngineer()
    
    # Test feature preparation
    print("Preparing features for training...")
    df_train, feature_cols = engineer.prepare_features_for_training(df)
    print(f"Training data shape: {df_train.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Test prediction features
    print("\nPreparing features for prediction...")
    df_pred, pred_cols = engineer.prepare_features_for_prediction(df)
    print(f"Prediction data shape: {df_pred.shape}")
    
    # Test latest features
    print("\nGetting latest features...")
    latest_features, _ = engineer.get_latest_features(df)
    if latest_features is not None:
        print(f"Latest features shape: {latest_features.shape}")
    
    # Test indicator summary
    print("\nIndicator summary:")
    summary = engineer.get_indicator_summary(df)
    for indicator, value in summary.items():
        print(f"{indicator}: {value:.4f}")

if __name__ == "__main__":
    main()
