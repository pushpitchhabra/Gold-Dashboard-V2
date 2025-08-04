"""
Model Training Module for Gold Trading Dashboard
Trains XGBoost classifier and saves the model for predictions
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import yaml
import os
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldModelTrainer:
    def __init__(self, config_path="config.yaml"):
        """Initialize model trainer with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_file = self.config['model']['model_file']
        self.feature_file = self.config['model']['feature_file']
        self.training_months = self.config['model']['training_months']
        
        # XGBoost parameters
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = None
        self.feature_columns = None
    
    def prepare_training_data(self, df_with_features, feature_columns):
        """
        Prepare data for training - use only recent months
        """
        logger.info("Preparing training data...")
        
        # Filter to recent months for training
        cutoff_date = datetime.now() - timedelta(days=30 * self.training_months)
        
        # Convert cutoff_date to timezone-aware if df index is timezone-aware
        if df_with_features.index.tz is not None:
            if cutoff_date.tzinfo is None:
                cutoff_date = cutoff_date.replace(tzinfo=df_with_features.index.tz)
        
        recent_data = df_with_features[df_with_features.index >= cutoff_date]
        
        logger.info(f"Using {len(recent_data)} records from last {self.training_months} months")
        logger.info(f"Date range: {recent_data.index.min()} to {recent_data.index.max()}")
        
        # Prepare features and target
        X = recent_data[feature_columns].copy()
        y = recent_data['Target'].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(X.mean())
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, X, y):
        """
        Train XGBoost model with cross-validation
        """
        logger.info("Training XGBoost model...")
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize XGBoost classifier
        self.model = xgb.XGBClassifier(**self.xgb_params)
        
        # Train the model
        try:
            # Try with early stopping (newer XGBoost versions)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        except TypeError:
            # Fallback for older XGBoost versions
            logger.info("Using fallback training method for XGBoost compatibility")
            self.model.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_val, y_pred))
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        logger.info(feature_importance.head(10).to_string(index=False))
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
    
    def save_model(self, feature_columns):
        """
        Save trained model and feature columns
        """
        logger.info("Saving model and features...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        
        # Save model
        joblib.dump(self.model, self.model_file)
        logger.info(f"Model saved to {self.model_file}")
        
        # Save feature columns
        joblib.dump(feature_columns, self.feature_file)
        logger.info(f"Feature columns saved to {self.feature_file}")
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'training_months': self.training_months,
            'n_features': len(feature_columns),
            'model_params': self.xgb_params
        }
        
        metadata_file = self.model_file.replace('.pkl', '_metadata.yaml')
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f)
        
        logger.info(f"Metadata saved to {metadata_file}")
    
    def load_model(self):
        """
        Load saved model and feature columns
        """
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.feature_file):
                self.model = joblib.load(self.model_file)
                self.feature_columns = joblib.load(self.feature_file)
                
                # Load metadata if available
                metadata = {}
                metadata_file = self.model_file.replace('.pkl', '_metadata.yaml')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = yaml.safe_load(f)
                
                logger.info("Model and features loaded successfully")
                return self.model, self.feature_columns, metadata
            else:
                logger.warning("Model or feature files not found")
                return None, None, {}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None, {}
    
    def train_and_save_model(self, df_with_features, feature_columns):
        """
        Complete training pipeline: prepare data, train, and save model
        """
        logger.info("Starting complete training pipeline...")
        
        # Prepare training data
        X, y = self.prepare_training_data(df_with_features, feature_columns)
        
        # Check if we have enough data
        if len(X) < 100:
            raise ValueError(f"Not enough data for training. Only {len(X)} samples available.")
        
        # Train model
        metrics = self.train_model(X, y)
        
        # Save model and features
        self.save_model(feature_columns)
        self.feature_columns = feature_columns
        
        logger.info("Training pipeline completed successfully!")
        return metrics
    
    def predict_probability(self, features):
        """
        Predict probability of price increase
        """
        if self.model is None:
            if not self.load_model():
                raise ValueError("No trained model available")
        
        # Ensure features are in the right order
        if self.feature_columns is not None:
            # Reorder features to match training
            features_ordered = features[self.feature_columns]
        else:
            features_ordered = features
        
        # Handle any NaN values
        features_ordered = features_ordered.fillna(features_ordered.mean())
        
        # Make prediction
        probability = self.model.predict_proba(features_ordered)[:, 1]
        prediction = self.model.predict(features_ordered)
        
        return probability[0], prediction[0]
    
    def get_model_info(self):
        """
        Get information about the current model
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        metadata_file = self.model_file.replace('.pkl', '_metadata.yaml')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = yaml.safe_load(f)
            return metadata
        else:
            return {
                'training_date': 'Unknown',
                'n_features': len(self.feature_columns) if self.feature_columns else 'Unknown'
            }

def main():
    """Test the model trainer"""
    from data_loader import GoldDataLoader
    from feature_engineer import GoldFeatureEngineer
    
    # Load and prepare data
    loader = GoldDataLoader()
    df = loader.load_historical_data()
    
    engineer = GoldFeatureEngineer()
    df_features, feature_cols = engineer.prepare_features_for_training(df)
    
    # Initialize and train model
    trainer = GoldModelTrainer()
    
    print("Training model...")
    metrics = trainer.train_and_save_model(df_features, feature_cols)
    
    print(f"\nTraining completed!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"CV Mean: {metrics['cv_mean']:.4f}")
    
    # Test prediction
    print("\nTesting prediction...")
    latest_features, _ = engineer.get_latest_features(df)
    if latest_features is not None:
        prob, pred = trainer.predict_probability(latest_features)
        print(f"Prediction probability: {prob:.4f}")
        print(f"Prediction: {'BUY' if pred == 1 else 'SELL'}")
    
    # Model info
    print("\nModel info:")
    info = trainer.get_model_info()
    print(info)

if __name__ == "__main__":
    main()
