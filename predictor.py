"""
Predictor Module for Gold Trading Dashboard
Makes predictions using trained model on live data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yaml
import os

# Import component classes
from data_loader import GoldDataLoader
from feature_engineer import GoldFeatureEngineer
from model_trainer import GoldModelTrainer
from strategy_analyzer import AdvancedStrategyAnalyzer
from macro_factors import MacroFactorsAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldPredictor:
    def __init__(self, config_path="config.yaml"):
        """Initialize predictor with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize components
        self.data_loader = GoldDataLoader()
        self.feature_engineer = GoldFeatureEngineer()
        self.model_trainer = GoldModelTrainer()
        self.strategy_analyzer = AdvancedStrategyAnalyzer()
        self.macro_analyzer = MacroFactorsAnalyzer()
        
        # Load model if available
        model, feature_columns, metadata = self.model_trainer.load_model()
        if model is not None and feature_columns is not None:
            self.model_trainer.model = model
            self.model_trainer.feature_columns = feature_columns
            logger.info(f"Predictor initialized with model and {len(feature_columns)} features")
        else:
            logger.warning("Predictor initialized without trained model")

    def get_live_prediction(self):
        """
        Get prediction for current market conditions
        """
        try:
            logger.info("Getting live prediction...")
            
            # Fetch recent data for feature calculation
            live_data = self.data_loader.fetch_live_data(days_back=30)
            
            if live_data.empty:
                logger.warning("No live data available")
                return self._get_fallback_prediction()
            
            # Get latest features
            latest_features, feature_columns = self.feature_engineer.get_latest_features(live_data)
            
            if latest_features is None or latest_features.empty:
                logger.warning("Could not calculate features from live data")
                return self._get_fallback_prediction()
            
            # Make prediction
            probability, prediction = self.model_trainer.predict_probability(latest_features)
            
            # Get current price and indicators
            current_price, current_time = self.data_loader.get_latest_price()
            indicators = self.feature_engineer.get_indicator_summary(live_data)
            
            # Create prediction result
            result = {
                'timestamp': current_time or datetime.now(),
                'current_price': current_price,
                'prediction': prediction,
                'probability': probability,
                'confidence': self._calculate_confidence(probability),
                'signal': self._get_trading_signal(prediction, probability),
                'indicators': indicators,
                'data_quality': 'live'
            }
            
            logger.info(f"Live prediction: {result['signal']} (prob: {probability:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error getting live prediction: {e}")
            return self._get_fallback_prediction()
    
    def _get_fallback_prediction(self):
        """
        Get fallback prediction when live data is not available
        """
        logger.info("Using fallback prediction...")
        
        try:
            # Try to load historical data
            historical_data = self.data_loader.load_historical_data()
            
            if not historical_data.empty:
                # Use last available data point
                latest_features, _ = self.feature_engineer.get_latest_features(historical_data)
                
                if latest_features is not None and not latest_features.empty:
                    probability, prediction = self.model_trainer.predict_probability(latest_features)
                    indicators = self.feature_engineer.get_indicator_summary(historical_data)
                    
                    result = {
                        'timestamp': historical_data.index[-1],
                        'current_price': historical_data['Close'].iloc[-1],
                        'prediction': prediction,
                        'probability': probability,
                        'confidence': self._calculate_confidence(probability),
                        'signal': self._get_trading_signal(prediction, probability),
                        'indicators': indicators,
                        'data_quality': 'historical'
                    }
                    
                    return result
        
        except Exception as e:
            logger.error(f"Error in fallback prediction: {e}")
        
        # Ultimate fallback
        return {
            'timestamp': datetime.now(),
            'current_price': None,
            'prediction': 0,
            'probability': 0.5,
            'confidence': 'Low',
            'signal': 'NEUTRAL',
            'indicators': {},
            'data_quality': 'unavailable'
        }
    
    def _calculate_confidence(self, probability):
        """
        Calculate confidence level based on probability
        """
        if probability >= 0.7 or probability <= 0.3:
            return 'High'
        elif probability >= 0.6 or probability <= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_trading_signal(self, prediction, probability):
        """
        Convert prediction to trading signal
        """
        if prediction == 1:
            if probability >= 0.7:
                return 'STRONG BUY'
            elif probability >= 0.6:
                return 'BUY'
            else:
                return 'WEAK BUY'
        else:
            if probability <= 0.3:
                return 'STRONG SELL'
            elif probability <= 0.4:
                return 'SELL'
            else:
                return 'WEAK SELL'
    
    def get_prediction_history(self, days=7):
        """
        Get prediction history for the last N days
        """
        try:
            # Load recent data
            recent_data = self.data_loader.fetch_live_data(days_back=days)
            
            if recent_data.empty:
                return []
            
            # Prepare features for all data points
            df_features, feature_columns = self.feature_engineer.prepare_features_for_prediction(recent_data)
            
            # Remove NaN rows
            df_clean = df_features.dropna()
            
            if df_clean.empty:
                return []
            
            # Make predictions for all points
            features_for_prediction = df_clean[feature_columns]
            probabilities = self.model_trainer.model.predict_proba(features_for_prediction)[:, 1]
            predictions = self.model_trainer.model.predict(features_for_prediction)
            
            # Create history
            history = []
            for i, (timestamp, row) in enumerate(df_clean.iterrows()):
                history.append({
                    'timestamp': timestamp,
                    'price': row['Close'],
                    'prediction': predictions[i],
                    'probability': probabilities[i],
                    'signal': self._get_trading_signal(predictions[i], probabilities[i])
                })
            
            return history[-50:]  # Return last 50 predictions
            
        except Exception as e:
            logger.error(f"Error getting prediction history: {e}")
            return []
    
    def analyze_prediction_accuracy(self, days=30):
        """
        Analyze recent prediction accuracy with robust error handling
        """
        try:
            logger.info(f"Analyzing prediction accuracy for last {days} days...")
            
            # Get historical data
            historical_data = self.data_loader.load_historical_data()
            
            if historical_data.empty:
                logger.warning("No historical data available for accuracy analysis")
                return None
            
            # Get recent data for analysis
            cutoff_date = datetime.now() - pd.Timedelta(days=days)
            
            # Handle timezone issues
            if historical_data.index.tz is not None:
                if cutoff_date.tzinfo is None:
                    cutoff_date = cutoff_date.replace(tzinfo=historical_data.index.tz)
            elif cutoff_date.tzinfo is not None:
                cutoff_date = cutoff_date.replace(tzinfo=None)
            
            recent_data = historical_data[historical_data.index >= cutoff_date]
            
            if len(recent_data) < 10:
                logger.warning(f"Insufficient recent data: {len(recent_data)} records")
                return {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'total_predictions': 0,
                    'buy_signals': 0,
                    'sell_signals': 0,
                    'analysis_period_days': days,
                    'error': 'Insufficient data'
                }
            
            # Add features to recent data
            recent_data_with_features = self.feature_engineer.add_technical_indicators(recent_data.copy())
            
            if recent_data_with_features.empty:
                logger.warning("Failed to add technical indicators")
                return None
            
            # Prepare features for prediction (not training)
            feature_columns = self.model_trainer.feature_columns
            if not feature_columns:
                logger.warning("No feature columns available")
                return None
            
            # Ensure we have the target column
            if 'Target' not in recent_data_with_features.columns:
                recent_data_with_features = self.feature_engineer.add_target_column(recent_data_with_features)
            
            # Clean data and align features
            available_features = [col for col in feature_columns if col in recent_data_with_features.columns]
            if len(available_features) < len(feature_columns) * 0.8:  # Need at least 80% of features
                logger.warning(f"Missing too many features: {len(available_features)}/{len(feature_columns)}")
                return None
            
            # Prepare data for prediction
            df_clean = recent_data_with_features[available_features + ['Target']].dropna()
            
            if len(df_clean) < 5:
                logger.warning(f"Insufficient clean data: {len(df_clean)} records")
                return None
            
            X = df_clean[available_features]
            y_true = df_clean['Target']
            
            # Make predictions
            y_pred = self.model_trainer.model.predict(X)
            y_prob = self.model_trainer.model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            accuracy = (y_pred == y_true).mean()
            
            # Calculate precision and recall for each class
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            result = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'total_predictions': int(len(y_pred)),
                'buy_signals': int((y_pred == 1).sum()),
                'sell_signals': int((y_pred == 0).sum()),
                'analysis_period_days': days,
                'features_used': len(available_features),
                'data_points': len(df_clean)
            }
            
            logger.info(f"Accuracy analysis complete: {accuracy:.3f} accuracy over {len(df_clean)} predictions")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing prediction accuracy: {e}")
            import traceback
            traceback.print_exc()
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'total_predictions': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'analysis_period_days': days,
                'error': str(e)
            }
    
    def get_market_sentiment(self):
        """
        Get overall market sentiment based on recent predictions
        """
        try:
            history = self.get_prediction_history(days=3)
            
            if not history:
                return "NEUTRAL"
            
            # Calculate average probability over recent predictions
            recent_probs = [h['probability'] for h in history[-10:]]  # Last 10 predictions
            avg_prob = np.mean(recent_probs)
            
            if avg_prob >= 0.65:
                return "BULLISH"
            elif avg_prob <= 0.35:
                return "BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
            return "NEUTRAL"
    
    def get_comprehensive_trading_recommendation(self):
        """
        Get comprehensive trading recommendation with advanced strategy analysis,
        risk management, and position sizing
        """
        try:
            logger.info("Getting comprehensive trading recommendation...")
            
            # Get live prediction first
            basic_prediction = self.get_live_prediction()
            
            if basic_prediction['data_quality'] == 'unavailable':
                return self._get_fallback_comprehensive_recommendation()
            
            # Get recent data for price action analysis
            live_data = self.data_loader.fetch_live_data(days_back=30)
            
            if live_data.empty:
                return self._get_fallback_comprehensive_recommendation()
            
            # Analyze price action using the full dataset
            price_action_analysis = self.strategy_analyzer.analyze_price_action(live_data)
            
            # Create ML prediction dict for strategy analyzer
            ml_prediction = {
                'prediction': basic_prediction['prediction'],
                'probability': basic_prediction['probability'],
                'signal': basic_prediction['signal']
            }
            
            # Generate comprehensive recommendation
            comprehensive_recommendation = self.strategy_analyzer.generate_trading_recommendation(
                ml_prediction=ml_prediction,
                technical_indicators=basic_prediction['indicators'],
                current_price=basic_prediction['current_price']
            )
            
            # Add price action analysis to the recommendation
            comprehensive_recommendation['price_action_analysis'] = price_action_analysis
            comprehensive_recommendation['basic_prediction'] = basic_prediction
            
            # Add market context
            comprehensive_recommendation['market_context'] = {
                'timestamp': basic_prediction['timestamp'],
                'data_quality': basic_prediction['data_quality'],
                'market_sentiment': self.get_market_sentiment(),
                'volatility_regime': self._assess_volatility_regime(live_data)
            }
            
            logger.info(f"Comprehensive recommendation: {comprehensive_recommendation['direction']} "
                       f"(Grade: {comprehensive_recommendation['setup_grade']})")
            
            return comprehensive_recommendation
            
        except Exception as e:
            logger.error(f"Error getting comprehensive recommendation: {e}")
            return self._get_fallback_comprehensive_recommendation()
    
    def _get_fallback_comprehensive_recommendation(self):
        """
        Fallback comprehensive recommendation when main method fails
        """
        try:
            basic_prediction = self.get_live_prediction()
            current_price = basic_prediction.get('current_price', 2000.0)
            
            # Create minimal comprehensive recommendation
            fallback_rec = {
                'setup_grade': 'D',
                'setup_quality': 'Poor - Using Fallback Data',
                'direction': 'NEUTRAL',
                'entry_price': current_price,
                'stop_loss': current_price * 0.99,
                'target_price': current_price * 1.01,
                'setup_reasoning': 'Fallback recommendation due to data unavailability',
                'technical_summary': 'Limited technical analysis available',
                'price_action_summary': 'Price action analysis unavailable',
                'risk_management': {
                    'max_risk_usd': 20.0,
                    'account_risk_pct': 2.0,
                    'potential_profit_usd': 20.0,
                    'risk_reward_ratio': 1.0
                },
                'position_size': {
                    'quantity': 0,
                    'leveraged_position_usd': 0.0,
                    'leverage_used': 1
                },
                'trade_validity': {
                    'is_valid': False,
                    'recommendation': 'NO TRADE',
                    'confidence_level': 'Low',
                    'issues': ['Data unavailable', 'Poor setup quality']
                },
                'price_action_analysis': {
                    'bullish_candle': False,
                    'bearish_candle': False,
                    'doji': False,
                    'hammer': False,
                    'shooting_star': False,
                    'bullish_engulfing': False,
                    'bearish_engulfing': False,
                    'price_action_strength': 0.0,
                    'current_candle_type': 'Unknown'
                },
                'market_context': {
                    'market_sentiment': 'NEUTRAL',
                    'volatility_regime': 'Unknown',
                    'data_quality': 'unavailable'
                }
            }
            
            return fallback_rec
            
        except Exception as e:
            logging.error(f"Error in fallback recommendation: {e}")
            # Return absolute minimal fallback
            return {
                'setup_grade': 'D',
                'setup_quality': 'System Error',
                'direction': 'NEUTRAL',
                'entry_price': 2000.0,
                'stop_loss': 1980.0,
                'target_price': 2020.0,
                'setup_reasoning': 'System error - no trading recommended',
                'technical_summary': 'Analysis unavailable',
                'price_action_summary': 'Analysis unavailable',
                'risk_management': {'max_risk_usd': 0, 'account_risk_pct': 0, 'potential_profit_usd': 0, 'risk_reward_ratio': 0},
                'position_size': {'quantity': 0, 'leveraged_position_usd': 0, 'leverage_used': 1},
                'trade_validity': {'is_valid': False, 'recommendation': 'SYSTEM ERROR', 'confidence_level': 'None', 'issues': ['System error']},
                'price_action_analysis': {},
                'market_context': {'market_sentiment': 'NEUTRAL', 'volatility_regime': 'Unknown', 'data_quality': 'unavailable'}
            }
    
    def get_macro_analysis(self):
        """
        Get comprehensive macroeconomic analysis affecting gold prices
        """
        try:
            logger.info("Getting macroeconomic analysis...")
            
            # Get recent gold data for correlation analysis
            gold_data = self.data_loader.fetch_live_data(days_back=30)
            
            if not gold_data.empty:
                gold_prices = gold_data['Close']
                macro_summary = self.macro_analyzer.update_all_factors(gold_prices)
            else:
                # Fallback without correlation analysis
                macro_summary = self.macro_analyzer.update_all_factors()
            
            # Add timestamp
            macro_summary['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Macro analysis complete. Overall sentiment: {macro_summary['overall_sentiment']}")
            return macro_summary
            
        except Exception as e:
            logger.error(f"Error getting macro analysis: {e}")
            return {
                'overall_sentiment': 'Unknown',
                'bullish_score': 0,
                'bearish_score': 0,
                'top_factors': [],
                'bullish_factors': [],
                'bearish_factors': [],
                'summary': 'Macro analysis unavailable due to data issues.',
                'last_updated': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def get_enhanced_comprehensive_recommendation(self):
        """
        Get enhanced comprehensive recommendation including macro factors
        """
        try:
            logger.info("Getting enhanced comprehensive recommendation with macro factors...")
            
            # Get base comprehensive recommendation
            base_recommendation = self.get_comprehensive_trading_recommendation()
            
            # Get macro analysis
            macro_analysis = self.get_macro_analysis()
            
            # Enhance the recommendation with macro insights
            base_recommendation['macro_analysis'] = macro_analysis
            
            # Adjust setup grade based on macro sentiment alignment
            self._adjust_setup_with_macro(base_recommendation, macro_analysis)
            
            logger.info("Enhanced comprehensive recommendation complete")
            return base_recommendation
            
        except Exception as e:
            logger.error(f"Error getting enhanced recommendation: {e}")
            return self.get_comprehensive_trading_recommendation()
    
    def _adjust_setup_with_macro(self, recommendation, macro_analysis):
        """
        Adjust trading setup based on macro factor alignment
        """
        try:
            trade_direction = recommendation.get('direction', 'NEUTRAL')
            macro_sentiment = macro_analysis.get('overall_sentiment', 'Mixed')
            
            # Check alignment between trade direction and macro sentiment
            alignment_score = 0
            
            if trade_direction == 'LONG' and macro_sentiment == 'Bullish':
                alignment_score = 1  # Perfect alignment
            elif trade_direction == 'SHORT' and macro_sentiment == 'Bearish':
                alignment_score = 1  # Perfect alignment
            elif macro_sentiment == 'Mixed':
                alignment_score = 0.5  # Neutral
            else:
                alignment_score = -0.5  # Conflicting signals
            
            # Adjust setup quality based on macro alignment
            current_grade = recommendation.get('setup_grade', 'D')
            grade_values = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
            
            if alignment_score > 0.5:
                # Upgrade grade if macro supports the trade
                if current_grade in ['C', 'D']:
                    recommendation['setup_grade'] = 'B'
                    recommendation['setup_quality'] = 'Good - Macro Support'
                elif current_grade == 'B':
                    recommendation['setup_grade'] = 'A'
                    recommendation['setup_quality'] = 'Excellent - Strong Macro Support'
            elif alignment_score < -0.3:
                # Downgrade if macro conflicts
                if current_grade in ['A', 'B']:
                    recommendation['setup_grade'] = 'C'
                    recommendation['setup_quality'] = 'Fair - Macro Headwinds'
                elif current_grade == 'C':
                    recommendation['setup_grade'] = 'D'
                    recommendation['setup_quality'] = 'Poor - Macro Conflict'
            
            # Add macro reasoning to setup explanation
            macro_reasoning = f" Macro environment is {macro_sentiment.lower()} for gold."
            if 'setup_reasoning' in recommendation:
                recommendation['setup_reasoning'] += macro_reasoning
            
        except Exception as e:
            logger.error(f"Error adjusting setup with macro: {e}")
    
    def _get_fallback_comprehensive_recommendation(self):
        """
        Fallback comprehensive recommendation when live data is unavailable
        """
        return {
            'direction': 'NEUTRAL',
            'signal_strength': 'NEUTRAL',
            'setup_grade': 'D',
            'setup_quality': 'Poor',
            'setup_reasoning': 'Insufficient data for analysis',
            'current_price': None,
            'entry_price': None,
            'stop_loss': None,
            'target_price': None,
            'position_size': {
                'setup_grade': 'D',
                'risk_amount_usd': 0,
                'quantity': 0,
                'max_loss_usd': 0
            },
            'risk_management': {
                'max_risk_usd': 0,
                'potential_profit_usd': 0,
                'risk_reward_ratio': 0,
                'account_risk_pct': 0
            },
            'confidence_score': 0.5,
            'technical_summary': 'Data unavailable',
            'price_action_summary': 'Data unavailable',
            'trade_validity': {
                'is_valid': False,
                'issues': ['No live data available'],
                'recommendation': 'SKIP TRADE',
                'confidence_level': 'LOW'
            },
            'price_action_analysis': {},
            'market_context': {
                'timestamp': datetime.now(),
                'data_quality': 'unavailable',
                'market_sentiment': 'NEUTRAL',
                'volatility_regime': 'Unknown'
            }
        }
    
    def _assess_volatility_regime(self, df):
        """
        Assess current volatility regime (Low/Medium/High)
        """
        try:
            # Calculate recent volatility
            df['returns'] = df['Close'].pct_change()
            recent_vol = df['returns'].tail(20).std() * np.sqrt(24 * 365)  # Annualized volatility
            
            # Historical volatility percentiles
            historical_vol = df['returns'].std() * np.sqrt(24 * 365)
            
            if recent_vol > historical_vol * 1.5:
                return 'High'
            elif recent_vol > historical_vol * 1.2:
                return 'Medium'
            else:
                return 'Low'
                
        except Exception:
            return 'Unknown'

def main():
    """Test the predictor"""
    predictor = GoldPredictor()
    
    print("Getting live prediction...")
    result = predictor.get_live_prediction()
    
    print(f"Timestamp: {result['timestamp']}")
    print(f"Current Price: ${result['current_price']:.2f}" if result['current_price'] else "Price unavailable")
    print(f"Signal: {result['signal']}")
    print(f"Probability: {result['probability']:.3f}")
    print(f"Confidence: {result['confidence']}")
    print(f"Data Quality: {result['data_quality']}")
    
    print("\nTechnical Indicators:")
    for indicator, value in result['indicators'].items():
        print(f"{indicator}: {value:.4f}")
    
    print(f"\nMarket Sentiment: {predictor.get_market_sentiment()}")
    
    # Test accuracy analysis
    print("\nRecent Accuracy Analysis:")
    accuracy = predictor.analyze_prediction_accuracy(days=30)
    if accuracy:
        print(f"Accuracy: {accuracy['accuracy']:.3f}")
        print(f"Precision: {accuracy['precision']:.3f}")
        print(f"Recall: {accuracy['recall']:.3f}")

if __name__ == "__main__":
    main()
