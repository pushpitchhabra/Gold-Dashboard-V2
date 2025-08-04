"""
Advanced Strategy Analyzer for Gold Trading Dashboard
Includes price action analysis, setup grading, risk management, and position sizing
"""

import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime
from typing import Dict, Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedStrategyAnalyzer:
    def __init__(self, config_path="config.yaml"):
        """Initialize advanced strategy analyzer"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Risk management parameters
        self.account_size = 1000  # $1000 account
        self.max_risk_per_trade = 20  # $20 max risk per trade
        self.leverage = 500  # 500x leverage
        self.risk_percentage = self.max_risk_per_trade / self.account_size  # 2% risk per trade
        
        # Setup quality grades
        self.setup_grades = {
            'A': {'quality': 'Excellent', 'risk_multiplier': 1.0, 'confidence_threshold': 0.8, 'risk_reward_ratio': 2.5},
            'B': {'quality': 'Good', 'risk_multiplier': 0.8, 'confidence_threshold': 0.7, 'risk_reward_ratio': 2.0},
            'C': {'quality': 'Average', 'risk_multiplier': 0.6, 'confidence_threshold': 0.6, 'risk_reward_ratio': 1.5},
            'D': {'quality': 'Poor', 'risk_multiplier': 0.4, 'confidence_threshold': 0.5, 'risk_reward_ratio': 1.0}
        }
    
    def analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """
        Analyze price action patterns including bullish/bearish candle conditions
        """
        logger.info("Analyzing price action patterns...")
        
        df_analysis = df.copy()
        
        # Calculate prior highs and lows
        df_analysis['Prior_High'] = df_analysis['High'].shift(1)
        df_analysis['Prior_Low'] = df_analysis['Low'].shift(1)
        df_analysis['Prior_Close'] = df_analysis['Close'].shift(1)
        
        # Bullish candle conditions
        # High above prior high, Low above prior low, Close above prior low
        df_analysis['Bullish_Candle'] = (
            (df_analysis['High'] > df_analysis['Prior_High']) &
            (df_analysis['Low'] > df_analysis['Prior_Low']) &
            (df_analysis['Close'] > df_analysis['Prior_Low'])
        )
        
        # Bearish candle conditions  
        # High below prior low, Low below prior low, Close below prior low
        df_analysis['Bearish_Candle'] = (
            (df_analysis['High'] < df_analysis['Prior_Low']) &
            (df_analysis['Low'] < df_analysis['Prior_Low']) &
            (df_analysis['Close'] < df_analysis['Prior_Low'])
        )
        
        # Additional price action patterns
        df_analysis['Doji'] = abs(df_analysis['Close'] - df_analysis['Open']) <= (df_analysis['High'] - df_analysis['Low']) * 0.1
        df_analysis['Hammer'] = (
            (df_analysis['Close'] > df_analysis['Open']) &
            ((df_analysis['Low'] - df_analysis['Open']) >= 2 * (df_analysis['Close'] - df_analysis['Open'])) &
            ((df_analysis['High'] - df_analysis['Close']) <= (df_analysis['Close'] - df_analysis['Open']) * 0.1)
        )
        df_analysis['Shooting_Star'] = (
            (df_analysis['Open'] > df_analysis['Close']) &
            ((df_analysis['High'] - df_analysis['Open']) >= 2 * (df_analysis['Open'] - df_analysis['Close'])) &
            ((df_analysis['Close'] - df_analysis['Low']) <= (df_analysis['Open'] - df_analysis['Close']) * 0.1)
        )
        
        # Engulfing patterns
        df_analysis['Bullish_Engulfing'] = (
            (df_analysis['Close'] > df_analysis['Open']) &
            (df_analysis['Prior_Close'] < df_analysis['Prior_Close'].shift(1)) &
            (df_analysis['Open'] < df_analysis['Prior_Close']) &
            (df_analysis['Close'] > df_analysis['Prior_High'])
        )
        
        df_analysis['Bearish_Engulfing'] = (
            (df_analysis['Close'] < df_analysis['Open']) &
            (df_analysis['Prior_Close'] > df_analysis['Prior_Close'].shift(1)) &
            (df_analysis['Open'] > df_analysis['Prior_Close']) &
            (df_analysis['Close'] < df_analysis['Prior_Low'])
        )
        
        # Get latest price action signals
        latest = df_analysis.iloc[-1]
        
        price_action_summary = {
            'bullish_candle': bool(latest['Bullish_Candle']),
            'bearish_candle': bool(latest['Bearish_Candle']),
            'doji': bool(latest['Doji']),
            'hammer': bool(latest['Hammer']),
            'shooting_star': bool(latest['Shooting_Star']),
            'bullish_engulfing': bool(latest['Bullish_Engulfing']),
            'bearish_engulfing': bool(latest['Bearish_Engulfing']),
            'current_candle_type': self._classify_candle_type(latest),
            'price_action_strength': self._calculate_price_action_strength(df_analysis.tail(5))
        }
        
        return price_action_summary
    
    def _classify_candle_type(self, candle_data) -> str:
        """Classify the current candle type"""
        if candle_data['Bullish_Candle']:
            return "Strong Bullish"
        elif candle_data['Bearish_Candle']:
            return "Strong Bearish"
        elif candle_data['Doji']:
            return "Doji (Indecision)"
        elif candle_data['Hammer']:
            return "Hammer (Bullish Reversal)"
        elif candle_data['Shooting_Star']:
            return "Shooting Star (Bearish Reversal)"
        elif candle_data['Bullish_Engulfing']:
            return "Bullish Engulfing"
        elif candle_data['Bearish_Engulfing']:
            return "Bearish Engulfing"
        elif candle_data['Close'] > candle_data['Open']:
            return "Bullish"
        elif candle_data['Close'] < candle_data['Open']:
            return "Bearish"
        else:
            return "Neutral"
    
    def _calculate_price_action_strength(self, recent_candles) -> float:
        """Calculate overall price action strength (0-1)"""
        bullish_signals = recent_candles['Bullish_Candle'].sum()
        bearish_signals = recent_candles['Bearish_Candle'].sum()
        total_signals = len(recent_candles)
        
        if bullish_signals > bearish_signals:
            return (bullish_signals / total_signals) * 0.8 + 0.2
        elif bearish_signals > bullish_signals:
            return (bearish_signals / total_signals) * -0.8 - 0.2
        else:
            return 0.0
    
    def grade_setup_quality(self, technical_indicators: Dict, price_action: Dict, ml_probability: float) -> Dict:
        """
        Grade trading setup quality from A (excellent) to D (poor)
        """
        logger.info("Grading setup quality...")
        
        score = 0
        max_score = 100
        
        # Technical indicator scoring (40 points)
        rsi = technical_indicators.get('RSI', 50)
        if 30 <= rsi <= 70:  # Good RSI range
            score += 15
        elif rsi < 20 or rsi > 80:  # Extreme levels
            score += 10
        
        macd = technical_indicators.get('MACD', 0)
        macd_signal = technical_indicators.get('MACD_Signal', 0)
        if (macd > macd_signal and macd > 0) or (macd < macd_signal and macd < 0):
            score += 15  # MACD alignment
        
        bb_position = technical_indicators.get('BB_Position', 0.5)
        if 0.2 <= bb_position <= 0.8:  # Not at extreme bands
            score += 10
        
        # Price action scoring (30 points)
        if price_action.get('bullish_candle') or price_action.get('bearish_candle'):
            score += 15  # Strong directional candle
        
        if price_action.get('hammer') or price_action.get('shooting_star'):
            score += 10  # Reversal patterns
        
        if abs(price_action.get('price_action_strength', 0)) > 0.5:
            score += 5  # Strong price action
        
        # ML model confidence scoring (30 points)
        if ml_probability >= 0.8 or ml_probability <= 0.2:
            score += 30  # Very high confidence
        elif ml_probability >= 0.7 or ml_probability <= 0.3:
            score += 20  # High confidence
        elif ml_probability >= 0.6 or ml_probability <= 0.4:
            score += 10  # Medium confidence
        
        # Determine grade
        percentage = (score / max_score) * 100
        
        if percentage >= 80:
            grade = 'A'
        elif percentage >= 65:
            grade = 'B'
        elif percentage >= 50:
            grade = 'C'
        else:
            grade = 'D'
        
        quality_description = self.setup_grades[grade]['quality']
        confidence_multiplier = 1.0
        risk_multiplier = self.setup_grades[grade]['risk_multiplier']
        
        return {
            'grade': grade,
            'quality': quality_description,
            'confidence_multiplier': confidence_multiplier,
            'risk_multiplier': risk_multiplier,
            'risk_reward_ratio': self.setup_grades[grade]['risk_reward_ratio'],
            'reasoning': self._generate_setup_reasoning(technical_indicators, price_action, ml_probability, score)
        }
    
    def _generate_setup_reasoning(self, tech_indicators: Dict, price_action: Dict, ml_prob: float, score: int) -> str:
        """Generate reasoning for setup grade"""
        reasons = []
        
        # Technical reasons
        rsi = tech_indicators.get('RSI', 50)
        if rsi < 30:
            reasons.append("RSI oversold (potential reversal)")
        elif rsi > 70:
            reasons.append("RSI overbought (potential reversal)")
        
        # Price action reasons
        if price_action.get('bullish_candle'):
            reasons.append("Strong bullish price action")
        elif price_action.get('bearish_candle'):
            reasons.append("Strong bearish price action")
        
        # ML confidence
        if ml_prob >= 0.8:
            reasons.append("Very high ML confidence (bullish)")
        elif ml_prob <= 0.2:
            reasons.append("Very high ML confidence (bearish)")
        
        if not reasons:
            reasons.append("Mixed signals, moderate setup quality")
        
        return "; ".join(reasons)
    
    def calculate_position_size(self, current_price: float, stop_loss_price: float, setup_grade: str) -> Dict:
        """
        Calculate position size based on risk management rules
        """
        logger.info("Calculating position size...")
        
        # Get risk multiplier based on setup grade
        risk_multiplier = self.setup_grades[setup_grade]['risk_multiplier']
        adjusted_risk = self.max_risk_per_trade * risk_multiplier
        
        # Calculate stop loss distance
        stop_loss_distance = abs(current_price - stop_loss_price)
        
        if stop_loss_distance == 0:
            # Default 1% stop loss if not provided
            stop_loss_distance = current_price * 0.01
            stop_loss_price = current_price - stop_loss_distance if current_price > stop_loss_price else current_price + stop_loss_distance
        
        # Calculate position size
        # Risk per unit = stop_loss_distance
        # Position size = Risk Amount / Risk per unit
        position_size_usd = adjusted_risk / stop_loss_distance
        
        # With leverage, we can control larger positions
        leveraged_position_size = position_size_usd * self.leverage
        
        # Calculate quantity (assuming gold is priced per ounce)
        quantity = leveraged_position_size / current_price
        
        # Risk-reward calculations
        risk_reward_ratio = self._calculate_risk_reward_ratio(current_price, stop_loss_price, setup_grade)
        
        position_info = {
            'setup_grade': setup_grade,
            'risk_amount_usd': adjusted_risk,
            'position_size_usd': position_size_usd,
            'leveraged_position_usd': leveraged_position_size,
            'quantity': round(quantity, 4),
            'stop_loss_price': round(stop_loss_price, 2),
            'stop_loss_distance': round(stop_loss_distance, 2),
            'risk_reward_ratio': risk_reward_ratio,
            'max_loss_usd': adjusted_risk,
            'potential_profit_usd': adjusted_risk * risk_reward_ratio,
            'leverage_used': self.leverage,
            'account_risk_percentage': (adjusted_risk / self.account_size) * 100
        }
        
        return position_info
    
    def _calculate_risk_reward_ratio(self, current_price: float, stop_loss_price: float, setup_grade: str) -> float:
        """Calculate expected risk-reward ratio based on setup quality"""
        base_ratios = {
            'A': 3.0,  # Excellent setups target 3:1
            'B': 2.5,  # Good setups target 2.5:1
            'C': 2.0,  # Average setups target 2:1
            'D': 1.5   # Poor setups target 1.5:1
        }
        
        return base_ratios.get(setup_grade, 2.0)
    
    def generate_trading_recommendation(self, ml_prediction: Dict, technical_indicators: Dict, 
                                      current_price: float) -> Dict:
        """
        Generate comprehensive trading recommendation
        """
        logger.info("Generating trading recommendation...")
        
        # Analyze price action
        # Note: This would need the full DataFrame, but for demo we'll simulate
        price_action = {
            'bullish_candle': ml_prediction['prediction'] == 1,
            'bearish_candle': ml_prediction['prediction'] == 0,
            'price_action_strength': (ml_prediction['probability'] - 0.5) * 2
        }
        
        # Grade setup quality
        setup_quality = self.grade_setup_quality(technical_indicators, price_action, ml_prediction['probability'])
        
        # Calculate stop loss (using ATR-based method)
        atr = technical_indicators.get('ATR', current_price * 0.01)
        
        if ml_prediction['prediction'] == 1:  # Bullish
            stop_loss_price = current_price - (atr * 2)
            target_price = current_price + (atr * setup_quality['risk_reward_ratio'] * 2)
            direction = "LONG"
        else:  # Bearish
            stop_loss_price = current_price + (atr * 2)
            target_price = current_price - (atr * setup_quality['risk_reward_ratio'] * 2)
            direction = "SHORT"
        
        # Calculate position size
        position_info = self.calculate_position_size(current_price, stop_loss_price, setup_quality['grade'])
        
        # Generate recommendation
        recommendation = {
            'direction': direction,
            'signal_strength': ml_prediction['signal'],
            'setup_grade': setup_quality['grade'],
            'setup_quality': setup_quality['quality'],
            'setup_reasoning': setup_quality['reasoning'],
            'current_price': current_price,
            'entry_price': current_price,
            'stop_loss': position_info['stop_loss_price'],
            'target_price': round(target_price, 2),
            'position_size': position_info,
            'risk_management': {
                'max_risk_usd': position_info['max_loss_usd'],
                'potential_profit_usd': position_info['potential_profit_usd'],
                'risk_reward_ratio': position_info['risk_reward_ratio'],
                'account_risk_pct': position_info['account_risk_percentage']
            },
            'confidence_score': ml_prediction['probability'],
            'technical_summary': self._generate_technical_summary(technical_indicators),
            'price_action_summary': self._generate_price_action_summary(price_action),
            'trade_validity': self._validate_trade_setup(setup_quality, ml_prediction['probability'])
        }
        
        return recommendation
    
    def _generate_technical_summary(self, indicators: Dict) -> str:
        """Generate technical analysis summary"""
        summaries = []
        
        rsi = indicators.get('RSI', 50)
        if rsi < 30:
            summaries.append("RSI oversold")
        elif rsi > 70:
            summaries.append("RSI overbought")
        else:
            summaries.append("RSI neutral")
        
        macd = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_Signal', 0)
        if macd > macd_signal:
            summaries.append("MACD bullish")
        else:
            summaries.append("MACD bearish")
        
        bb_pos = indicators.get('BB_Position', 0.5)
        if bb_pos > 0.8:
            summaries.append("Near upper Bollinger Band")
        elif bb_pos < 0.2:
            summaries.append("Near lower Bollinger Band")
        
        return "; ".join(summaries)
    
    def _generate_price_action_summary(self, price_action: Dict) -> str:
        """Generate price action summary"""
        if price_action.get('bullish_candle'):
            return "Strong bullish price action with momentum"
        elif price_action.get('bearish_candle'):
            return "Strong bearish price action with momentum"
        else:
            return "Mixed price action signals"
    
    def _validate_trade_setup(self, setup_quality: Dict, ml_confidence: float) -> Dict:
        """Validate if trade setup meets minimum criteria"""
        is_valid = True
        issues = []
        
        # Check setup grade
        if setup_quality['grade'] == 'D':
            is_valid = False
            issues.append("Poor setup quality (Grade D)")
        
        # Check ML confidence
        if 0.4 <= ml_confidence <= 0.6:
            is_valid = False
            issues.append("Low ML confidence (neutral zone)")
        
        # Check risk-reward
        if setup_quality.get('risk_reward_ratio', 0) < 1.5:
            issues.append("Low risk-reward ratio")
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'recommendation': "EXECUTE TRADE" if is_valid else "SKIP TRADE",
            'confidence_level': "HIGH" if is_valid and not issues else "LOW"
        }
    
    def analyze_current_setup(self, df: pd.DataFrame) -> Dict:
        """
        Analyze current trading setup - main method used by dashboard and diagnostic scripts
        """
        try:
            logger.info("Analyzing current trading setup...")
            
            if df.empty:
                logger.warning("No data provided for setup analysis")
                return self._get_fallback_setup_analysis()
            
            # Get current price
            current_price = float(df['Close'].iloc[-1])
            
            # Calculate technical indicators for analysis
            technical_indicators = self._calculate_basic_indicators(df)
            
            # Analyze price action
            price_action = self.analyze_price_action(df)
            
            # Create mock ML prediction for setup analysis
            # In real implementation, this would come from the predictor
            mock_ml_prediction = {
                'prediction': 1 if df['Close'].iloc[-1] > df['Close'].iloc[-2] else 0,
                'probability': 0.65,  # Default moderate confidence
                'signal': 'BUY' if df['Close'].iloc[-1] > df['Close'].iloc[-2] else 'SELL'
            }
            
            # Generate comprehensive recommendation
            recommendation = self.generate_trading_recommendation(
                mock_ml_prediction, technical_indicators, current_price
            )
            
            logger.info(f"Setup analysis complete: {recommendation['direction']} setup with grade {recommendation['setup_grade']}")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error in setup analysis: {e}")
            return self._get_fallback_setup_analysis()
    
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calculate basic technical indicators for setup analysis
        """
        try:
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9).mean()
            
            # Calculate Bollinger Bands
            bb_period = 20
            bb_std = 2
            sma = df['Close'].rolling(window=bb_period).mean()
            std = df['Close'].rolling(window=bb_period).std()
            bb_upper = sma + (std * bb_std)
            bb_lower = sma - (std * bb_std)
            bb_position = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            
            # Calculate ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            return {
                'RSI': float(rsi.iloc[-1]) if not rsi.empty else 50.0,
                'MACD': float(macd.iloc[-1]) if not macd.empty else 0.0,
                'MACD_Signal': float(macd_signal.iloc[-1]) if not macd_signal.empty else 0.0,
                'BB_Position': float(bb_position.iloc[-1]) if not bb_position.empty else 0.5,
                'ATR': float(atr.iloc[-1]) if not atr.empty else 10.0,
                'EMA_Signal': 1.0 if df['Close'].iloc[-1] > df['Close'].ewm(span=20).mean().iloc[-1] else -1.0
            }
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {
                'RSI': 50.0,
                'MACD': 0.0,
                'MACD_Signal': 0.0,
                'BB_Position': 0.5,
                'ATR': 10.0,
                'EMA_Signal': 0.0
            }
    
    def _get_fallback_setup_analysis(self) -> Dict:
        """
        Provide fallback setup analysis when main analysis fails
        """
        return {
            'direction': 'NEUTRAL',
            'signal_strength': 'WEAK',
            'setup_grade': 'D',
            'setup_quality': 'Poor - Insufficient Data',
            'setup_reasoning': 'Unable to analyze current market setup due to data issues',
            'current_price': 2000.0,
            'entry_price': 2000.0,
            'stop_loss': 1980.0,
            'target_price': 2020.0,
            'position_size': {
                'setup_grade': 'D',
                'risk_amount_usd': 0,
                'quantity': 0,
                'max_loss_usd': 0
            },
            'risk_management': {
                'max_risk_usd': 0,
                'potential_profit_usd': 0,
                'risk_reward_ratio': 1.0,
                'account_risk_pct': 0
            },
            'confidence_score': 0.5,
            'technical_summary': 'Data unavailable for technical analysis',
            'price_action_summary': 'Data unavailable for price action analysis',
            'trade_validity': {
                'is_valid': False,
                'issues': ['Insufficient data for analysis'],
                'recommendation': 'SKIP TRADE',
                'confidence_level': 'LOW'
            },
            'market_context': {
                'timestamp': datetime.now(),
                'data_quality': 'fallback',
                'market_sentiment': 'NEUTRAL'
            }
        }

def main():
    """Test the advanced strategy analyzer"""
    analyzer = AdvancedStrategyAnalyzer()
    
    # Mock data for testing
    mock_ml_prediction = {
        'prediction': 1,
        'probability': 0.75,
        'signal': 'BUY'
    }
    
    mock_technical_indicators = {
        'RSI': 35,
        'MACD': 0.5,
        'MACD_Signal': 0.3,
        'BB_Position': 0.3,
        'ATR': 5.2,
        'EMA_Signal': 1.2
    }
    
    current_price = 2650.0
    
    # Test recommendation generation
    recommendation = analyzer.generate_trading_recommendation(
        mock_ml_prediction, mock_technical_indicators, current_price
    )
    
    print("=== TRADING RECOMMENDATION ===")
    print(f"Direction: {recommendation['direction']}")
    print(f"Setup Grade: {recommendation['setup_grade']} ({recommendation['setup_quality']})")
    print(f"Entry Price: ${recommendation['entry_price']}")
    print(f"Stop Loss: ${recommendation['stop_loss']}")
    print(f"Target: ${recommendation['target_price']}")
    print(f"Position Size: {recommendation['position_size']['quantity']} units")
    print(f"Risk: ${recommendation['risk_management']['max_risk_usd']}")
    print(f"Potential Profit: ${recommendation['risk_management']['potential_profit_usd']}")
    print(f"Risk/Reward: 1:{recommendation['risk_management']['risk_reward_ratio']}")
    print(f"Trade Valid: {recommendation['trade_validity']['is_valid']}")

if __name__ == "__main__":
    main()
