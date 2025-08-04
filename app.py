"""
Streamlit Dashboard for AI-Powered Gold Trading
Main application interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import os
from datetime import datetime, timedelta
import time
import logging
import pytz
from zoneinfo import ZoneInfo

# Configure page
st.set_page_config(
    page_title="AI Gold Trading Dashboard",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timezone utility functions
def get_current_time_with_tz():
    """Get current time in user's local timezone"""
    try:
        # Get current local time
        local_tz = datetime.now().astimezone().tzinfo
        current_time = datetime.now(local_tz)
        return current_time, local_tz
    except Exception:
        # Fallback to UTC if local timezone detection fails
        utc_tz = pytz.UTC
        current_time = datetime.now(utc_tz)
        return current_time, utc_tz

def format_time_with_source_tz(timestamp, source_tz_name="UTC", target_tz=None):
    """Format timestamp showing both local time and source timezone"""
    try:
        if target_tz is None:
            target_tz = datetime.now().astimezone().tzinfo
        
        if isinstance(timestamp, str):
            # Parse string timestamp
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Convert to target timezone if needed
        if timestamp.tzinfo is None:
            # Assume UTC if no timezone info
            timestamp = pytz.UTC.localize(timestamp)
        
        local_time = timestamp.astimezone(target_tz)
        
        local_str = local_time.strftime('%Y-%m-%d %H:%M:%S')
        tz_name = str(target_tz).split('/')[-1] if hasattr(target_tz, 'zone') else str(target_tz)
        
        return f"{local_str} (Local) | Source: {source_tz_name}"
    except Exception as e:
        return f"Time parsing error: {str(e)}"

def get_data_source_timezones():
    """Return timezone information for different data sources"""
    return {
        'FRED': 'US/Eastern (Federal Reserve)',
        'Yahoo Finance': 'US/Eastern (NYSE)',
        'Market Data': 'US/Eastern (NYSE)',
        'Economic Data': 'US/Eastern (Federal Reserve)',
        'Gold Prices': 'UTC (Global Markets)',
        'System': str(datetime.now().astimezone().tzinfo)
    }

# Import our modules
try:
    from data_loader import GoldDataLoader
    from feature_engineer import GoldFeatureEngineer
    from model_trainer import GoldModelTrainer
    from predictor import GoldPredictor
    from updater import GoldModelUpdater
    from macro_factors import MacroFactorsAnalyzer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Load configuration
@st.cache_data
def load_config():
    """Load configuration file"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return {}

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all components"""
    try:
        data_loader = GoldDataLoader()
        feature_engineer = GoldFeatureEngineer()
        model_trainer = GoldModelTrainer()
        predictor = GoldPredictor()
        updater = GoldModelUpdater()
        macro_analyzer = MacroFactorsAnalyzer()
        
        return data_loader, feature_engineer, model_trainer, predictor, updater, macro_analyzer
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None, None, None, None, None

def create_candlestick_chart(df, title="Gold Price Chart"):
    """Create candlestick chart with technical indicators"""
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Indicators', 'RSI', 'MACD'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Gold Price"
        ),
        row=1, col=1
    )
    
    # Add technical indicators if available
    if 'EMA_Fast' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_Fast'],
                name="EMA Fast",
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
    
    if 'EMA_Slow' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['EMA_Slow'],
                name="EMA Slow",
                line=dict(color='red', width=1)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if all(col in df.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                name="BB Upper",
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                name="BB Lower",
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name="RSI",
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
    
    # MACD
    if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name="MACD",
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                name="MACD Signal",
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        if 'MACD_Histogram' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['MACD_Histogram'],
                    name="MACD Histogram",
                    marker_color='gray',
                    opacity=0.6
                ),
                row=3, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=True,
        legend=dict(x=0, y=1)
    )
    
    return fig

def generate_enhanced_strategy_explanation(comprehensive_rec, price_action, market_context):
    """
    Generate an enhanced, LLM-style strategy explanation
    """
    direction = comprehensive_rec['direction']
    grade = comprehensive_rec['setup_grade']
    confidence = comprehensive_rec.get('trade_validity', {}).get('confidence_level', 'Medium')
    
    # Build comprehensive explanation
    explanation = f"""
### 🎯 **{direction} Setup Analysis - Grade {grade}**

**Market Assessment:**
The current gold market is showing **{market_context['market_sentiment'].lower()}** sentiment with **{market_context['volatility_regime'].lower()}** volatility conditions. 
Our AI model has identified a **{confidence.lower()} confidence** {direction.lower()} opportunity.

**Technical Foundation:**
{comprehensive_rec.get('technical_summary', 'Technical analysis indicates favorable conditions for the recommended trading approach based on current market indicators.')}

**Price Action Analysis:**
{comprehensive_rec.get('price_action_summary', 'Current price action analysis indicates market conditions suitable for the recommended trading strategy.')}

**Risk Management Strategy:**
- **Account Protection:** Risking only {comprehensive_rec.get('risk_management', {}).get('account_risk_pct') or 2.0:.1f}% of account (${comprehensive_rec.get('risk_management', {}).get('max_risk_usd') or 1000:.2f} USD)
- **Position Sizing:** {comprehensive_rec.get('position_size', {}).get('quantity') or 0} units with {comprehensive_rec.get('position_size', {}).get('leverage_used') or 1}x leverage
- **Risk:Reward:** Targeting 1:{comprehensive_rec.get('risk_management', {}).get('risk_reward_ratio') or 2.0:.1f} ratio for optimal expectancy

**Setup Reasoning:**
{comprehensive_rec.get('setup_reasoning', 'Analysis based on current market conditions and technical indicators.')}

**Execution Plan:**
1. **Entry:** ${comprehensive_rec.get('entry_price') or 0:.2f} USD (current market price)
2. **Stop Loss:** ${comprehensive_rec.get('stop_loss') or 0:.2f} USD (risk management level)
3. **Target:** ${comprehensive_rec.get('target_price') or 0:.2f} USD (profit objective)
4. **Position Size:** {comprehensive_rec.get('position_size', {}).get('quantity') or 0} units

**Key Considerations:**
- Monitor price action around key levels
- Adjust position size based on market volatility
- Use proper risk management at all times
- Consider market news and economic events

**Confidence Level:** {confidence} - This setup meets our criteria for a **Grade {grade}** trading opportunity.
    """
    
    return explanation

def display_prediction_card(prediction_result):
    """Display prediction result in a card format"""
    
    signal = prediction_result['signal']
    probability = prediction_result['probability']
    confidence = prediction_result['confidence']
    
    # Determine colors based on signal
    if 'BUY' in signal:
        color = 'green'
        icon = '📈'
    elif 'SELL' in signal:
        color = 'red'
        icon = '📉'
    else:
        color = 'gray'
        icon = '➡️'
    
    # Create columns for the prediction card
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 Trading Signal",
            value=signal,
            help=f"AI prediction based on technical analysis"
        )
    
    with col2:
        st.metric(
            label="📊 Probability",
            value=f"{probability:.1%}",
            help="Confidence level of the prediction"
        )
    
    with col3:
        st.metric(
            label="🔍 Confidence",
            value=confidence,
            help="Overall confidence in the signal"
        )
    
    # Additional info
    if prediction_result['current_price']:
        st.info(f"💰 Current Gold Price: ${prediction_result['current_price']:.2f}")
    
    # Data quality indicator
    quality_color = {
        'live': '🟢',
        'historical': '🟡',
        'unavailable': '🔴'
    }
    
    st.caption(f"Data Quality: {quality_color.get(prediction_result['data_quality'], '🔴')} {prediction_result['data_quality'].title()}")

def display_technical_indicators(indicators):
    """Display technical indicators in a formatted way"""
    
    if not indicators:
        st.warning("No technical indicators available")
        return
    
    st.subheader("📊 Technical Indicators")
    
    # Create columns for indicators
    col1, col2 = st.columns(2)
    
    with col1:
        if 'RSI' in indicators:
            rsi_value = indicators['RSI']
            rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
            st.metric("RSI (14)", f"{rsi_value:.1f}", help=f"Status: {rsi_status}")
        
        if 'MACD' in indicators:
            st.metric("MACD", f"{indicators['MACD']:.4f}")
        
        if 'BB_Position' in indicators:
            bb_pos = indicators['BB_Position']
            bb_status = "Near Upper Band" if bb_pos > 0.8 else "Near Lower Band" if bb_pos < 0.2 else "Middle Range"
            st.metric("Bollinger Band Position", f"{bb_pos:.2f}", help=f"Status: {bb_status}")
    
    with col2:
        if 'EMA_Signal' in indicators:
            ema_signal = indicators['EMA_Signal']
            ema_status = "Bullish" if ema_signal > 0 else "Bearish"
            st.metric("EMA Signal", f"{ema_signal:.2f}", help=f"Trend: {ema_status}")
        
        if 'ATR' in indicators:
            st.metric("ATR (14)", f"{indicators['ATR']:.2f}", help="Average True Range - Volatility measure")
        
        if 'Volatility' in indicators:
            st.metric("Volatility (20)", f"{indicators['Volatility']:.4f}", help="20-period price volatility")

def generate_strategy_explanation(indicators, signal):
    """Generate strategy explanation (placeholder for GPT integration)"""
    
    # This is a simplified rule-based explanation
    # In a full implementation, you would integrate with OpenAI GPT here
    
    explanations = []
    
    if 'RSI' in indicators:
        rsi = indicators['RSI']
        if rsi < 30:
            explanations.append("RSI indicates oversold conditions, suggesting potential mean reversion opportunity.")
        elif rsi > 70:
            explanations.append("RSI shows overbought levels, indicating possible reversal or consolidation.")
    
    if 'BB_Position' in indicators:
        bb_pos = indicators['BB_Position']
        if bb_pos < 0.2:
            explanations.append("Price near lower Bollinger Band suggests potential bounce (mean reversion).")
        elif bb_pos > 0.8:
            explanations.append("Price near upper Bollinger Band indicates potential resistance.")
    
    if 'MACD' in indicators and 'MACD_Signal' in indicators:
        macd = indicators['MACD']
        macd_signal = indicators['MACD_Signal']
        if macd > macd_signal:
            explanations.append("MACD above signal line suggests bullish momentum.")
        else:
            explanations.append("MACD below signal line indicates bearish momentum.")
    
    # Strategy recommendation
    if 'BUY' in signal:
        strategy = "Consider a **breakout strategy** if momentum indicators align, or **mean reversion** if price is oversold."
    elif 'SELL' in signal:
        strategy = "Consider **short-term selling** or **profit-taking** if holding long positions."
    else:
        strategy = "**Wait and watch** - mixed signals suggest staying on the sidelines."
    
    if explanations:
        return " ".join(explanations) + f" {strategy}"
    else:
        return f"Limited indicator data available. {strategy}"

def main():
    """Main Streamlit application"""
    
    # Load configuration
    config = load_config()
    
    # Initialize components
    data_loader, feature_engineer, model_trainer, predictor, updater, macro_analyzer = initialize_components()
    
    if not all([data_loader, feature_engineer, model_trainer, predictor, updater, macro_analyzer]):
        st.error("Failed to initialize components. Please check your setup.")
        return
    
    # Sidebar
    st.sidebar.title("🥇 Gold Trading AI")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate",
        ["📊 Dashboard", "🎯 Advanced Trading", "🌍 Macro Factors", "📈 Live Prediction", "🔧 Model Management", "📋 System Status"]
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto Refresh (5 min)", value=False)
    if auto_refresh:
        st.sidebar.info("Dashboard will refresh every 5 minutes")
        time.sleep(300)  # 5 minutes
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Main content based on selected page
    if page == "📊 Dashboard":
        st.title("🥇 AI-Powered Gold Trading Dashboard")
        st.markdown("Real-time gold price analysis with machine learning predictions")
        
        # Get live prediction with error handling
        with st.spinner("Getting live prediction..."):
            try:
                prediction_result = predictor.get_live_prediction()
                
                # Display prediction with proper formatting
                st.subheader("🎯 Current AI Prediction")
                
                if prediction_result and isinstance(prediction_result, dict):
                    # Enhanced prediction display with units
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        signal = prediction_result.get('signal', 'NEUTRAL')
                        signal_color = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡', 'NEUTRAL': '⚪'}
                        st.metric(
                            label="🎯 Signal",
                            value=f"{signal_color.get(signal, '⚪')} {signal}",
                            help="AI-generated trading signal based on current market conditions"
                        )
                    
                    with col2:
                        confidence = prediction_result.get('confidence', 0)
                        probability = prediction_result.get('probability', 0.5)
                        
                        if isinstance(confidence, (int, float)):
                            st.metric(
                                label="🎲 Confidence",
                                value=f"{confidence:.1f}%",
                                help="Model confidence level in the prediction"
                            )
                        elif isinstance(confidence, str):
                            # Convert string confidence to percentage
                            confidence_map = {'High': 85, 'Medium': 65, 'Low': 45}
                            conf_pct = confidence_map.get(confidence, int(probability * 100))
                            st.metric(
                                label="🎲 Confidence",
                                value=f"{conf_pct}% ({confidence})",
                                help="Model confidence level in the prediction"
                            )
                        else:
                            # Use probability as fallback
                            conf_pct = int(probability * 100) if isinstance(probability, (int, float)) else 50
                            st.metric(label="🎲 Confidence", value=f"{conf_pct}%")
                    
                    with col3:
                        # Calculate target price based on current price and prediction
                        current_price = prediction_result.get('current_price', 0)
                        prediction = prediction_result.get('prediction', 0)
                        probability = prediction_result.get('probability', 0.5)
                        
                        if isinstance(current_price, (int, float)) and current_price > 0:
                            # Calculate target price based on prediction direction and confidence
                            if prediction == 1:  # Bullish prediction
                                target_multiplier = 1 + (probability - 0.5) * 0.02  # Up to 1% move
                            else:  # Bearish prediction
                                target_multiplier = 1 - (probability - 0.5) * 0.02  # Up to 1% move
                            
                            target_price = current_price * target_multiplier
                            price_change = target_price - current_price
                            
                            st.metric(
                                label="💰 Target Price",
                                value=f"${target_price:.2f} USD",
                                delta=f"{price_change:+.2f} USD",
                                help="Predicted price target based on AI analysis"
                            )
                        else:
                            st.metric(label="💰 Target Price", value="Data Unavailable")
                    
                    with col4:
                        timeframe = prediction_result.get('timeframe', '1D')
                        st.metric(
                            label="🕰️ Timeframe",
                            value=timeframe,
                            help="Prediction timeframe"
                        )
                    
                    # Display technical indicators with proper formatting
                    if prediction_result.get('indicators'):
                        st.subheader("📈 Technical Indicators")
                        indicators = prediction_result['indicators']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            rsi = indicators.get('RSI', 0)
                            if isinstance(rsi, (int, float)):
                                rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                                st.metric(
                                    label="RSI (14)",
                                    value=f"{rsi:.1f}",
                                    delta=rsi_status,
                                    help="Relative Strength Index - momentum oscillator"
                                )
                        
                        with col2:
                            macd = indicators.get('MACD', 0)
                            if isinstance(macd, (int, float)):
                                st.metric(
                                    label="MACD",
                                    value=f"{macd:.3f}",
                                    help="Moving Average Convergence Divergence"
                                )
                        
                        with col3:
                            bb_position = indicators.get('BB_Position', 0)
                            if isinstance(bb_position, (int, float)):
                                bb_status = "Upper" if bb_position > 0.8 else "Lower" if bb_position < 0.2 else "Middle"
                                st.metric(
                                    label="Bollinger Bands",
                                    value=f"{bb_position:.2f}",
                                    delta=bb_status,
                                    help="Position within Bollinger Bands (0-1 scale)"
                                )
                else:
                    st.error("❌ Unable to get live prediction. Please check model status.")
                    
            except Exception as e:
                st.error(f"❌ Error getting live prediction: {str(e)}")
                logger.error(f"Live prediction error: {e}")
                
                # Fallback display
                st.info("🔄 Attempting to use cached prediction data...")
                try:
                    # Try to get basic market data instead
                    live_data = data_loader.fetch_live_data(days_back=1)
                    if not live_data.empty:
                        current_price = live_data['Close'].iloc[-1]
                        st.metric(
                            label="💰 Current Gold Price",
                            value=f"${current_price:.2f} USD",
                            help="Latest available gold price"
                        )
                except:
                    st.warning("⚠️ Unable to load any prediction or price data.")
        
        # Strategy explanation
        st.subheader("💡 Strategy Insight")
        explanation = generate_strategy_explanation(
            prediction_result['indicators'], 
            prediction_result['signal']
        )
        st.markdown(explanation)
        
        # Load and display chart
        st.subheader("📈 Price Chart with Technical Indicators")
        
        with st.spinner("Loading chart data..."):
            try:
                # Get recent data for chart using smart data access
                chart_data, data_source = data_loader.get_data_for_analysis(days_back=30)
                
                if not chart_data.empty and len(chart_data) > 0:
                    st.info(f"📊 Chart data source: {data_source.title()} ({len(chart_data)} data points)")
                    # Add technical indicators with error handling
                    try:
                        chart_data = feature_engineer.add_technical_indicators(chart_data)
                        
                        # Ensure we have required columns for chart
                        required_cols = ['Open', 'High', 'Low', 'Close']
                        if all(col in chart_data.columns for col in required_cols):
                            # Create and display chart
                            fig = create_candlestick_chart(chart_data.tail(200), 
                                                         title=f"Gold Price Chart - Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Local)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display chart info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.info(f"📅 **Data Points:** {len(chart_data)} candles")
                            with col2:
                                st.info(f"🕒 **Timeframe:** {chart_data.index[-1] - chart_data.index[0]}")
                            with col3:
                                current_price = chart_data['Close'].iloc[-1]
                                st.info(f"💰 **Current Price:** ${current_price:.2f} USD")
                        else:
                            st.error(f"Missing required price columns. Available: {list(chart_data.columns)}")
                    except Exception as indicator_error:
                        st.error(f"Error adding technical indicators: {indicator_error}")
                        # Try to display basic price chart without indicators
                        try:
                            basic_fig = go.Figure(data=go.Candlestick(
                                x=chart_data.index,
                                open=chart_data['Open'],
                                high=chart_data['High'],
                                low=chart_data['Low'],
                                close=chart_data['Close']
                            ))
                            basic_fig.update_layout(title="Basic Gold Price Chart (No Indicators)", height=400)
                            st.plotly_chart(basic_fig, use_container_width=True)
                        except:
                            st.error("Unable to display any chart data")
                else:
                    st.warning("⚠️ Unable to load chart data. Please check data connection.")
                    # Show data loading status
                    st.info(f"Data loader status: {type(data_loader).__name__} initialized")
                    
            except Exception as e:
                st.error(f"❌ Error loading chart: {str(e)}")
                logger.error(f"Chart loading error: {e}")
        
        # Market sentiment
        sentiment = predictor.get_market_sentiment()
        sentiment_colors = {
            'BULLISH': '🟢',
            'BEARISH': '🔴',
            'NEUTRAL': '🟡'
        }
        st.info(f"📊 Market Sentiment: {sentiment_colors.get(sentiment, '🟡')} {sentiment}")
    
    elif page == "🎯 Advanced Trading":
        st.title("🎯 Advanced Trading Analysis & Risk Management")
        st.markdown("**Professional-grade trading recommendations with risk management and position sizing**")
        
        # Get comprehensive trading recommendation
        with st.spinner("Analyzing market conditions and generating trading recommendation..."):
            try:
                comprehensive_rec = predictor.get_comprehensive_trading_recommendation()
            except Exception as e:
                st.error(f"Error getting comprehensive recommendation: {e}")
                comprehensive_rec = predictor._get_fallback_comprehensive_recommendation()
        
        # Display setup grade prominently
        grade_colors = {
            'A': '🟢', 'B': '🟡', 'C': '🟠', 'D': '🔴'
        }
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            grade = comprehensive_rec['setup_grade']
            st.markdown(f"### {grade_colors.get(grade, '⚪')} Setup Grade: **{grade}** ({comprehensive_rec['setup_quality']})", 
                       unsafe_allow_html=True)
        
        # Main trading recommendation
        st.subheader("📋 Trading Recommendation")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            direction_color = "🟢" if comprehensive_rec['direction'] == 'LONG' else "🔴" if comprehensive_rec['direction'] == 'SHORT' else "⚪"
            st.metric(
                label="🎯 Direction",
                value=f"{direction_color} {comprehensive_rec['direction']}",
                help="Recommended trading direction"
            )
        
        with col2:
            if comprehensive_rec['current_price']:
                st.metric(
                    label="💰 Entry Price",
                    value=f"${comprehensive_rec['entry_price']:.2f}",
                    help="Recommended entry price"
                )
        
        with col3:
            if comprehensive_rec['stop_loss']:
                st.metric(
                    label="🛡️ Stop Loss",
                    value=f"${comprehensive_rec['stop_loss']:.2f}",
                    help="Risk management stop loss level"
                )
        
        with col4:
            if comprehensive_rec['target_price']:
                st.metric(
                    label="🎯 Target",
                    value=f"${comprehensive_rec['target_price']:.2f}",
                    help="Profit target level"
                )
        
        # Risk Management Section
        st.subheader("⚖️ Risk Management & Position Sizing")
        
        risk_mgmt = comprehensive_rec['risk_management']
        position_info = comprehensive_rec['position_size']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Risk Metrics**")
            st.info(f"**Max Risk:** ${risk_mgmt['max_risk_usd']:.2f} ({risk_mgmt['account_risk_pct']:.1f}% of account)")
            st.info(f"**Potential Profit:** ${risk_mgmt['potential_profit_usd']:.2f}")
            st.info(f"**Risk:Reward Ratio:** 1:{risk_mgmt['risk_reward_ratio']:.1f}")
            
            # Position sizing details
            st.markdown("**📈 Position Details**")
            if position_info['quantity'] > 0:
                st.success(f"**Recommended Quantity:** {position_info['quantity']} units")
                st.info(f"**Leveraged Position:** ${position_info['leveraged_position_usd']:,.2f}")
                st.info(f"**Leverage Used:** {position_info['leverage_used']}x")
            else:
                st.warning("No position recommended due to poor setup quality")
        
        with col2:
            st.markdown("**🎯 Setup Analysis**")
            st.info(f"**Setup Reasoning:** {comprehensive_rec['setup_reasoning']}")
            st.info(f"**Technical Summary:** {comprehensive_rec['technical_summary']}")
            st.info(f"**Price Action:** {comprehensive_rec['price_action_summary']}")
        
        # Trade Validity Check
        trade_validity = comprehensive_rec['trade_validity']
        
        if trade_validity['is_valid']:
            st.success(f"✅ **{trade_validity['recommendation']}** - {trade_validity['confidence_level']} Confidence")
        else:
            st.error(f"❌ **{trade_validity['recommendation']}** - Issues: {', '.join(trade_validity['issues'])}")
        
        # Price Action Analysis
        st.subheader("🕯️ Advanced Price Action Analysis")
        
        price_action = comprehensive_rec.get('price_action_analysis', {})
        
        if price_action:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Candlestick Patterns**")
                patterns = [
                    ("Bullish Candle", price_action.get('bullish_candle', False)),
                    ("Bearish Candle", price_action.get('bearish_candle', False)),
                    ("Doji", price_action.get('doji', False)),
                    ("Hammer", price_action.get('hammer', False)),
                    ("Shooting Star", price_action.get('shooting_star', False))
                ]
                
                for pattern, detected in patterns:
                    if detected:
                        st.success(f"✅ {pattern}")
                    else:
                        st.info(f"⚪ {pattern}")
            
            with col2:
                st.markdown("**Engulfing Patterns**")
                if price_action.get('bullish_engulfing'):
                    st.success("✅ Bullish Engulfing")
                else:
                    st.info("⚪ Bullish Engulfing")
                
                if price_action.get('bearish_engulfing'):
                    st.success("✅ Bearish Engulfing")
                else:
                    st.info("⚪ Bearish Engulfing")
            
            with col3:
                st.markdown("**Price Action Strength**")
                strength = price_action.get('price_action_strength', 0)
                if strength > 0.5:
                    st.success(f"🟢 Strong Bullish ({strength:.2f})")
                elif strength < -0.5:
                    st.error(f"🔴 Strong Bearish ({strength:.2f})")
                else:
                    st.info(f"🟡 Neutral ({strength:.2f})")
                
                candle_type = price_action.get('current_candle_type', 'Unknown')
                st.info(f"**Current Candle:** {candle_type}")
        
        # Market Context
        st.subheader("🌍 Market Context")
        
        market_context = comprehensive_rec['market_context']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment = market_context['market_sentiment']
            sentiment_colors = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '🟡'}
            st.info(f"**Market Sentiment:** {sentiment_colors.get(sentiment, '🟡')} {sentiment}")
        
        with col2:
            volatility = market_context['volatility_regime']
            vol_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
            st.info(f"**Volatility Regime:** {vol_colors.get(volatility, '⚪')} {volatility}")
        
        with col3:
            data_quality = market_context['data_quality']
            quality_colors = {'live': '🟢', 'historical': '🟡', 'unavailable': '🔴'}
            st.info(f"**Data Quality:** {quality_colors.get(data_quality, '🔴')} {data_quality.title()}")
        
        # Enhanced Strategy Explanation with LLM-style analysis
        st.subheader("🧠 AI Strategy Explanation")
        
        strategy_explanation = generate_enhanced_strategy_explanation(
            comprehensive_rec, price_action, market_context
        )
        
        st.markdown(strategy_explanation)
        
        # Trading Journal Entry Template
        st.subheader("📝 Trading Journal Template")
        
        with st.expander("Click to generate trading journal entry"):
            journal_entry = f"""
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Symbol:** Gold (GC=F)
**Setup Grade:** {comprehensive_rec['setup_grade']} ({comprehensive_rec['setup_quality']})
**Direction:** {comprehensive_rec['direction']}
**Entry:** ${comprehensive_rec['entry_price']:.2f}
**Stop Loss:** ${comprehensive_rec['stop_loss']:.2f}
**Target:** ${comprehensive_rec['target_price']:.2f}
**Risk:** ${risk_mgmt['max_risk_usd']:.2f}
**R:R Ratio:** 1:{risk_mgmt['risk_reward_ratio']:.1f}
**Position Size:** {position_info['quantity']} units

**Analysis:**
{comprehensive_rec['setup_reasoning']}

**Technical Conditions:**
{comprehensive_rec['technical_summary']}

**Price Action:**
{comprehensive_rec['price_action_summary']}

**Market Context:**
Sentiment: {market_context['market_sentiment']}
Volatility: {market_context['volatility_regime']}

**Trade Decision:** {trade_validity['recommendation']}
**Confidence:** {trade_validity['confidence_level']}
            """
            
            st.code(journal_entry, language='markdown')
    
    elif page == "🌍 Macro Factors":
        st.title("🌍 Comprehensive Macroeconomic Analysis")
        st.markdown("**Real-time monitoring of 100+ quantitative factors from FRED API influencing gold prices**")
        
        # Add loading indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Get comprehensive macro analysis using our enhanced analyzer
        try:
            status_text.text("Loading historical gold data...")
            progress_bar.progress(10)
            
            # Load historical gold data for correlation analysis
            gold_data = data_loader.load_historical_data()
            if not gold_data.empty:
                gold_prices = gold_data['Close']
                status_text.text(f"Gold data loaded: {len(gold_prices)} records")
            else:
                gold_prices = None
                status_text.text("No gold data available")
            
            progress_bar.progress(30)
            status_text.text("Fetching FRED economic indicators...")
            
            # Update all macro factors with comprehensive FRED data
            macro_summary = macro_analyzer.update_all_factors(gold_prices)
            
            progress_bar.progress(70)
            status_text.text("Processing dashboard data...")
            
            dashboard_data = macro_analyzer.get_macro_dashboard_summary()
            
            progress_bar.progress(100)
            status_text.text("✅ Data loading complete!")
            
            # Clear progress indicators
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error fetching comprehensive macro data: {e}")
            macro_summary = {
                'overall_sentiment': 'Unknown',
                'top_factors': [],
                'summary': 'Comprehensive macro analysis unavailable'
            }
            dashboard_data = {
                'overall_summary': macro_summary,
                'categories': {},
                'category_stats': {},
                'total_indicators': 0
            }
        
        # COMPREHENSIVE EXECUTIVE SUMMARY
        st.subheader("📈 Executive Summary - Macro Economic Impact on Gold")
        
        overall_summary = dashboard_data.get('overall_summary', {})
        categories = dashboard_data.get('categories', {})
        
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_indicators = dashboard_data.get('total_indicators', 0)
            st.metric(
                label="📊 Total Indicators",
                value=total_indicators,
                help="Total FRED economic indicators monitored"
            )
        
        with col2:
            valid_indicators = sum(len([f for f in factors if f.get('current_value') is not None]) 
                                 for factors in categories.values())
            success_rate = (valid_indicators / total_indicators * 100) if total_indicators > 0 else 0
            st.metric(
                label="✅ Data Success",
                value=f"{success_rate:.0f}%",
                delta=f"{valid_indicators}/{total_indicators}",
                help="Percentage of indicators with current valid data"
            )
        
        with col3:
            sentiment = overall_summary.get('overall_sentiment', 'Unknown')
            sentiment_colors = {'Bullish': '🟢', 'Bearish': '🔴', 'Mixed': '🟡', 'Unknown': '⚪'}
            st.metric(
                label="🎯 Gold Sentiment",
                value=f"{sentiment_colors.get(sentiment, '⚪')} {sentiment}",
                help="Overall macroeconomic sentiment for gold prices"
            )
        
        with col4:
            bullish_count = len(overall_summary.get('bullish_factors', []))
            bearish_count = len(overall_summary.get('bearish_factors', []))
            net_bias = bullish_count - bearish_count
            st.metric(
                label="⚖️ Factor Balance",
                value=f"{bullish_count}B / {bearish_count}B",
                delta=f"{net_bias:+d} net bias",
                help="Bullish vs Bearish factors count"
            )
        
        with col5:
            data_freshness = dashboard_data.get('data_freshness', 'Unknown')
            current_time, local_tz = get_current_time_with_tz()
            tz_info = get_data_source_timezones()
            
            if data_freshness != 'Unknown':
                formatted_time = format_time_with_source_tz(data_freshness, "FRED/Market Data")
                display_time = formatted_time.split(' (Local)')[0].split(' ')[-1]  # Show just time
            else:
                display_time = "Unknown"
            
            st.metric(
                label="🕒 Last Updated",
                value=display_time,
                help=f"Last data refresh time\nFRED Data: {tz_info['FRED']}\nMarket Data: {tz_info['Market Data']}"
            )
        
        # Impact Summary Cards
        st.markdown("### 📊 Impact Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # High Impact Factors
            high_impact_factors = []
            for factors in categories.values():
                for factor in factors:
                    influence = factor.get('influence_score', 0)
                    if isinstance(influence, (int, float)) and abs(influence) > 0.7:
                        high_impact_factors.append(factor)
            
            st.markdown("🔥 **High Impact Factors**")
            if high_impact_factors:
                for factor in high_impact_factors[:3]:
                    impact = factor.get('expected_impact', 'Neutral')
                    impact_color = {'Bullish': '🟢', 'Bearish': '🔴', 'Neutral': '🟡'}.get(impact, '⚪')
                    st.write(f"{impact_color} {factor.get('name', 'Unknown')[:30]}...")
            else:
                st.write("⚪ No high-impact factors identified")
        
        with col2:
            # Recent Changes
            st.markdown("🔄 **Recent Significant Changes**")
            significant_changes = []
            for factors in categories.values():
                for factor in factors:
                    change = factor.get('change_pct', 0)
                    if isinstance(change, (int, float)) and abs(change) > 2.0:
                        significant_changes.append(factor)
            
            if significant_changes:
                # Sort by absolute change
                significant_changes.sort(key=lambda x: abs(x.get('change_pct', 0)), reverse=True)
                for factor in significant_changes[:3]:
                    change = factor.get('change_pct', 0)
                    change_color = '🟢' if change > 0 else '🔴'
                    st.write(f"{change_color} {factor.get('name', 'Unknown')[:25]}: {change:+.1f}%")
            else:
                st.write("⚪ No significant changes detected")
        
        with col3:
            # Next Updates
            st.markdown("🕰️ **Next Update Schedule**")
            # Use the already imported datetime from top of file
            
            # Calculate next update times for different data frequencies with timezone info
            try:
                current_time, local_tz = get_current_time_with_tz()
                tz_info = get_data_source_timezones()
                
                daily_next = (current_time + timedelta(days=1)).replace(hour=8, minute=30, second=0)
                weekly_next = current_time + timedelta(days=(4 - current_time.weekday()) % 7 or 7)
                monthly_next = current_time.replace(day=1) + timedelta(days=32)
                monthly_next = monthly_next.replace(day=15, hour=8, minute=30, second=0)
                
                st.write(f"📅 **Daily indicators:** {daily_next.strftime('%m/%d %H:%M')} (Local)")
                st.write(f"📅 **Weekly indicators:** {weekly_next.strftime('%m/%d %H:%M')} (Local)")
                st.write(f"📅 **Monthly indicators:** {monthly_next.strftime('%m/%d %H:%M')} (Local)")
                st.caption(f"**Data Sources:** FRED ({tz_info['FRED']}), Market ({tz_info['Market Data']})")
            except Exception as e:
                st.write("📅 **Daily indicators:** Next business day (Local)")
                st.write("📅 **Weekly indicators:** Next week (Local)")
                st.write("📅 **Monthly indicators:** Next month (Local)")
                st.caption("**Note:** Times shown in your local timezone")
        
        st.divider()
        
        # Overall Macro Sentiment
        overall_summary = dashboard_data.get('overall_summary', {})
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            sentiment = overall_summary.get('overall_sentiment', 'Unknown')
            sentiment_colors = {
                'Bullish': '🟢',
                'Bearish': '🔴', 
                'Mixed': '🟡',
                'Unknown': '⚪'
            }
            st.markdown(f"### {sentiment_colors.get(sentiment, '⚪')} Overall Macro Sentiment: **{sentiment}**")
        
        # Sentiment Scores
        bullish_score = overall_summary.get('bullish_score', 0)
        bearish_score = overall_summary.get('bearish_score', 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="🟢 Bullish Factors Score",
                value=f"{bullish_score:.1f}",
                help="Combined influence score of bullish factors"
            )
        
        with col2:
            st.metric(
                label="🔴 Bearish Factors Score",
                value=f"{bearish_score:.1f}",
                help="Combined influence score of bearish factors"
            )
        
        with col3:
            net_score = bullish_score - bearish_score
            st.metric(
                label="⚖️ Net Sentiment Score",
                value=f"{net_score:+.1f}",
                delta=f"{'Bullish' if net_score > 0 else 'Bearish' if net_score < 0 else 'Neutral'} Bias",
                help="Net macro sentiment (Bullish - Bearish)"
            )
        
        # Summary Table for Changed Factors
        st.subheader("🔄 Recent Changes Summary")
        
        # Create summary table for factors with significant changes
        changed_factors = []
        all_factors = []
        
        # Collect all factors from categories
        for category_name, factors in categories.items():
            for factor in factors:
                all_factors.append(factor)
                change_pct = factor.get('change_pct', 0)
                if isinstance(change_pct, (int, float)) and abs(change_pct) > 0.5:  # Show changes > 0.5%
                    changed_factors.append(factor)
        
        if changed_factors:
            # Sort by absolute change percentage
            changed_factors.sort(key=lambda x: abs(x.get('change_pct', 0)), reverse=True)
            
            # Create summary table
            summary_data = []
            for factor in changed_factors[:15]:  # Show top 15 changed factors
                change_pct = factor.get('change_pct', 0)
                influence = factor.get('influence_score', 0)
                
                # Determine change significance
                if abs(change_pct) > 5:
                    significance = "🔥 Major"
                elif abs(change_pct) > 2:
                    significance = "⚡ Significant"
                else:
                    significance = "📈 Minor"
                
                # Determine influence level
                if isinstance(influence, (int, float)):
                    if abs(influence) > 0.7:
                        influence_level = "🎯 High"
                    elif abs(influence) > 0.4:
                        influence_level = "🎲 Medium"
                    else:
                        influence_level = "📊 Low"
                else:
                    influence_level = "❓ Unknown"
                
                # Impact direction
                impact = factor.get('expected_impact', 'Neutral')
                impact_emoji = {'Bullish': '🟢', 'Bearish': '🔴', 'Neutral': '🟡', 'Mixed': '🟠'}.get(impact, '⚪')
                
                # Determine data source timezone
                category = factor.get('category', 'Economic')
                if 'market' in category.lower() or 'gold' in factor.get('name', '').lower():
                    data_source = "Market (NYSE)"
                else:
                    data_source = "FRED (Fed)"
                
                summary_data.append({
                    'Factor': factor.get('name', 'Unknown')[:35],
                    'Category': factor.get('category', 'N/A'),
                    'Change': f"{change_pct:+.2f}%",
                    'Significance': significance,
                    'Influence': influence_level,
                    'Gold Impact': f"{impact_emoji} {impact}",
                    'Data Source': data_source,
                    'Frequency': factor.get('frequency', 'N/A')
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                major_changes = len([f for f in changed_factors if abs(f.get('change_pct', 0)) > 5])
                st.metric("Major Changes", major_changes, help="Factors with >5% change")
            with col2:
                bullish_changes = len([f for f in changed_factors if f.get('expected_impact') == 'Bullish'])
                st.metric("Bullish Changes", bullish_changes, help="Changes supporting gold")
            with col3:
                bearish_changes = len([f for f in changed_factors if f.get('expected_impact') == 'Bearish'])
                st.metric("Bearish Changes", bearish_changes, help="Changes opposing gold")
            with col4:
                high_influence_changes = len([f for f in changed_factors if abs(f.get('influence_score', 0)) > 0.7])
                st.metric("High Impact Changes", high_influence_changes, help="High influence factors that changed")
        else:
            st.info("No significant factor changes detected in recent data.")
        
        st.divider()
        
        # Enhanced Top Influencing Factors - Sorted by Update Frequency
        st.subheader("📊 Factors by Update Frequency - Most Current Data First")
        
        top_factors = overall_summary.get('top_factors', [])
        
        if top_factors:
            # Sort factors by update frequency (most frequent first)
            frequency_priority = {
                'Daily': 1,
                'Weekly': 2, 
                'Monthly': 3,
                'Quarterly': 4,
                'Annual': 5,
                'N/A': 6
            }
            
            # Sort by frequency priority, then by influence score
            sorted_factors = sorted(top_factors, key=lambda x: (
                frequency_priority.get(x.get('frequency', 'N/A'), 6),
                -abs(x.get('influence_score', 0))
            ))
            
            # Group by frequency for better organization
            frequency_groups = {}
            for factor in sorted_factors:
                freq = factor.get('frequency', 'N/A')
                if freq not in frequency_groups:
                    frequency_groups[freq] = []
                frequency_groups[freq].append(factor)
            
            # Display factors grouped by frequency
            for freq in ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Annual', 'N/A']:
                if freq in frequency_groups and frequency_groups[freq]:
                    factors_in_group = frequency_groups[freq]
                    
                    # Frequency section header
                    freq_emoji = {
                        'Daily': '📅',
                        'Weekly': '📆', 
                        'Monthly': '🗓️',
                        'Quarterly': '📋',
                        'Annual': '📊',
                        'N/A': '❓'
                    }.get(freq, '📈')
                    
                    st.markdown(f"### {freq_emoji} {freq} Updates ({len(factors_in_group)} factors)")
                    
                    # Enhanced factor cards with comprehensive information
                    for i, factor in enumerate(factors_in_group[:8], 1):  # Show top 8 per frequency
                        with st.expander(f"🔢 {i}. {factor.get('name', 'Unknown')} - {factor.get('category', 'N/A')}", expanded=(i <= 2)):
                            
                            # Factor description and impact
                            factor_descriptions = {
                                'Consumer Price Index': 'Measures average change in prices paid by consumers for goods and services. Higher CPI typically supports gold as inflation hedge.',
                                'Federal Funds Rate': 'Interest rate at which banks lend to each other overnight. Higher rates generally negative for gold due to opportunity cost.',
                                'Unemployment Rate': 'Percentage of labor force that is unemployed. Higher unemployment often leads to economic uncertainty, supporting gold.',
                                'Gross Domestic Product': 'Total value of goods and services produced. Strong GDP growth typically negative for gold as safe haven.',
                                'Core CPI': 'CPI excluding volatile food and energy prices. More stable inflation measure, key for Fed policy decisions.',
                                'Nonfarm Payrolls': 'Number of paid workers excluding farm, government, and non-profit employees. Strong jobs data typically negative for gold.',
                                '10-Year Treasury Rate': 'Yield on 10-year US government bonds. Higher yields increase opportunity cost of holding non-yielding gold.',
                                'M2 Money Supply': 'Broad measure of money supply including cash, deposits, and near money. Expansion typically supports gold prices.',
                                'Consumer Sentiment': 'Index measuring consumer confidence in economic conditions. Higher confidence typically negative for gold safe haven demand.',
                                'Housing Starts': 'Number of new residential construction projects. Strong housing data indicates economic growth, typically negative for gold.'
                            }
                            
                            factor_name = factor.get('name', 'Unknown')
                            description = factor_descriptions.get(factor_name, f"Economic indicator in {factor.get('category', 'N/A')} category that influences gold prices through market sentiment and economic fundamentals.")
                            
                            st.markdown(f"**📝 Description:** {description}")
                            
                            # Key metrics in columns
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                current_val = factor.get('current_value')
                                unit = factor.get('unit', '')
                                if current_val is not None:
                                    display_val = f"{current_val:.2f}"
                                    if unit and unit != 'N/A':
                                        display_val += f" {unit}"
                                    st.metric(
                                        label="Current Value",
                                        value=display_val,
                                        delta=f"{factor.get('change_pct', 0):+.2f}%"
                                    )
                                else:
                                    st.metric(label="Current Value", value="N/A")
                            
                            with col2:
                                influence = factor.get('influence_score', 0)
                                if isinstance(influence, (int, float)):
                                    influence_level = "High" if abs(influence) > 0.7 else "Medium" if abs(influence) > 0.4 else "Low"
                                    st.metric(
                                        label="Influence Score",
                                        value=f"{influence:.2f}",
                                        help=f"Impact level: {influence_level}"
                                    )
                                else:
                                    st.metric(label="Influence Score", value="N/A")
                            
                            with col3:
                                impact = factor.get('expected_impact', 'Neutral')
                                impact_colors = {'Bullish': '🟢', 'Bearish': '🔴', 'Neutral': '🟡', 'Mixed': '🟠'}
                                st.metric(
                                    label="Gold Impact",
                                    value=f"{impact_colors.get(impact, '⚪')} {impact}",
                                    help="Expected impact on gold prices"
                                )
                            
                            with col4:
                                quality = factor.get('data_quality', 'unknown')
                                quality_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔴', 'unknown': '⚪'}
                                st.metric(
                                    label="Data Quality",
                                    value=f"{quality_colors.get(quality, '⚪')} {quality.title()}",
                                    help="Reliability of current data"
                                )
                            
                            # Timing information
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                last_updated = factor.get('last_updated', 'N/A')
                                tz_info = get_data_source_timezones()
                                
                                if last_updated != 'N/A':
                                    try:
                                        # Determine data source timezone based on factor category
                                        category = factor.get('category', 'Economic')
                                        if 'market' in category.lower() or 'gold' in factor.get('name', '').lower():
                                            source_tz = tz_info['Market Data']
                                        else:
                                            source_tz = tz_info['FRED']
                                        
                                        formatted_time = format_time_with_source_tz(last_updated, source_tz)
                                        st.info(f"📅 **Last Updated:** {formatted_time}")
                                    except:
                                        st.info(f"📅 **Last Updated:** {last_updated} (Source TZ: {tz_info['FRED']})")
                                else:
                                    st.info("📅 **Last Updated:** Unknown")
                            
                            with col2:
                                frequency = factor.get('frequency', 'N/A')
                                tz_info = get_data_source_timezones()
                                current_time, local_tz = get_current_time_with_tz()
                                
                                # Calculate next update with timezone info
                                try:
                                    if frequency == 'Daily':
                                        next_update_time = (current_time + timedelta(days=1)).replace(hour=8, minute=30)
                                        next_update = f"Tomorrow 8:30 AM (Local)"
                                    elif frequency == 'Weekly':
                                        days_until_friday = (4 - current_time.weekday()) % 7
                                        if days_until_friday == 0:
                                            days_until_friday = 7
                                        next_update_time = current_time + timedelta(days=days_until_friday)
                                        next_update = f"{next_update_time.strftime('%a %m/%d')} (Local)"
                                    elif frequency == 'Monthly':
                                        if current_time.day < 15:
                                            next_update = "Mid-month (Local)"
                                        else:
                                            next_month = current_time.replace(day=1) + timedelta(days=32)
                                            next_update = f"{next_month.strftime('%b %Y')} (Local)"
                                    else:
                                        next_update = f"Next {frequency.lower()} (Local)"
                                except:
                                    next_update = f"Next {frequency.lower()} (Local)" if frequency != 'N/A' else "Unknown"
                                
                                # Determine source timezone
                                category = factor.get('category', 'Economic')
                                if 'market' in category.lower():
                                    source_tz_info = tz_info['Market Data']
                                else:
                                    source_tz_info = tz_info['FRED']
                                
                                st.info(f"🔄 **Next Update:** {next_update}\n**Source TZ:** {source_tz_info}")
                            
                            with col3:
                                correlation = factor.get('correlation', 0)
                                if isinstance(correlation, (int, float)):
                                    corr_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"
                                    corr_direction = "Positive" if correlation > 0 else "Negative" if correlation < 0 else "Neutral"
                                    st.info(f"🔗 **Correlation:** {corr_strength} {corr_direction}")
                                else:
                                    st.info("🔗 **Correlation:** Not calculated")
                            
                            # Impact assessment
                            st.markdown("**🎯 Impact Assessment:**")
                            
                            # Generate impact explanation based on factor type and current trend
                            change_pct = factor.get('change_pct', 0)
                            if isinstance(change_pct, (int, float)):
                                if abs(change_pct) > 5:
                                    impact_magnitude = "significant"
                                elif abs(change_pct) > 2:
                                    impact_magnitude = "moderate"
                                else:
                                    impact_magnitude = "minimal"
                                
                                trend_direction = "increasing" if change_pct > 0 else "decreasing" if change_pct < 0 else "stable"
                                
                                impact_text = f"This indicator is currently {trend_direction} by {abs(change_pct):.1f}%, representing a {impact_magnitude} change. "
                                
                                if impact == 'Bullish':
                                    impact_text += "This trend is generally supportive of higher gold prices."
                                elif impact == 'Bearish':
                                    impact_text += "This trend is generally negative for gold prices."
                                else:
                                    impact_text += "The impact on gold prices is currently neutral or mixed."
                                
                                st.write(impact_text)
                            else:
                                st.write("Impact assessment unavailable due to insufficient data.")
            
            # Comprehensive Category Analysis with Enhanced FRED Data
            st.subheader("📈 Comprehensive Factor Analysis by Category")
            
            categories = dashboard_data.get('categories', {})
            category_stats = dashboard_data.get('category_stats', {})
            
            if categories:
                # Category overview metrics
                st.markdown("**Category Overview:**")
                cols = st.columns(min(len(categories), 4))
                
                for idx, (category, stats) in enumerate(list(category_stats.items())[:4]):
                    with cols[idx]:
                        valid_count = stats.get('valid_indicators', 0)
                        total_count = stats.get('total_indicators', 0)
                        st.metric(
                            label=f"📈 {category}",
                            value=f"{valid_count}/{total_count}",
                            delta=f"{stats.get('avg_change', 0):+.1f}% avg",
                            help=f"Valid indicators / Total indicators in {category}"
                        )
                
                st.divider()
                
                # Display each category with enhanced data
                for category, factors in categories.items():
                    stats = category_stats.get(category, {})
                    valid_factors = [f for f in factors if f.get('current_value') is not None]
                    
                    # Category header with statistics
                    bullish_count = stats.get('bullish_count', 0)
                    bearish_count = stats.get('bearish_count', 0)
                    
                    header_text = f"📈 {category} ({len(valid_factors)}/{len(factors)} active)"
                    if bullish_count > 0 or bearish_count > 0:
                        header_text += f" | 🟢 {bullish_count} Bullish | 🔴 {bearish_count} Bearish"
                    
                    with st.expander(header_text, expanded=(category in ['Inflation', 'Interest Rates', 'Employment'])):
                        
                        if valid_factors:
                            # Top factor in category
                            top_factor = stats.get('top_factor')
                            if top_factor:
                                st.info(f"🏆 **Top Factor:** {top_factor.get('name', 'Unknown')} (Influence: {top_factor.get('influence_score', 0):.1f})")
                            
                            # Display factors in a more organized way
                            for i, factor in enumerate(valid_factors[:8]):  # Show top 8 per category
                                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                                
                                with col1:
                                    st.write(f"**{factor.get('name', 'Unknown')}**")
                                    st.caption(f"Symbol: {factor.get('symbol', 'N/A')} | {factor.get('frequency', 'N/A')}")
                                
                                with col2:
                                    current_val = factor.get('current_value')
                                    if current_val is not None:
                                        unit = factor.get('unit', '')
                                        if unit and unit != 'N/A':
                                            st.metric(
                                                label="Value",
                                                value=f"{current_val:.2f}",
                                                delta=f"{factor.get('change_pct', 0):+.2f}%",
                                                help=f"Unit: {unit}"
                                            )
                                        else:
                                            st.metric(
                                                label="Value",
                                                value=f"{current_val:.2f}",
                                                delta=f"{factor.get('change_pct', 0):+.2f}%"
                                            )
                                    else:
                                        st.write("N/A")
                                
                                with col3:
                                    correlation = factor.get('correlation', 0)
                                    corr_strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.4 else "Weak"
                                    st.write(f"**Corr:** {correlation:.3f}")
                                    st.caption(corr_strength)
                                
                                with col4:
                                    influence = factor.get('influence_score', 0)
                                    st.write(f"**Influence:** {influence:.1f}")
                                    quality = factor.get('data_quality', 'unknown')
                                    quality_color = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(quality, '⚪')
                                    st.caption(f"{quality_color} {quality.title()}")
                                
                                with col5:
                                    impact = factor.get('expected_impact', 'Neutral')
                                    impact_color = {
                                        'Bullish': '🟢',
                                        'Bearish': '🔴',
                                        'Neutral': '🟡',
                                        'Mixed': '🟠'
                                    }.get(impact, '⚪')
                                    
                                    st.write(f"**{impact_color} {impact}**")
                                    st.caption(f"Updated: {factor.get('last_updated', 'N/A')[:10]}")
                                
                                if i < len(valid_factors[:8]) - 1:
                                    st.divider()
                        else:
                            st.warning(f"No valid data available for {category} indicators")
            
            # Comprehensive AI Macro Analysis
            st.subheader("🧠 AI Macro Analysis - Enhanced with 100+ FRED Indicators")
            
            macro_summary = overall_summary.get('summary', 'No comprehensive macro summary available.')
            st.markdown(f"**Current Assessment:** {macro_summary}")
            
            # Enhanced macro insights with comprehensive data
            bullish_factors = overall_summary.get('bullish_factors', [])
            bearish_factors = overall_summary.get('bearish_factors', [])
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="🟢 Bullish Indicators",
                    value=len(bullish_factors),
                    help="Number of indicators showing bullish signals for gold"
                )
            with col2:
                st.metric(
                    label="🔴 Bearish Indicators",
                    value=len(bearish_factors),
                    help="Number of indicators showing bearish signals for gold"
                )
            with col3:
                total_active = len([f for f in top_factors if f.get('current_value') is not None])
                st.metric(
                    label="📈 Active Indicators",
                    value=total_active,
                    help="Total indicators with current valid data"
                )
            
            if bullish_factors or bearish_factors:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🟢 Top Bullish Factors:**")
                    if bullish_factors:
                        for factor in bullish_factors[:6]:  # Top 6
                            influence = factor.get('influence_score', 0)
                            change = factor.get('change_pct', 0)
                            st.success(f"✅ **{factor.get('name', 'Unknown')}** (Influence: {influence:.1f}, Change: {change:+.1f}%)")
                    else:
                        st.info("No significant bullish factors identified")
                
                with col2:
                    st.markdown("**🔴 Top Bearish Factors:**")
                    if bearish_factors:
                        for factor in bearish_factors[:6]:  # Top 6
                            influence = factor.get('influence_score', 0)
                            change = factor.get('change_pct', 0)
                            st.error(f"❌ **{factor.get('name', 'Unknown')}** (Influence: {influence:.1f}, Change: {change:+.1f}%)")
                    else:
                        st.info("No significant bearish factors identified")
            
            # FRED Data Quality Summary
            st.divider()
            st.subheader("📊 FRED Data Quality & Coverage")
            
            if categories:
                quality_stats = {'high': 0, 'medium': 0, 'low': 0, 'unavailable': 0}
                for factors in categories.values():
                    for factor in factors:
                        quality = factor.get('data_quality', 'unavailable')
                        quality_stats[quality] = quality_stats.get(quality, 0) + 1
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        label="🟢 High Quality",
                        value=quality_stats['high'],
                        help="Indicators with high data quality (10+ recent observations)"
                    )
                with col2:
                    st.metric(
                        label="🟡 Medium Quality",
                        value=quality_stats['medium'],
                        help="Indicators with medium data quality (5-9 recent observations)"
                    )
                with col3:
                    st.metric(
                        label="🔴 Low Quality",
                        value=quality_stats['low'],
                        help="Indicators with low data quality (<5 recent observations)"
                    )
                with col4:
                    st.metric(
                        label="⚪ Unavailable",
                        value=quality_stats['unavailable'],
                        help="Indicators with no current data available"
                    )
            
            # Correlation Heatmap (if we have enough data)
            if len(top_factors) >= 5:
                st.subheader("🔥 Factor Correlation Heatmap")
                
                # Create correlation matrix for visualization
                factor_names = [f.get('name', 'Unknown')[:15] + '...' if len(f.get('name', '')) > 15 else f.get('name', 'Unknown') for f in top_factors[:8]]
                correlations = [f.get('correlation', 0) for f in top_factors[:8]]
                
                # Create a simple correlation visualization
                fig = go.Figure(data=go.Bar(
                    x=factor_names,
                    y=correlations,
                    marker_color=['green' if c > 0 else 'red' for c in correlations],
                    text=[f"{c:.3f}" for c in correlations],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Gold Price Correlations with Key Factors",
                    xaxis_title="Factors",
                    yaxis_title="Correlation with Gold",
                    height=400,
                    xaxis_tickangle=-45
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            
            # Data Quality and Update Info
            st.subheader("ℹ️ Data Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                last_updated = dashboard_data.get('data_freshness', 'Unknown')
                if last_updated != 'Unknown':
                    try:
                        st.info(f"**Last Updated:** {last_updated}")
                    except:
                        st.info(f"**Last Updated:** {last_updated}")
                else:
                    st.info("**Last Updated:** Unknown")
            
            with col2:
                total_factors = len(top_factors)
                st.info(f"**Factors Analyzed:** {total_factors}")
            
            with col3:
                if dashboard_data.get('total_indicators', 0) > 0:
                    st.success("✅ Data quality: Good")
                else:
                    st.warning("⚠️ Some data unavailable")
            
            # Manual refresh button
            if st.button("🔄 Refresh Macro Analysis", key="refresh_macro"):
                st.rerun()
        
        else:
            st.warning("⚠️ No macro factors data available. Please check your API keys and internet connection.")
            
            # Show configuration help
            with st.expander("📋 Configuration Help"):
                st.markdown("""
                **To enable macro factors analysis:**
                
                1. **FRED API Key**: Get a free API key from [FRED Economic Data](https://fred.stlouisfed.org/docs/api/api_key.html)
                2. **Update config.yaml**: Add your FRED API key to the `api_keys.fred_key` field
                3. **Internet Connection**: Ensure you have internet access for real-time data
                
                **Supported Data Sources:**
                - **Market Data**: yfinance (DXY, Treasury yields, currencies, commodities)
                - **Economic Data**: FRED API (inflation, employment, GDP, Fed rates)
                
                **Key Factors Monitored:**
                - 🏛️ **Monetary Policy**: Fed Funds Rate, Treasury yields
                - 💱 **Currency**: USD Index (DXY), major forex pairs
                - 📊 **Economic**: Inflation (CPI), employment, GDP growth
                - 📈 **Market Sentiment**: VIX, S&P 500, risk-on/risk-off
                - 🛢️ **Commodities**: Oil, silver, copper prices
                """)
    
    elif page == "📈 Live Prediction":
        st.title("📈 Live Prediction Analysis")
        st.markdown("**Real-time AI predictions with comprehensive market analysis**")
        
        # Get current time for display
        current_time, local_tz = get_current_time_with_tz()
        tz_info = get_data_source_timezones()
        
        # Display current time and data sources
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"🕰️ **Analysis Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S')} (Local)")
        with col2:
            st.info(f"📊 **Data Source:** {tz_info['Market Data']}")
        with col3:
            st.info(f"🌍 **System TZ:** {tz_info['System']}")
        
        st.divider()
        
        # Get detailed prediction with error handling
        with st.spinner("Generating comprehensive prediction analysis..."):
            try:
                prediction_result = predictor.get_live_prediction()
                
                if prediction_result and isinstance(prediction_result, dict):
                    # Enhanced prediction display
                    st.subheader("🎯 Current AI Prediction")
                    
                    # Main prediction metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        signal = prediction_result.get('signal', 'NEUTRAL')
                        signal_colors = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡', 'NEUTRAL': '⚪'}
                        st.metric(
                            label="🎯 Trading Signal",
                            value=f"{signal_colors.get(signal, '⚪')} {signal}",
                            help="AI-generated trading recommendation"
                        )
                    
                    with col2:
                        confidence = prediction_result.get('confidence', 0)
                        if isinstance(confidence, (int, float)):
                            confidence_level = "High" if confidence > 75 else "Medium" if confidence > 50 else "Low"
                            st.metric(
                                label="🎲 Confidence Level",
                                value=f"{confidence:.1f}%",
                                delta=confidence_level,
                                help="Model confidence in the prediction"
                            )
                        else:
                            st.metric(label="🎲 Confidence Level", value="Calculating...")
                    
                    with col3:
                        predicted_price = prediction_result.get('predicted_price', 0)
                        if isinstance(predicted_price, (int, float)) and predicted_price > 0:
                            st.metric(
                                label="💰 Target Price",
                                value=f"${predicted_price:.2f} USD",
                                help="Predicted gold price target"
                            )
                        else:
                            st.metric(label="💰 Target Price", value="Analyzing...")
                    
                    with col4:
                        timeframe = prediction_result.get('timeframe', '24H')
                        st.metric(
                            label="🕰️ Timeframe",
                            value=timeframe,
                            help="Prediction time horizon"
                        )
                    
                    st.divider()
                    
                    # Detailed analysis sections
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Technical Analysis")
                        
                        if prediction_result.get('indicators'):
                            indicators = prediction_result['indicators']
                            
                            # RSI Analysis
                            rsi = indicators.get('RSI', 0)
                            if isinstance(rsi, (int, float)):
                                rsi_interpretation = (
                                    "Oversold - Potential buying opportunity" if rsi < 30 else
                                    "Overbought - Potential selling opportunity" if rsi > 70 else
                                    "Neutral - No extreme conditions"
                                )
                                st.metric(
                                    label="RSI (14-period)",
                                    value=f"{rsi:.1f}",
                                    delta=rsi_interpretation,
                                    help="Relative Strength Index - measures momentum"
                                )
                            
                            # MACD Analysis
                            macd = indicators.get('MACD', 0)
                            macd_signal = indicators.get('MACD_Signal', 0)
                            if isinstance(macd, (int, float)) and isinstance(macd_signal, (int, float)):
                                macd_trend = "Bullish" if macd > macd_signal else "Bearish"
                                st.metric(
                                    label="MACD",
                                    value=f"{macd:.4f}",
                                    delta=f"{macd_trend} trend",
                                    help="Moving Average Convergence Divergence"
                                )
                            
                            # Bollinger Bands
                            bb_position = indicators.get('BB_Position', 0.5)
                            if isinstance(bb_position, (int, float)):
                                bb_interpretation = (
                                    "Near upper band - Potential resistance" if bb_position > 0.8 else
                                    "Near lower band - Potential support" if bb_position < 0.2 else
                                    "Middle range - Neutral position"
                                )
                                st.metric(
                                    label="Bollinger Band Position",
                                    value=f"{bb_position:.2f}",
                                    delta=bb_interpretation,
                                    help="Position within Bollinger Bands (0=lower, 1=upper)"
                                )
                        else:
                            st.info("🔄 Technical indicators are being calculated...")
                    
                    with col2:
                        st.subheader("📉 Prediction History")
                        
                        # Get prediction history with error handling
                        try:
                            history = predictor.get_prediction_history(days=7)
                            
                            if history and len(history) > 0:
                                history_df = pd.DataFrame(history)
                                
                                # Format the history data properly
                                if 'timestamp' in history_df.columns:
                                    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                                    history_df['Time'] = history_df['timestamp'].dt.strftime('%m/%d %H:%M')
                                
                                # Format price and probability columns
                                if 'price' in history_df.columns:
                                    history_df['Price (USD)'] = history_df['price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else "N/A")
                                
                                if 'probability' in history_df.columns:
                                    history_df['Confidence (%)'] = history_df['probability'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else "N/A")
                                
                                # Display formatted history
                                display_cols = []
                                if 'Time' in history_df.columns:
                                    display_cols.append('Time')
                                if 'Price (USD)' in history_df.columns:
                                    display_cols.append('Price (USD)')
                                if 'signal' in history_df.columns:
                                    display_cols.append('signal')
                                if 'Confidence (%)' in history_df.columns:
                                    display_cols.append('Confidence (%)')
                                
                                if display_cols:
                                    st.dataframe(
                                        history_df[display_cols].tail(10),
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                else:
                                    st.info("📈 History data is being processed...")
                            else:
                                st.info("📈 Building prediction history...")
                                
                        except Exception as history_error:
                            st.warning(f"⚠️ Unable to load prediction history: {str(history_error)}")
                            logger.error(f"Prediction history error: {history_error}")
                
                else:
                    st.error("❌ Unable to generate prediction. Please check model status.")
                    
            except Exception as e:
                st.error(f"❌ Error generating prediction: {str(e)}")
                logger.error(f"Live prediction page error: {e}")
                
                # Fallback to basic market data
                st.info("🔄 Attempting to show basic market data...")
                try:
                    live_data = data_loader.fetch_live_data(days_back=1)
                    if not live_data.empty:
                        current_price = live_data['Close'].iloc[-1]
                        price_change = live_data['Close'].iloc[-1] - live_data['Close'].iloc[-2] if len(live_data) > 1 else 0
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label="💰 Current Gold Price",
                                value=f"${current_price:.2f} USD",
                                delta=f"{price_change:+.2f} USD",
                                help="Latest available gold price"
                            )
                        with col2:
                            st.metric(
                                label="📅 Data Points",
                                value=f"{len(live_data)} records",
                                help="Available historical data points"
                            )
                except:
                    st.warning("⚠️ Unable to load any market data.")
                
                # Plot probability over time
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df['timestamp'],
                    y=history_df['probability'],
                    mode='lines+markers',
                    name='Prediction Probability'
                ))
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title="Prediction Probability Over Time",
                    yaxis_title="Probability",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No prediction history available")
        
        # Accuracy analysis
        st.subheader("📊 Recent Accuracy Analysis")
        
        accuracy_result = predictor.analyze_prediction_accuracy(days=30)
        
        if accuracy_result:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{accuracy_result['accuracy']:.1%}")
            
            with col2:
                st.metric("Precision", f"{accuracy_result['precision']:.1%}")
            
            with col3:
                st.metric("Recall", f"{accuracy_result['recall']:.1%}")
            
            with col4:
                st.metric("Total Predictions", accuracy_result['total_predictions'])
            
            st.info(f"Analysis based on last {accuracy_result['analysis_period_days']} days")
        else:
            st.warning("Unable to calculate accuracy metrics")
    
    elif page == "🔧 Model Management":
        st.title("🔧 Model Management")
        
        # Model information
        st.subheader("📋 Current Model Info")
        
        model_info = model_trainer.get_model_info()
        if model_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Training Date:** {model_info.get('training_date', 'Unknown')}")
                st.info(f"**Features:** {model_info.get('n_features', 'Unknown')}")
            
            with col2:
                st.info(f"**Training Months:** {model_info.get('training_months', 'Unknown')}")
                st.info(f"**Model Type:** XGBoost Classifier")
        else:
            st.warning("No model information available")
        
        # Manual operations
        st.subheader("🔧 Manual Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Update Data", help="Fetch latest market data"):
                with st.spinner("Updating data..."):
                    success = updater.update_historical_data()
                    if success:
                        st.success("Data updated successfully!")
                    else:
                        st.warning("No new data to update")
        
        with col2:
            if st.button("🔄 Retrain Model", help="Retrain model with latest data"):
                with st.spinner("Retraining model... This may take a few minutes."):
                    success = updater.force_retrain_now()
                    if success:
                        st.success("Model retrained successfully!")
                        st.cache_resource.clear()  # Clear cached components
                    else:
                        st.error("Model retraining failed")
        
        with col3:
            if st.button("🏃 Full Update", help="Update data and retrain model"):
                with st.spinner("Running full update..."):
                    result = updater.run_update_now()
                    
                    if result['data_updated']:
                        st.success("✅ Data updated")
                    
                    if result['model_retrained']:
                        st.success("✅ Model retrained")
                        st.cache_resource.clear()
                    
                    if result['errors']:
                        for error in result['errors']:
                            st.error(f"❌ {error}")
    
    elif page == "📋 System Status":
        st.title("📋 System Status")
        
        # Update status
        st.subheader("🔄 Update Status")
        
        status = updater.get_update_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Last Update:** {status['last_update'] or 'Never'}")
        
        with col2:
            st.info(f"**Last Retrain:** {status['last_retrain'] or 'Never'}")
        
        # Recent updates
        if status['recent_updates']:
            st.subheader("📝 Recent Update Log")
            
            updates_df = pd.DataFrame(status['recent_updates'])
            st.dataframe(updates_df, use_container_width=True)
        
        # File status
        st.subheader("📁 File Status")
        
        files_to_check = [
            ('Historical Data', config.get('data', {}).get('historical_file', 'data/gold_30min.csv')),
            ('Model File', config.get('model', {}).get('model_file', 'saved_models/gold_model_xgb.pkl')),
            ('Feature File', config.get('model', {}).get('feature_file', 'saved_models/feature_columns.pkl'))
        ]
        
        for file_name, file_path in files_to_check:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                st.success(f"✅ {file_name}: {file_size:,} bytes (Modified: {file_time.strftime('%Y-%m-%d %H:%M')})")
            else:
                st.error(f"❌ {file_name}: File not found")
    
    # Footer with timezone information
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🥇 Gold Trading AI Dashboard**")
    st.sidebar.markdown("Powered by XGBoost & Technical Analysis")
    
    # Get current time with timezone
    current_time, local_tz = get_current_time_with_tz()
    tz_info = get_data_source_timezones()
    
    st.sidebar.caption(f"Dashboard Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.caption(f"Local Timezone: {tz_info['System']}")

if __name__ == "__main__":
    main()
