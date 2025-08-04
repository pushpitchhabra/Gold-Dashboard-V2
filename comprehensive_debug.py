#!/usr/bin/env python3
"""
Comprehensive Debug Script for Gold AI Dashboard
Tests all components and identifies issues with live data flow
"""

import sys
import os
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_yfinance_connection():
    """Test direct yfinance connection for gold data"""
    print("\n" + "="*60)
    print("🔍 TESTING YFINANCE CONNECTION")
    print("="*60)
    
    try:
        import yfinance as yf
        
        # Test gold futures symbol
        gold_symbol = "GC=F"
        print(f"📊 Testing symbol: {gold_symbol}")
        
        ticker = yf.Ticker(gold_symbol)
        
        # Test recent data (last 7 days)
        print("📈 Fetching recent 7-day data...")
        recent_data = ticker.history(period="7d", interval="30m")
        
        if not recent_data.empty:
            print(f"✅ SUCCESS: Got {len(recent_data)} records")
            print(f"📅 Date range: {recent_data.index.min()} to {recent_data.index.max()}")
            print(f"💰 Latest price: ${recent_data['Close'].iloc[-1]:.2f}")
            print(f"📊 Columns: {list(recent_data.columns)}")
            
            # Test data quality
            null_count = recent_data.isnull().sum().sum()
            print(f"🔍 Null values: {null_count}")
            
            if null_count > 0:
                print("⚠️  WARNING: Data contains null values")
                print(recent_data.isnull().sum())
        else:
            print("❌ ERROR: No data received from yfinance")
            return False
            
        # Test alternative symbols
        alt_symbols = ["GOLD", "IAU", "GLD"]
        print(f"\n🔄 Testing alternative symbols: {alt_symbols}")
        
        for symbol in alt_symbols:
            try:
                alt_ticker = yf.Ticker(symbol)
                alt_data = alt_ticker.history(period="5d")
                if not alt_data.empty:
                    print(f"✅ {symbol}: ${alt_data['Close'].iloc[-1]:.2f}")
                else:
                    print(f"❌ {symbol}: No data")
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")
        
        return True
        
    except ImportError:
        print("❌ ERROR: yfinance not installed")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

def test_data_loader():
    """Test the data loader component"""
    print("\n" + "="*60)
    print("🔍 TESTING DATA LOADER")
    print("="*60)
    
    try:
        from data_loader import GoldDataLoader
        
        loader = GoldDataLoader()
        print(f"📊 Symbol: {loader.symbol}")
        print(f"⏰ Interval: {loader.interval}")
        print(f"🌍 Timezone: {loader.timezone}")
        
        # Test live data fetch
        print("\n📈 Testing live data fetch...")
        live_data = loader.fetch_live_data(days_back=7)
        
        if not live_data.empty:
            print(f"✅ SUCCESS: Got {len(live_data)} live records")
            print(f"📅 Date range: {live_data.index.min()} to {live_data.index.max()}")
            print(f"💰 Latest price: ${live_data['Close'].iloc[-1]:.2f}")
            
            # Check data quality
            print(f"🔍 Data quality check:")
            print(f"   - Null values: {live_data.isnull().sum().sum()}")
            print(f"   - Duplicate timestamps: {live_data.index.duplicated().sum()}")
            print(f"   - Price range: ${live_data['Close'].min():.2f} - ${live_data['Close'].max():.2f}")
        else:
            print("❌ ERROR: No live data available")
            return False
        
        # Test latest price
        print("\n💰 Testing latest price...")
        price, timestamp = loader.get_latest_price()
        
        if price:
            print(f"✅ Latest price: ${price:.2f} at {timestamp}")
        else:
            print("❌ ERROR: Could not get latest price")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

def test_feature_engineer():
    """Test the feature engineering component"""
    print("\n" + "="*60)
    print("🔍 TESTING FEATURE ENGINEER")
    print("="*60)
    
    try:
        from feature_engineer import GoldFeatureEngineer
        from data_loader import GoldDataLoader
        
        loader = GoldDataLoader()
        engineer = GoldFeatureEngineer()
        
        # Get some data using smart data access
        data, data_source = loader.get_data_for_analysis(days_back=30)
        
        if data.empty:
            print("❌ ERROR: No data available for feature engineering")
            return False
        
        print(f"📊 Data source: {data_source}")
        
        print(f"📊 Input data: {len(data)} records")
        
        # Test feature calculation
        print("\n🔧 Testing feature calculation...")
        features, feature_columns = engineer.get_latest_features(data)
        
        if features is not None and not features.empty:
            print(f"✅ SUCCESS: Generated {len(feature_columns)} features")
            print(f"📊 Feature shape: {features.shape}")
            print(f"🔍 Features: {feature_columns[:10]}...")  # Show first 10
            
            # Check for null values
            null_count = features.isnull().sum().sum()
            print(f"🔍 Null features: {null_count}")
            
            if null_count > 0:
                print("⚠️  WARNING: Some features contain null values")
        else:
            print("❌ ERROR: Could not generate features")
            return False
        
        # Test indicator summary
        print("\n📈 Testing indicator summary...")
        indicators = engineer.get_indicator_summary(data)
        
        if indicators:
            print(f"✅ SUCCESS: Generated {len(indicators)} indicators")
            for name, value in list(indicators.items())[:5]:
                print(f"   - {name}: {value:.4f}")
        else:
            print("❌ ERROR: Could not generate indicators")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

def test_model_trainer():
    """Test the model trainer component"""
    print("\n" + "="*60)
    print("🔍 TESTING MODEL TRAINER")
    print("="*60)
    
    try:
        from model_trainer import GoldModelTrainer
        
        trainer = GoldModelTrainer()
        
        # Test model loading
        print("🤖 Testing model loading...")
        model, feature_columns, metadata = trainer.load_model()
        
        if model is not None:
            print(f"✅ SUCCESS: Model loaded")
            print(f"📊 Features: {len(feature_columns) if feature_columns else 0}")
            print(f"📋 Metadata: {metadata}")
        else:
            print("⚠️  WARNING: No trained model found")
            print("🔄 This is normal for first-time setup")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

def test_predictor():
    """Test the predictor component"""
    print("\n" + "="*60)
    print("🔍 TESTING PREDICTOR")
    print("="*60)
    
    try:
        from predictor import GoldPredictor
        
        predictor = GoldPredictor()
        
        # Test live prediction
        print("🎯 Testing live prediction...")
        prediction = predictor.get_live_prediction()
        
        if prediction:
            print(f"✅ SUCCESS: Got prediction")
            print(f"🎯 Signal: {prediction.get('signal', 'N/A')}")
            print(f"🎲 Confidence: {prediction.get('confidence', 'N/A')}")
            print(f"💰 Price: ${prediction.get('current_price', 0):.2f}")
            print(f"📊 Data quality: {prediction.get('data_quality', 'N/A')}")
            
            # Check indicators
            indicators = prediction.get('indicators', {})
            if indicators:
                print(f"📈 Indicators: {len(indicators)} available")
            else:
                print("⚠️  WARNING: No indicators in prediction")
        else:
            print("❌ ERROR: Could not get prediction")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

def test_macro_factors():
    """Test the macro factors component"""
    print("\n" + "="*60)
    print("🔍 TESTING MACRO FACTORS")
    print("="*60)
    
    try:
        from macro_factors import MacroFactorsAnalyzer
        
        analyzer = MacroFactorsAnalyzer()
        
        # Test FRED API connection
        print("🏛️ Testing FRED API connection...")
        
        # Check if API key is configured
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        fred_key = config.get('api_keys', {}).get('fred_key')
        if fred_key:
            print(f"✅ FRED API key configured: {fred_key[:10]}...")
        else:
            print("❌ ERROR: FRED API key not configured")
            return False
        
        # Test macro data fetch
        print("📊 Testing macro data fetch...")
        macro_data = analyzer.get_dashboard_data()
        
        if macro_data and 'factors' in macro_data:
            factors = macro_data['factors']
            print(f"✅ SUCCESS: Got {len(factors)} macro factors")
            
            # Show sample factors
            for i, factor in enumerate(factors[:3]):
                name = factor.get('name', 'Unknown')
                value = factor.get('current_value', 'N/A')
                print(f"   - {name}: {value}")
        else:
            print("❌ ERROR: Could not get macro factors data")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

def test_strategy_analyzer():
    """Test the strategy analyzer component"""
    print("\n" + "="*60)
    print("🔍 TESTING STRATEGY ANALYZER")
    print("="*60)
    
    try:
        from strategy_analyzer import AdvancedStrategyAnalyzer
        from data_loader import GoldDataLoader
        
        analyzer = AdvancedStrategyAnalyzer()
        loader = GoldDataLoader()
        
        # Get some data using smart data access
        data, data_source = loader.get_data_for_analysis(days_back=30)
        
        if data.empty:
            print("❌ ERROR: No data available for strategy analysis")
            return False
        
        print(f"📊 Data source: {data_source}")
        
        print(f"📊 Input data: {len(data)} records")
        
        # Test strategy analysis
        print("📈 Testing strategy analysis...")
        analysis = analyzer.analyze_current_setup(data)
        
        if analysis:
            print(f"✅ SUCCESS: Got strategy analysis")
            print(f"🎯 Direction: {analysis.get('direction', 'N/A')}")
            print(f"📊 Setup grade: {analysis.get('setup_grade', 'N/A')}")
            print(f"💰 Entry price: ${analysis.get('entry_price', 0):.2f}")
        else:
            print("❌ ERROR: Could not get strategy analysis")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

def test_dashboard_integration():
    """Test dashboard integration and data flow"""
    print("\n" + "="*60)
    print("🔍 TESTING DASHBOARD INTEGRATION")
    print("="*60)
    
    try:
        # Test if all required files exist
        required_files = [
            'app.py',
            'config.yaml',
            'data_loader.py',
            'feature_engineer.py',
            'model_trainer.py',
            'predictor.py',
            'macro_factors.py',
            'strategy_analyzer.py'
        ]
        
        print("📁 Checking required files...")
        missing_files = []
        for file in required_files:
            if os.path.exists(file):
                print(f"✅ {file}")
            else:
                print(f"❌ {file} - MISSING")
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ ERROR: Missing files: {missing_files}")
            return False
        
        # Test data directory
        print("\n📁 Checking data directory...")
        os.makedirs('data', exist_ok=True)
        os.makedirs('saved_models', exist_ok=True)
        print("✅ Data directories created/verified")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive debugging"""
    print("🚀 GOLD AI DASHBOARD - COMPREHENSIVE DEBUG")
    print("=" * 80)
    print(f"🕐 Started at: {datetime.now()}")
    print("=" * 80)
    
    # Track test results
    tests = [
        ("YFinance Connection", test_yfinance_connection),
        ("Data Loader", test_data_loader),
        ("Feature Engineer", test_feature_engineer),
        ("Model Trainer", test_model_trainer),
        ("Predictor", test_predictor),
        ("Macro Factors", test_macro_factors),
        ("Strategy Analyzer", test_strategy_analyzer),
        ("Dashboard Integration", test_dashboard_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ CRITICAL ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Dashboard should work properly!")
    else:
        print("⚠️  ISSUES FOUND - Dashboard may have problems")
        print("\n🔧 RECOMMENDED ACTIONS:")
        
        for test_name, result in results.items():
            if not result:
                print(f"   - Fix {test_name}")
    
    print(f"\n🕐 Completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
