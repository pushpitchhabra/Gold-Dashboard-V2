"""
Comprehensive diagnostic script to identify and fix dashboard issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import traceback

def test_data_loading():
    """Test data loading functionality"""
    print("🔍 Testing Data Loading...")
    try:
        from data_loader import GoldDataLoader
        loader = GoldDataLoader()
        
        # Test historical data
        print("  - Loading historical data...")
        historical_data = loader.load_historical_data()
        print(f"  ✅ Historical data: {len(historical_data)} records")
        
        # Test live data
        print("  - Fetching live data...")
        live_data = loader.fetch_live_data(days_back=5)
        print(f"  ✅ Live data: {len(live_data)} records")
        
        return True, historical_data, live_data
        
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        traceback.print_exc()
        return False, None, None

def test_feature_engineering(data):
    """Test feature engineering"""
    print("\n🔍 Testing Feature Engineering...")
    try:
        from feature_engineer import GoldFeatureEngineer
        engineer = GoldFeatureEngineer()
        
        if data is None or data.empty:
            print("  ❌ No data available for feature engineering")
            return False, None
        
        # Add technical indicators
        print("  - Adding technical indicators...")
        data_with_features = engineer.add_technical_indicators(data.copy())
        print(f"  ✅ Features added: {len(data_with_features.columns)} columns")
        
        # Test feature preparation
        print("  - Preparing features for ML...")
        features, target = engineer.prepare_features_for_training(data_with_features)
        print(f"  ✅ ML features: {len(features.columns)} features, {len(target)} target values")
        
        return True, data_with_features
        
    except Exception as e:
        print(f"  ❌ Feature engineering failed: {e}")
        traceback.print_exc()
        return False, None

def test_model_training(features_data):
    """Test model training and prediction"""
    print("\n🔍 Testing Model Training & Prediction...")
    try:
        from model_trainer import GoldModelTrainer
        trainer = GoldModelTrainer()
        
        if features_data is None or features_data.empty:
            print("  ❌ No features data available for model training")
            return False
        
        # Check if model exists
        model_path = 'saved_models/gold_model.joblib'
        if os.path.exists(model_path):
            print("  ✅ Model file exists")
            
            # Test model loading and prediction
            print("  - Testing model prediction...")
            model, feature_columns, metadata = trainer.load_model()
            
            if model is not None:
                print(f"  ✅ Model loaded with {len(feature_columns)} expected features")
                
                # Test prediction on sample data
                sample_data = features_data.tail(1)
                prediction = trainer.predict(sample_data)
                print(f"  ✅ Sample prediction: {prediction}")
                
                return True
            else:
                print("  ❌ Model failed to load")
                return False
        else:
            print("  ⚠️ Model file not found - training new model...")
            # Try to train a new model
            from feature_engineer import GoldFeatureEngineer
            engineer = GoldFeatureEngineer()
            
            features, target = engineer.prepare_features_for_training(features_data)
            
            if len(features) > 100:  # Need sufficient data
                result = trainer.train_model(features, target)
                accuracy = result['accuracy']
                print(f"  ✅ New model trained with accuracy: {accuracy:.3f}")
                return True
            else:
                print(f"  ❌ Insufficient data for training: {len(features)} samples")
                return False
        
    except Exception as e:
        print(f"  ❌ Model training/prediction failed: {e}")
        traceback.print_exc()
        return False

def test_predictor():
    """Test predictor functionality"""
    print("\n🔍 Testing Predictor...")
    try:
        from predictor import GoldPredictor
        predictor = GoldPredictor()
        
        # Test basic prediction
        print("  - Testing live prediction...")
        prediction = predictor.get_live_prediction()
        print(f"  ✅ Live prediction: {prediction.get('signal', 'Unknown')} ({prediction.get('probability', 0):.3f})")
        
        # Test comprehensive recommendation
        print("  - Testing comprehensive recommendation...")
        comprehensive = predictor.get_comprehensive_trading_recommendation()
        print(f"  ✅ Comprehensive rec: {comprehensive.get('direction', 'Unknown')} (Grade: {comprehensive.get('setup_grade', 'Unknown')})")
        
        # Test macro analysis
        print("  - Testing macro analysis...")
        macro = predictor.get_macro_analysis()
        print(f"  ✅ Macro analysis: {macro.get('overall_sentiment', 'Unknown')} sentiment")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Predictor failed: {e}")
        traceback.print_exc()
        return False

def test_strategy_analyzer():
    """Test strategy analyzer"""
    print("\n🔍 Testing Strategy Analyzer...")
    try:
        from strategy_analyzer import AdvancedStrategyAnalyzer
        analyzer = AdvancedStrategyAnalyzer()
        
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'Open': [2000, 2010, 2005],
            'High': [2020, 2025, 2015],
            'Low': [1995, 2005, 2000],
            'Close': [2015, 2020, 2010],
            'Volume': [1000, 1100, 900]
        })
        
        # Test price action analysis
        print("  - Testing price action analysis...")
        price_action = analyzer.analyze_price_action(sample_data)
        print(f"  ✅ Price action analysis completed")
        
        # Test trading recommendation
        print("  - Testing trading recommendation...")
        ml_prediction = {'prediction': 1, 'probability': 0.75, 'signal': 'BUY'}
        technical_indicators = {'RSI': 45, 'MACD': 0.5}
        
        recommendation = analyzer.generate_trading_recommendation(
            ml_prediction=ml_prediction,
            technical_indicators=technical_indicators,
            current_price=2010
        )
        print(f"  ✅ Trading recommendation: {recommendation.get('direction', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Strategy analyzer failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive diagnostics"""
    print("🏥 COMPREHENSIVE DASHBOARD DIAGNOSTICS")
    print("=" * 50)
    
    results = {}
    
    # Test each component
    results['data_loading'], historical_data, live_data = test_data_loading()
    
    if results['data_loading']:
        results['feature_engineering'], features_data = test_feature_engineering(historical_data)
        
        if results['feature_engineering']:
            results['model_training'] = test_model_training(features_data)
        else:
            results['model_training'] = False
    else:
        results['feature_engineering'] = False
        results['model_training'] = False
    
    results['predictor'] = test_predictor()
    results['strategy_analyzer'] = test_strategy_analyzer()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component.replace('_', ' ').title()}: {'WORKING' if status else 'FAILED'}")
    
    working_components = sum(results.values())
    total_components = len(results)
    
    print(f"\n🎯 Overall Status: {working_components}/{total_components} components working")
    
    if working_components == total_components:
        print("🎉 All systems operational!")
    else:
        print("⚠️ Issues detected - see details above")
    
    return results

if __name__ == "__main__":
    main()
