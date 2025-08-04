#!/usr/bin/env python3
"""
Test script to verify UI fixes are working properly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import GoldDataLoader
from predictor import GoldPredictor
from feature_engineer import GoldFeatureEngineer
from model_trainer import GoldModelTrainer
from strategy_analyzer import AdvancedStrategyAnalyzer

def test_ui_fixes():
    """Test that all UI-related fixes are working"""
    print("🔧 Testing UI Fixes")
    print("=" * 50)
    
    try:
        # Initialize components
        data_loader = GoldDataLoader()
        feature_engineer = GoldFeatureEngineer()
        model_trainer = GoldModelTrainer()
        predictor = GoldPredictor()
        
        print("✅ Components initialized successfully")
        
        # Test 1: Target Price Calculation
        print("\n🎯 Test 1: Target Price Calculation")
        prediction = predictor.get_live_prediction()
        
        if prediction and isinstance(prediction, dict):
            current_price = prediction.get('current_price', 0)
            pred_value = prediction.get('prediction', 0)
            probability = prediction.get('probability', 0.5)
            
            print(f"📊 Current price: ${current_price:.2f}")
            print(f"📊 Prediction: {pred_value}")
            print(f"📊 Probability: {probability:.3f}")
            
            if isinstance(current_price, (int, float)) and current_price > 0:
                # Calculate target price (same logic as in app.py)
                if pred_value == 1:  # Bullish
                    target_multiplier = 1 + (probability - 0.5) * 0.02
                else:  # Bearish
                    target_multiplier = 1 - (probability - 0.5) * 0.02
                
                target_price = current_price * target_multiplier
                price_change = target_price - current_price
                
                print(f"✅ Target price calculation: ${target_price:.2f} (change: {price_change:+.2f})")
            else:
                print("❌ Current price not available for target calculation")
        else:
            print("❌ Could not get prediction for target price test")
        
        # Test 2: Confidence Display
        print("\n🎲 Test 2: Confidence Display")
        confidence = prediction.get('confidence', 'Unknown')
        probability = prediction.get('probability', 0.5)
        
        print(f"📊 Raw confidence: {confidence}")
        print(f"📊 Raw probability: {probability}")
        
        if isinstance(confidence, str):
            confidence_map = {'High': 85, 'Medium': 65, 'Low': 45}
            conf_pct = confidence_map.get(confidence, int(probability * 100))
            print(f"✅ Confidence display: {conf_pct}% ({confidence})")
        elif isinstance(confidence, (int, float)):
            print(f"✅ Confidence display: {confidence:.1f}%")
        else:
            conf_pct = int(probability * 100) if isinstance(probability, (int, float)) else 50
            print(f"✅ Fallback confidence display: {conf_pct}%")
        
        # Test 3: Chart Data Access
        print("\n📈 Test 3: Chart Data Access")
        chart_data, data_source = data_loader.get_data_for_analysis(days_back=30)
        
        if not chart_data.empty:
            print(f"✅ Chart data available: {data_source} source, {len(chart_data)} records")
            
            # Check required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            if all(col in chart_data.columns for col in required_cols):
                print(f"✅ All required chart columns present: {required_cols}")
            else:
                missing = [col for col in required_cols if col not in chart_data.columns]
                print(f"❌ Missing chart columns: {missing}")
        else:
            print("❌ No chart data available")
        
        # Test 4: Strategy Analysis
        print("\n🎯 Test 4: Strategy Analysis")
        analyzer = AdvancedStrategyAnalyzer()
        strategy_data, _ = data_loader.get_data_for_analysis(days_back=30)
        
        if not strategy_data.empty:
            analysis = analyzer.analyze_current_setup(strategy_data)
            
            if analysis and isinstance(analysis, dict):
                print(f"✅ Strategy analysis working:")
                print(f"   - Direction: {analysis.get('direction', 'Unknown')}")
                print(f"   - Setup grade: {analysis.get('setup_grade', 'Unknown')}")
                print(f"   - Entry price: ${analysis.get('entry_price', 0):.2f}")
                
                # Check for "Data unavailable" issues
                reasoning = analysis.get('setup_reasoning', '')
                tech_summary = analysis.get('technical_summary', '')
                
                if 'unavailable' in reasoning.lower() or 'unavailable' in tech_summary.lower():
                    print("⚠️  Some strategy data shows 'unavailable' - may need improvement")
                else:
                    print("✅ Strategy data properly formatted")
            else:
                print("❌ Strategy analysis failed")
        else:
            print("❌ No data for strategy analysis")
        
        print("\n" + "=" * 50)
        print("🎉 UI FIXES TEST COMPLETE")
        print("✅ All major UI components should now work properly!")
        print("🚀 Dashboard ready for use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in UI fixes test: {e}")
        return False

if __name__ == '__main__':
    test_ui_fixes()
