"""
Targeted diagnostic script for model retraining and macro correlation issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import traceback

def test_model_retraining():
    """Test the complete model retraining workflow"""
    print("🔄 Testing Model Retraining Workflow...")
    
    try:
        from data_loader import GoldDataLoader
        from feature_engineer import GoldFeatureEngineer
        from model_trainer import GoldModelTrainer
        
        # Load components
        loader = GoldDataLoader()
        engineer = GoldFeatureEngineer()
        trainer = GoldModelTrainer()
        
        print("  - Loading historical data...")
        historical_data = loader.load_historical_data()
        print(f"  ✅ Loaded {len(historical_data)} historical records")
        
        print("  - Preparing features for training...")
        features, target = engineer.prepare_features_for_training(historical_data)
        print(f"  ✅ Features prepared: {features.shape[0]} samples, {features.shape[1]} features")
        
        print("  - Training new model...")
        training_result = trainer.train_model(features, target)
        print(f"  ✅ Model trained with accuracy: {training_result['accuracy']:.3f}")
        
        print("  - Saving model...")
        trainer.save_model(features.columns.tolist())
        print("  ✅ Model saved successfully")
        
        print("  - Testing model loading...")
        loaded_model, feature_columns, metadata = trainer.load_model()
        if loaded_model is not None:
            print(f"  ✅ Model loaded with {len(feature_columns)} features")
            return True
        else:
            print("  ❌ Model failed to load after saving")
            return False
            
    except Exception as e:
        print(f"  ❌ Model retraining failed: {e}")
        traceback.print_exc()
        return False

def test_macro_correlation():
    """Test macro factors correlation functionality"""
    print("\n📊 Testing Macro Factors Correlation...")
    
    try:
        from macro_factors import MacroFactorsAnalyzer
        from data_loader import GoldDataLoader
        
        # Initialize components
        analyzer = MacroFactorsAnalyzer()
        loader = GoldDataLoader()
        
        print("  - Fetching gold data for correlation...")
        gold_data = loader.fetch_live_data(days_back=30)
        if gold_data.empty:
            print("  ⚠️ No live gold data, using historical...")
            gold_data = loader.load_historical_data().tail(30)
        
        gold_prices = gold_data['Close']
        print(f"  ✅ Gold data: {len(gold_prices)} price points")
        
        print("  - Fetching market factors...")
        market_data = analyzer.fetch_market_factors(period='1mo')
        print(f"  ✅ Market factors: {len(market_data)} factors fetched")
        
        print("  - Testing correlation calculation...")
        analyzer.factor_data = {'market': market_data}
        correlations = analyzer.calculate_correlations_with_gold(gold_prices)
        print(f"  ✅ Correlations calculated: {len(correlations)} factors")
        
        # Display sample correlations
        if correlations:
            print("  📈 Sample correlations:")
            for symbol, corr_data in list(correlations.items())[:3]:
                factor_name = market_data.get(symbol, {}).get('name', symbol)
                correlation = corr_data.get('correlation', 0)
                strength = corr_data.get('strength', 'Unknown')
                print(f"    - {factor_name}: {correlation:.3f} ({strength})")
        
        print("  - Testing influence scores...")
        analyzer.correlations = correlations
        influence_scores = analyzer.calculate_influence_scores()
        print(f"  ✅ Influence scores: {len(influence_scores)} factors scored")
        
        print("  - Testing top factors...")
        top_factors = analyzer.get_top_influencing_factors(5)
        print(f"  ✅ Top factors: {len(top_factors)} factors identified")
        
        if top_factors:
            print("  🔝 Top 3 influencing factors:")
            for i, factor in enumerate(top_factors[:3], 1):
                print(f"    {i}. {factor['name']}: {factor['influence_score']:.1f} ({factor['expected_impact']})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Macro correlation failed: {e}")
        traceback.print_exc()
        return False

def test_dashboard_integration():
    """Test dashboard integration of these features"""
    print("\n🖥️ Testing Dashboard Integration...")
    
    try:
        from predictor import GoldPredictor
        
        predictor = GoldPredictor()
        
        print("  - Testing enhanced comprehensive recommendation...")
        enhanced_rec = predictor.get_enhanced_comprehensive_recommendation()
        
        # Check if macro analysis is included
        if 'macro_analysis' in enhanced_rec:
            macro_data = enhanced_rec['macro_analysis']
            print(f"  ✅ Macro analysis included: {macro_data.get('overall_sentiment', 'Unknown')} sentiment")
            
            top_factors = macro_data.get('top_factors', [])
            if top_factors:
                print(f"  ✅ Top factors available: {len(top_factors)} factors")
            else:
                print("  ⚠️ No top factors in macro analysis")
        else:
            print("  ❌ Macro analysis missing from comprehensive recommendation")
            return False
        
        print("  - Testing standalone macro analysis...")
        macro_analysis = predictor.get_macro_analysis()
        
        if macro_analysis.get('top_factors'):
            print(f"  ✅ Standalone macro analysis working: {len(macro_analysis['top_factors'])} factors")
        else:
            print("  ❌ Standalone macro analysis not working properly")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Dashboard integration failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run targeted diagnostics"""
    print("🎯 TARGETED DIAGNOSTICS: RETRAIN & CORRELATION")
    print("=" * 60)
    
    results = {}
    
    # Test each component
    results['model_retraining'] = test_model_retraining()
    results['macro_correlation'] = test_macro_correlation()
    results['dashboard_integration'] = test_dashboard_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component.replace('_', ' ').title()}: {'WORKING' if status else 'FAILED'}")
    
    working_components = sum(results.values())
    total_components = len(results)
    
    print(f"\n🎯 Overall Status: {working_components}/{total_components} components working")
    
    if working_components == total_components:
        print("🎉 All targeted systems operational!")
    else:
        print("⚠️ Issues detected - see details above")
        
        # Provide specific recommendations
        if not results['model_retraining']:
            print("\n🔧 Model Retraining Fixes Needed:")
            print("  - Check model saving/loading paths")
            print("  - Verify feature column consistency")
            print("  - Ensure proper error handling")
        
        if not results['macro_correlation']:
            print("\n🔧 Macro Correlation Fixes Needed:")
            print("  - Verify data alignment between gold and factors")
            print("  - Check correlation calculation logic")
            print("  - Ensure influence scoring works properly")
    
    return results

if __name__ == "__main__":
    main()
