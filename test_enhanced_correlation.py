"""
Test enhanced correlation system with multiple models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import traceback

def test_enhanced_correlations():
    """Test the enhanced correlation system"""
    print("🔬 Testing Enhanced Multi-Model Correlation System...")
    
    try:
        from macro_factors import MacroFactorsAnalyzer
        from data_loader import GoldDataLoader
        
        # Initialize components
        analyzer = MacroFactorsAnalyzer()
        loader = GoldDataLoader()
        
        print("  - Fetching gold data...")
        gold_data = loader.fetch_live_data(days_back=30)
        if gold_data.empty:
            gold_data = loader.load_historical_data().tail(30)
        
        gold_prices = gold_data['Close']
        print(f"  ✅ Gold data: {len(gold_prices)} price points")
        
        print("  - Fetching market factors...")
        market_data = analyzer.fetch_market_factors(period='1mo')
        print(f"  ✅ Market factors: {len(market_data)} factors")
        
        print("  - Fetching FRED economic data...")
        fred_data = analyzer.fetch_fred_indicators()
        print(f"  ✅ FRED data: {len(fred_data)} indicators")
        
        print("  - Calculating enhanced correlations...")
        analyzer.factor_data = {'market': market_data, 'economic': fred_data}
        correlations = analyzer.calculate_correlations_with_gold(gold_prices)
        
        print(f"  ✅ Enhanced correlations calculated: {len(correlations)} factors")
        
        if correlations:
            print("\n  📊 ENHANCED CORRELATION RESULTS:")
            print("  " + "="*80)
            
            # Sort by absolute correlation strength
            sorted_correlations = sorted(
                correlations.items(),
                key=lambda x: abs(x[1]['correlation']),
                reverse=True
            )
            
            for i, (symbol, data) in enumerate(sorted_correlations[:10], 1):
                factor_name = analyzer.factor_data.get('market', {}).get(symbol, {}).get('name') or \
                             analyzer.factor_data.get('economic', {}).get(symbol, {}).get('name', symbol)
                
                correlation = data['correlation']
                method = data['method']
                confidence = data['confidence']
                strength = data['strength']
                data_points = data['data_points']
                
                print(f"  {i:2d}. {factor_name[:30]:<30} | "
                      f"Corr: {correlation:+.3f} | "
                      f"Method: {method:<12} | "
                      f"Conf: {confidence:<6} | "
                      f"Strength: {strength:<8} | "
                      f"Points: {data_points}")
            
            print("  " + "="*80)
            
            # Test influence scores
            print("\n  - Testing influence score calculation...")
            analyzer.correlations = correlations
            influence_scores = analyzer.calculate_influence_scores()
            print(f"  ✅ Influence scores: {len(influence_scores)} factors")
            
            # Test top factors
            print("  - Testing top factors retrieval...")
            top_factors = analyzer.get_top_influencing_factors(10)
            print(f"  ✅ Top factors: {len(top_factors)} factors")
            
            if top_factors:
                print("\n  🏆 TOP 5 INFLUENCING FACTORS:")
                print("  " + "="*70)
                for i, factor in enumerate(top_factors[:5], 1):
                    correlation_val = correlations.get(factor['symbol'], {}).get('correlation', 0)
                    print(f"  {i}. {factor['name'][:25]:<25} | "
                          f"Score: {factor['influence_score']:5.1f} | "
                          f"Corr: {correlation_val:+.3f} | "
                          f"Impact: {factor['expected_impact']}")
                print("  " + "="*70)
            
            return True
        else:
            print("  ❌ No correlations calculated")
            return False
            
    except Exception as e:
        print(f"  ❌ Enhanced correlation test failed: {e}")
        traceback.print_exc()
        return False

def test_dashboard_integration():
    """Test dashboard integration with enhanced correlations"""
    print("\n🖥️ Testing Dashboard Integration...")
    
    try:
        from predictor import GoldPredictor
        
        predictor = GoldPredictor()
        
        print("  - Getting macro analysis with enhanced correlations...")
        macro_analysis = predictor.get_macro_analysis()
        
        if 'top_factors' in macro_analysis and macro_analysis['top_factors']:
            print(f"  ✅ Dashboard macro analysis: {len(macro_analysis['top_factors'])} factors")
            
            print("\n  📈 DASHBOARD TOP FACTORS:")
            for i, factor in enumerate(macro_analysis['top_factors'][:5], 1):
                correlation = factor.get('correlation', 0)
                influence = factor.get('influence_score', 0)
                print(f"  {i}. {factor['name'][:30]:<30} | "
                      f"Corr: {correlation:+.3f} | "
                      f"Influence: {influence:5.1f}")
            
            # Check if correlations are non-zero
            non_zero_correlations = [f for f in macro_analysis['top_factors'] 
                                   if abs(f.get('correlation', 0)) > 0.001]
            
            if non_zero_correlations:
                print(f"  ✅ Non-zero correlations: {len(non_zero_correlations)} factors")
                return True
            else:
                print("  ❌ All correlations are still zero in dashboard")
                return False
        else:
            print("  ❌ No top factors in dashboard macro analysis")
            return False
            
    except Exception as e:
        print(f"  ❌ Dashboard integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run enhanced correlation tests"""
    print("🎯 ENHANCED CORRELATION SYSTEM TEST")
    print("=" * 60)
    
    results = {}
    
    # Test enhanced correlations
    results['enhanced_correlations'] = test_enhanced_correlations()
    results['dashboard_integration'] = test_dashboard_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ENHANCED CORRELATION TEST SUMMARY")
    print("=" * 60)
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component.replace('_', ' ').title()}: {'WORKING' if status else 'FAILED'}")
    
    working_components = sum(results.values())
    total_components = len(results)
    
    print(f"\n🎯 Overall Status: {working_components}/{total_components} components working")
    
    if working_components == total_components:
        print("🎉 Enhanced correlation system fully operational!")
        print("📈 Dashboard should now show accurate correlation values!")
    else:
        print("⚠️ Issues detected - correlations may still show as zero")
    
    return results

if __name__ == "__main__":
    main()
