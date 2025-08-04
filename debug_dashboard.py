#!/usr/bin/env python3
"""
Debug script to test the exact same flow that the dashboard uses
"""

import sys
import pandas as pd
from datetime import datetime

def debug_dashboard_flow():
    """Test the exact same flow that the dashboard uses for macro factors"""
    
    print("🔍 Debugging Dashboard Macro Factors Flow")
    print("="*60)
    
    try:
        # Import the same modules as the dashboard
        from data_loader import GoldDataLoader
        from macro_factors import MacroFactorsAnalyzer
        
        print("✅ Modules imported successfully")
        
        # Initialize components (same as dashboard)
        data_loader = GoldDataLoader()
        macro_analyzer = MacroFactorsAnalyzer()
        
        print("✅ Components initialized")
        
        # Load historical gold data (same as dashboard)
        print("\n📊 Loading historical gold data...")
        gold_data = data_loader.load_historical_data()
        
        if not gold_data.empty:
            gold_prices = gold_data['Close']
            print(f"✅ Gold data loaded: {len(gold_prices)} records")
            print(f"   Latest price: ${gold_prices.iloc[-1]:.2f}")
        else:
            gold_prices = None
            print("⚠️ No gold data available")
        
        # Update macro factors (same as dashboard)
        print("\n🌍 Updating macro factors...")
        try:
            macro_summary = macro_analyzer.update_all_factors(gold_prices)
            print("✅ Macro factors updated successfully")
            
            # Get dashboard data (same as dashboard)
            dashboard_data = macro_analyzer.get_macro_dashboard_summary()
            print("✅ Dashboard data retrieved")
            
            # Print summary statistics
            print(f"\n📈 Summary Statistics:")
            print(f"   Overall Sentiment: {macro_summary.get('overall_sentiment', 'Unknown')}")
            print(f"   Top Factors Count: {len(macro_summary.get('top_factors', []))}")
            print(f"   Bullish Factors: {len(macro_summary.get('bullish_factors', []))}")
            print(f"   Bearish Factors: {len(macro_summary.get('bearish_factors', []))}")
            
            # Check dashboard data structure
            categories = dashboard_data.get('categories', {})
            print(f"   Categories: {len(categories)}")
            print(f"   Total Indicators: {dashboard_data.get('total_indicators', 0)}")
            
            # Show category breakdown
            if categories:
                print(f"\n📊 Category Breakdown:")
                for category, factors in categories.items():
                    valid_count = len([f for f in factors if f.get('current_value') is not None])
                    print(f"   {category}: {valid_count}/{len(factors)} valid")
                    
                    # Show a sample factor from each category
                    if factors and valid_count > 0:
                        sample = next(f for f in factors if f.get('current_value') is not None)
                        print(f"      Sample: {sample.get('name', 'Unknown')} = {sample.get('current_value', 'N/A')}")
            
            # Test if this is the same data the dashboard should see
            print(f"\n🎯 Dashboard Data Check:")
            print(f"   Data structure looks correct: {bool(categories and dashboard_data.get('total_indicators', 0) > 0)}")
            
            return True, dashboard_data
            
        except Exception as e:
            print(f"❌ Error in macro factors update: {e}")
            import traceback
            traceback.print_exc()
            return False, None
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def check_dashboard_config():
    """Check if dashboard config loading matches our test"""
    
    print("\n🔧 Checking Dashboard Config Loading")
    print("="*40)
    
    try:
        import yaml
        
        # Load config the same way as dashboard
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        fred_key = config.get('api_keys', {}).get('fred_key', '')
        
        print(f"✅ Config loaded successfully")
        print(f"🔑 FRED Key: {fred_key[:10]}...{fred_key[-4:] if len(fred_key) > 14 else fred_key}")
        print(f"🌍 Macro factors enabled: {config.get('macro_factors', {}).get('enabled', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Dashboard Flow Debug Test")
    print("="*60)
    
    # Test 1: Config loading
    config_ok = check_dashboard_config()
    
    if config_ok:
        # Test 2: Full dashboard flow
        success, data = debug_dashboard_flow()
        
        if success and data:
            print(f"\n🎉 SUCCESS: Dashboard flow works correctly!")
            print(f"   The macro factors data should be visible in the dashboard.")
            print(f"   If it's not showing, there might be a Streamlit caching or display issue.")
        else:
            print(f"\n❌ FAILED: Dashboard flow has issues.")
            print(f"   This explains why the dashboard shows 'No macro factors data available'.")
    else:
        print(f"\n❌ FAILED: Config loading issues.")
    
    print("\n" + "="*60)
    print("Debug test completed.")
