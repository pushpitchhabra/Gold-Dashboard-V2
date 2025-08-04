"""
Test script to verify macro factors functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from macro_factors import MacroFactorsAnalyzer
import pandas as pd

def test_macro_factors():
    print("🧪 Testing Macro Factors Analyzer...")
    
    # Initialize analyzer
    analyzer = MacroFactorsAnalyzer()
    
    # Test market factors fetch
    print("\n📊 Testing Market Factors...")
    market_data = analyzer.fetch_market_factors(period='5d')
    
    print(f"✅ Fetched {len(market_data)} market factors")
    for symbol, data in list(market_data.items())[:3]:  # Show first 3
        if 'error' not in data:
            print(f"  - {data['name']}: ${data['current_value']:.2f} ({data['change_pct']:+.2f}%)")
        else:
            print(f"  - {data['name']}: ERROR - {data['error']}")
    
    # Test FRED indicators
    print("\n🏛️ Testing FRED Economic Indicators...")
    fred_data = analyzer.fetch_fred_indicators()
    
    print(f"✅ Fetched {len(fred_data)} FRED indicators")
    for symbol, data in list(fred_data.items())[:3]:  # Show first 3
        if 'error' not in data:
            print(f"  - {data['name']}: {data['current_value']:.2f} ({data['change_pct']:+.2f}%)")
        else:
            print(f"  - {data['name']}: ERROR - {data['error']}")
    
    # Test full analysis with sample gold data
    print("\n🥇 Testing Full Analysis...")
    sample_gold_data = pd.Series([2000, 2010, 2005, 2020, 2015, 2025, 2030, 2018] * 5)
    
    summary = analyzer.update_all_factors(sample_gold_data)
    
    print(f"✅ Overall Sentiment: {summary['overall_sentiment']}")
    print(f"✅ Bullish Score: {summary['bullish_score']:.1f}")
    print(f"✅ Bearish Score: {summary['bearish_score']:.1f}")
    print(f"✅ Top Factors: {len(summary['top_factors'])}")
    
    if summary['top_factors']:
        print("\n🔝 Top 3 Influencing Factors:")
        for i, factor in enumerate(summary['top_factors'][:3], 1):
            print(f"  {i}. {factor['name']}: {factor['influence_score']:.1f} ({factor['expected_impact']})")
    
    print(f"\n📝 Summary: {summary['summary']}")
    
    return summary

if __name__ == "__main__":
    try:
        result = test_macro_factors()
        print("\n✅ Macro factors test completed successfully!")
        print(f"Total factors analyzed: {len(result.get('top_factors', []))}")
    except Exception as e:
        print(f"\n❌ Macro factors test failed: {e}")
        import traceback
        traceback.print_exc()
