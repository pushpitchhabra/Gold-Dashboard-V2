#!/usr/bin/env python3
"""
Test script to verify FRED API connection and debug macro factors
"""

import requests
import yaml
import sys
from datetime import datetime, timedelta

def test_fred_api():
    """Test FRED API connection with the configured key"""
    
    # Load config
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        print("✅ Config file loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False
    
    # Get FRED API key
    fred_key = config.get('api_keys', {}).get('fred_key', '')
    print(f"🔑 FRED API Key: {fred_key[:10]}...{fred_key[-4:] if len(fred_key) > 14 else fred_key}")
    
    if not fred_key or fred_key == 'your_fred_api_key_here':
        print("❌ FRED API key not configured properly")
        return False
    
    # Test API connection with a simple indicator
    test_series = 'CPIAUCSL'  # Consumer Price Index
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': test_series,
        'api_key': fred_key,
        'file_type': 'json',
        'limit': 5,
        'sort_order': 'desc'
    }
    
    print(f"🔍 Testing API connection with series: {test_series}")
    print(f"📡 Request URL: {url}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            observations = data.get('observations', [])
            
            if observations:
                print("✅ FRED API connection successful!")
                print(f"📈 Retrieved {len(observations)} observations")
                
                # Show latest data
                latest = observations[0]
                print(f"🔢 Latest CPI value: {latest['value']} (Date: {latest['date']})")
                return True
            else:
                print("⚠️ API responded but no observations returned")
                print(f"Response data: {data}")
                return False
                
        elif response.status_code == 400:
            print("❌ Bad Request - Check API key or parameters")
            print(f"Response: {response.text}")
            return False
            
        elif response.status_code == 429:
            print("❌ Rate limit exceeded")
            return False
            
        else:
            print(f"❌ HTTP Error {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout - Check internet connection")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False

def test_macro_factors_class():
    """Test the MacroFactorsAnalyzer class directly"""
    
    print("\n" + "="*50)
    print("🧪 Testing MacroFactorsAnalyzer Class")
    print("="*50)
    
    try:
        from macro_factors import MacroFactorsAnalyzer
        print("✅ MacroFactorsAnalyzer imported successfully")
        
        # Initialize analyzer
        analyzer = MacroFactorsAnalyzer()
        print("✅ MacroFactorsAnalyzer initialized")
        
        # Check if API key is loaded
        if hasattr(analyzer, 'fred_api_key'):
            key = analyzer.fred_api_key
            if key and key != 'your_fred_api_key_here':
                print(f"✅ FRED API key loaded: {key[:10]}...{key[-4:]}")
            else:
                print("❌ FRED API key not loaded properly")
                return False
        
        # Test fetching a few indicators
        print("🔍 Testing FRED indicators fetch...")
        fred_data = analyzer.fetch_fred_indicators()
        
        if fred_data:
            print(f"✅ Fetched {len(fred_data)} FRED indicators")
            
            # Show some sample data
            valid_indicators = [k for k, v in fred_data.items() if v.get('current_value') is not None]
            print(f"📊 Valid indicators with data: {len(valid_indicators)}")
            
            if valid_indicators:
                sample = fred_data[valid_indicators[0]]
                print(f"📈 Sample indicator: {sample.get('name', 'Unknown')}")
                print(f"   Value: {sample.get('current_value', 'N/A')}")
                print(f"   Change: {sample.get('change_pct', 0):+.2f}%")
                print(f"   Quality: {sample.get('data_quality', 'unknown')}")
            
            return True
        else:
            print("❌ No FRED data retrieved")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing MacroFactorsAnalyzer: {e}")
        return False

if __name__ == "__main__":
    print("🚀 FRED API Connection Test")
    print("="*50)
    
    # Test 1: Direct API connection
    api_success = test_fred_api()
    
    # Test 2: MacroFactorsAnalyzer class
    if api_success:
        class_success = test_macro_factors_class()
        
        if class_success:
            print("\n🎉 All tests passed! FRED API integration is working.")
        else:
            print("\n⚠️ API works but MacroFactorsAnalyzer has issues.")
    else:
        print("\n❌ FRED API connection failed. Check your API key and internet connection.")
    
    print("\n" + "="*50)
    print("Test completed.")
