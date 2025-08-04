#!/usr/bin/env python3
"""
Test different gold symbols to find working alternatives
"""

import yfinance as yf
from datetime import datetime, timedelta

def test_gold_symbols():
    """Test various gold-related symbols"""
    symbols = [
        # Gold Futures
        ("GC=F", "Gold Futures (CME)"),
        ("GCZ24.CMX", "Gold Dec 2024 Futures"),
        ("GCM25.CMX", "Gold Jun 2025 Futures"),
        
        # Gold ETFs
        ("GLD", "SPDR Gold Trust ETF"),
        ("IAU", "iShares Gold Trust ETF"),
        ("SGOL", "Aberdeen Standard Physical Gold ETF"),
        ("OUNZ", "VanEck Merk Gold Trust"),
        
        # Gold Mining Stocks
        ("GOLD", "Barrick Gold Corporation"),
        ("NEM", "Newmont Corporation"),
        ("AEM", "Agnico Eagle Mines"),
        
        # Alternative Gold Instruments
        ("XAUUSD=X", "Gold/USD Forex"),
        ("^XAU", "Philadelphia Gold and Silver Index"),
    ]
    
    working_symbols = []
    
    print("Testing Gold Symbols for Live Data Availability")
    print("=" * 60)
    
    for symbol, description in symbols:
        try:
            print(f"\n📊 Testing {symbol} - {description}")
            
            ticker = yf.Ticker(symbol)
            
            # Test recent data
            data = ticker.history(period="5d", interval="1d")
            
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                latest_date = data.index[-1]
                
                print(f"✅ SUCCESS: ${latest_price:.2f} on {latest_date.date()}")
                print(f"   Records: {len(data)}")
                
                # Test intraday data
                try:
                    intraday = ticker.history(period="1d", interval="30m")
                    if not intraday.empty:
                        print(f"   30m data: {len(intraday)} records available")
                    else:
                        print(f"   30m data: Not available")
                except:
                    print(f"   30m data: Error fetching")
                
                working_symbols.append((symbol, description, latest_price))
            else:
                print(f"❌ FAILED: No data available")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY - Working Symbols:")
    print("=" * 60)
    
    if working_symbols:
        for symbol, description, price in working_symbols:
            print(f"✅ {symbol:12} - {description:30} - ${price:.2f}")
    else:
        print("❌ No working symbols found!")
    
    return working_symbols

if __name__ == "__main__":
    working = test_gold_symbols()
    
    if working:
        print(f"\n🎯 RECOMMENDATION: Use {working[0][0]} as primary symbol")
        print(f"   Latest price: ${working[0][2]:.2f}")
    else:
        print("\n⚠️  No working symbols found - check internet connection or yfinance installation")
