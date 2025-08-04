#!/usr/bin/env python3
"""
Test script to determine if yfinance issues are due to market timing or technical problems
"""

import yfinance as yf
from datetime import datetime, timedelta
import pytz

def test_market_timing():
    print('🕐 Current time analysis:')
    print(f'Local time: {datetime.now()}')
    print(f'UTC time: {datetime.utcnow()}')

    # Check US market hours (NYSE/COMEX where gold futures trade)
    us_tz = pytz.timezone('US/Eastern')
    us_time = datetime.now(us_tz)
    print(f'US Eastern time: {us_time}')
    print(f'US market typically open: 9:30 AM - 4:00 PM ET (Mon-Fri)')
    
    # Gold futures (COMEX) trade almost 24/7 except weekends
    print(f'Gold futures market: Nearly 24/7 except weekends')

    # Test different symbols and periods
    symbols = ['GC=F', 'GLD', 'GOLD', '^GSPC', 'AAPL']
    periods = ['1d', '5d', '1mo']

    for symbol in symbols:
        print(f'\n📊 Testing {symbol}:')
        for period in periods:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                if not data.empty:
                    latest = data.index[-1]
                    price = data['Close'].iloc[-1]
                    print(f'  ✅ {period}: {len(data)} records, latest: {latest.date()}, price: ${price:.2f}')
                    break
                else:
                    print(f'  ❌ {period}: No data')
            except Exception as e:
                print(f'  ❌ {period}: Error - {str(e)[:50]}')
    
    # Test specific recent timeframes
    print(f'\n🔍 Testing recent timeframes for GC=F:')
    test_periods = [
        ('1d', '1 day'),
        ('2d', '2 days'), 
        ('5d', '5 days'),
        ('1wk', '1 week'),
        ('1mo', '1 month')
    ]
    
    for period, desc in test_periods:
        try:
            ticker = yf.Ticker('GC=F')
            data = ticker.history(period=period)
            if not data.empty:
                print(f'  ✅ {desc}: {len(data)} records, latest: {data.index[-1].date()}')
            else:
                print(f'  ❌ {desc}: No data')
        except Exception as e:
            print(f'  ❌ {desc}: Error - {str(e)[:30]}')

if __name__ == '__main__':
    test_market_timing()
