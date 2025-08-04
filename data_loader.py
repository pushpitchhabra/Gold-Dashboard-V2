"""
Data Loader Module for Gold Trading Dashboard
Handles loading historical data and fetching live gold prices
"""

import pandas as pd
import yfinance as yf
import yaml
import os
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoldDataLoader:
    def __init__(self, config_path="config.yaml"):
        """Initialize the data loader with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.symbol = self.config['data']['symbol']
        self.interval = self.config['data']['interval']
        self.historical_file = self.config['data']['historical_file']
        self.live_file = self.config['data']['live_file']
        self.timezone = self.config['data']['timezone']
    
    def load_historical_data(self):
        """
        Load historical gold price data from CSV file
        If file doesn't exist, download from yfinance
        """
        if os.path.exists(self.historical_file):
            logger.info(f"Loading historical data from {self.historical_file}")
            df = pd.read_csv(self.historical_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            return df
        else:
            logger.info("Historical file not found. Downloading from yfinance...")
            return self.download_historical_data()
    
    def download_historical_data(self, start_date="2020-01-01"):
        """
        Download historical gold price data from yfinance
        Uses daily data for historical periods and 30-minute data for recent periods
        """
        try:
            logger.info(f"Downloading {self.symbol} data from {start_date}")
            
            ticker = yf.Ticker(self.symbol)
            
            # For historical data, use daily intervals to get more history
            # Yahoo Finance limits 30-minute data to ~60 days
            if self.interval == "30m":
                # Get daily data for historical analysis
                logger.info("Downloading daily data for historical analysis...")
                df_daily = ticker.history(start=start_date, interval="1d")
                
                if df_daily.empty:
                    raise ValueError("No daily data received from yfinance")
                
                # Get recent 30-minute data (last 60 days)
                recent_start = datetime.now() - timedelta(days=60)
                logger.info("Downloading recent 30-minute data...")
                df_30m = ticker.history(start=recent_start, interval="30m")
                
                if not df_30m.empty:
                    # Use 30-minute data for recent period, daily for historical
                    cutoff_date = df_30m.index.min()
                    df_historical = df_daily[df_daily.index < cutoff_date]
                    
                    # Combine datasets
                    df = pd.concat([df_historical, df_30m])
                    logger.info(f"Combined {len(df_historical)} daily + {len(df_30m)} 30-minute records")
                else:
                    # Fallback to daily data only
                    df = df_daily
                    logger.info("Using daily data only (30-minute data unavailable)")
            else:
                # Use specified interval
                df = ticker.history(start=start_date, interval=self.interval)
            
            if df.empty:
                raise ValueError("No data received from yfinance")
            
            # Clean and prepare data
            df = df.dropna()
            df.index.name = 'datetime'
            
            # Convert timezone if needed
            if df.index.tz is not None:
                df.index = df.index.tz_convert(self.timezone)
            else:
                df.index = df.index.tz_localize(self.timezone)
            
            # Reset index to save datetime as column
            df_save = df.reset_index()
            
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(self.historical_file), exist_ok=True)
            
            # Save to CSV
            df_save.to_csv(self.historical_file, index=False)
            logger.info(f"Historical data saved to {self.historical_file} ({len(df)} records)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading historical data: {e}")
            raise
    
    def fetch_live_data(self, days_back=7):
        """
        Fetch recent live gold price data using yfinance
        """
        try:
            logger.info(f"Fetching live {self.symbol} data")
            
            # Calculate date range for recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Download recent data using yfinance
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=start_date, end=end_date, interval=self.interval)
            
            if df.empty:
                raise ValueError("No live data received from yfinance")
            
            # Clean and prepare data
            df = df.dropna()
            df.index.name = 'datetime'
            
            # Convert timezone if needed
            if df.index.tz is not None:
                df.index = df.index.tz_convert(self.timezone)
            else:
                df.index = df.index.tz_localize(self.timezone)
            
            # Save live data
            df_save = df.reset_index()
            os.makedirs(os.path.dirname(self.live_file), exist_ok=True)
            df_save.to_csv(self.live_file, index=False)
            
            logger.info(f"Live data saved to {self.live_file}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            # Return empty DataFrame if yfinance fetch fails
            return pd.DataFrame()
    
    def get_data_for_analysis(self, days_back=7, prefer_live=True):
        """
        Smart data access method that tries live data first, then falls back to historical data
        This ensures components always get data for analysis when possible
        """
        if prefer_live:
            logger.info("Attempting to get live data for analysis...")
            live_data = self.fetch_live_data(days_back=days_back)
            
            if not live_data.empty:
                logger.info(f"✅ Using live data: {len(live_data)} records")
                return live_data, 'live'
            else:
                logger.warning("Live data unavailable, falling back to historical data...")
        
        # Fall back to historical data
        try:
            logger.info("Loading historical data for analysis...")
            historical_data = self.load_historical_data()
            
            if not historical_data.empty:
                # Get recent portion of historical data
                recent_data = historical_data.tail(days_back * 48)  # Approximate 30min intervals
                logger.info(f"✅ Using historical data: {len(recent_data)} records")
                return recent_data, 'historical'
            else:
                logger.error("No historical data available")
                return pd.DataFrame(), 'unavailable'
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame(), 'error'

    
    def get_latest_price(self):
        """
        Get the most recent gold price
        """
        try:
            live_data = self.fetch_live_data(days_back=1)
            if not live_data.empty:
                latest_price = live_data['Close'].iloc[-1]
                latest_time = live_data.index[-1]
                return latest_price, latest_time
            else:
                return None, None
        except Exception as e:
            logger.error(f"Error getting latest price: {e}")
            return None, None
    
    def update_historical_data(self):
        """
        Update historical data with recent data
        """
        try:
            # Load existing historical data
            historical_df = self.load_historical_data()
            
            # Get the last date in historical data
            last_date = historical_df.index.max()
            
            # Fetch new data from last date to now
            start_date = last_date + timedelta(minutes=30)  # Start from next 30-min interval
            end_date = datetime.now()
            
            if start_date >= end_date:
                logger.info("Historical data is already up to date")
                return historical_df
            
            logger.info(f"Updating historical data from {start_date} to {end_date}")
            
            # Download new data
            ticker = yf.Ticker(self.symbol)
            new_df = ticker.history(start=start_date, end=end_date, interval=self.interval)
            
            if not new_df.empty:
                # Clean new data
                new_df = new_df.dropna()
                new_df.index.name = 'datetime'
                
                # Convert timezone
                if new_df.index.tz is not None:
                    new_df.index = new_df.index.tz_convert(self.timezone)
                else:
                    new_df.index = new_df.index.tz_localize(self.timezone)
                
                # Combine with historical data
                combined_df = pd.concat([historical_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
                
                # Save updated data
                combined_save = combined_df.reset_index()
                combined_save.to_csv(self.historical_file, index=False)
                
                logger.info(f"Added {len(new_df)} new records to historical data")
                return combined_df
            else:
                logger.info("No new data to add")
                return historical_df
                
        except Exception as e:
            logger.error(f"Error updating historical data: {e}")
            return self.load_historical_data()

def main():
    """Test the data loader"""
    loader = GoldDataLoader()
    
    # Test historical data loading
    print("Loading historical data...")
    hist_data = loader.load_historical_data()
    print(f"Historical data shape: {hist_data.shape}")
    print(f"Date range: {hist_data.index.min()} to {hist_data.index.max()}")
    
    # Test live data fetching
    print("\nFetching live data...")
    live_data = loader.fetch_live_data()
    if not live_data.empty:
        print(f"Live data shape: {live_data.shape}")
        print(f"Latest price: ${live_data['Close'].iloc[-1]:.2f}")
    
    # Test latest price
    print("\nGetting latest price...")
    price, time = loader.get_latest_price()
    if price:
        print(f"Latest price: ${price:.2f} at {time}")

if __name__ == "__main__":
    main()
