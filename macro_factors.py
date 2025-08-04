"""
Macroeconomic Factors Module for Gold Trading Dashboard
Tracks and analyzes quantitative factors that influence gold prices
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yaml
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroFactorsAnalyzer:
    """
    Analyzes macroeconomic and market factors that influence gold prices
    """
    
    def __init__(self, config_path='config.yaml'):
        """Initialize with configuration"""
        self.config = self._load_config(config_path)
        self.fred_api_key = self.config.get('api_keys', {}).get('fred_key', '')
        
        # Define key factors that influence gold
        self.market_factors = {
            # Currency & Dollar Strength
            'DX-Y.NYB': {'name': 'US Dollar Index (DXY)', 'category': 'Currency', 'correlation': 'negative'},
            'EURUSD=X': {'name': 'EUR/USD', 'category': 'Currency', 'correlation': 'positive'},
            'GBPUSD=X': {'name': 'GBP/USD', 'category': 'Currency', 'correlation': 'positive'},
            'USDJPY=X': {'name': 'USD/JPY', 'category': 'Currency', 'correlation': 'negative'},
            
            # Interest Rates & Bonds
            '^TNX': {'name': '10-Year Treasury Yield', 'category': 'Interest Rates', 'correlation': 'negative'},
            '^FVX': {'name': '5-Year Treasury Yield', 'category': 'Interest Rates', 'correlation': 'negative'},
            '^IRX': {'name': '3-Month Treasury Bill', 'category': 'Interest Rates', 'correlation': 'negative'},
            'TLT': {'name': '20+ Year Treasury Bond ETF', 'category': 'Bonds', 'correlation': 'positive'},
            
            # Market Sentiment & Risk
            '^VIX': {'name': 'CBOE Volatility Index', 'category': 'Risk Sentiment', 'correlation': 'positive'},
            '^GSPC': {'name': 'S&P 500 Index', 'category': 'Equities', 'correlation': 'negative'},
            
            # Commodities & Inflation
            'CL=F': {'name': 'Crude Oil Futures', 'category': 'Commodities', 'correlation': 'positive'},
            'SI=F': {'name': 'Silver Futures', 'category': 'Precious Metals', 'correlation': 'positive'},
            'HG=F': {'name': 'Copper Futures', 'category': 'Industrial Metals', 'correlation': 'positive'},
            
            # Crypto (modern alternative store of value)
            'BTC-USD': {'name': 'Bitcoin', 'category': 'Digital Assets', 'correlation': 'mixed'}
        }
        
        # FRED economic indicators - Comprehensive set
        self.fred_indicators = {
            # Inflation Indicators
            'CPIAUCSL': {'name': 'Consumer Price Index', 'category': 'Inflation', 'correlation': 'positive'},
            'CPILFESL': {'name': 'Core CPI', 'category': 'Inflation', 'correlation': 'positive'},
            'CPIENGSL': {'name': 'CPI: Energy', 'category': 'Inflation', 'correlation': 'positive'},
            'CPIFABSL': {'name': 'CPI: Food and Beverages', 'category': 'Inflation', 'correlation': 'positive'},
            'CPIHOSSL': {'name': 'CPI: Housing', 'category': 'Inflation', 'correlation': 'positive'},
            'CPIMEDSL': {'name': 'CPI: Medical Care', 'category': 'Inflation', 'correlation': 'positive'},
            'CPIAPPSL': {'name': 'CPI: Apparel', 'category': 'Inflation', 'correlation': 'positive'},
            'CPITRNSL': {'name': 'CPI: Transportation', 'category': 'Inflation', 'correlation': 'positive'},
            'CPIRECSL': {'name': 'CPI: Recreation', 'category': 'Inflation', 'correlation': 'positive'},
            'CPIUFDSL': {'name': 'CPI: Food', 'category': 'Inflation', 'correlation': 'positive'},
            'DFEDTARU': {'name': 'Fed Target Rate Upper Limit', 'category': 'Inflation', 'correlation': 'positive'},
            'DFEDTARL': {'name': 'Fed Target Rate Lower Limit', 'category': 'Inflation', 'correlation': 'positive'},
            'T5YIE': {'name': '5-Year Breakeven Inflation Rate', 'category': 'Inflation', 'correlation': 'positive'},
            'T10YIE': {'name': '10-Year Breakeven Inflation Rate', 'category': 'Inflation', 'correlation': 'positive'},
            'T5YIFR': {'name': '5-Year, 5-Year Forward Inflation Expectation Rate', 'category': 'Inflation', 'correlation': 'positive'},
            
            # Employment & Labor Market
            'UNRATE': {'name': 'Unemployment Rate', 'category': 'Employment', 'correlation': 'positive'},
            'PAYEMS': {'name': 'Nonfarm Payrolls', 'category': 'Employment', 'correlation': 'negative'},
            'CIVPART': {'name': 'Labor Force Participation Rate', 'category': 'Employment', 'correlation': 'negative'},
            'EMRATIO': {'name': 'Employment-Population Ratio', 'category': 'Employment', 'correlation': 'negative'},
            'UEMPMED': {'name': 'Median Duration of Unemployment', 'category': 'Employment', 'correlation': 'positive'},
            'UEMP27OV': {'name': 'Long-term Unemployment Rate', 'category': 'Employment', 'correlation': 'positive'},
            'AWHMAN': {'name': 'Average Weekly Hours: Manufacturing', 'category': 'Employment', 'correlation': 'negative'},
            'AHETPI': {'name': 'Average Hourly Earnings', 'category': 'Employment', 'correlation': 'positive'},
            'ICSA': {'name': 'Initial Jobless Claims', 'category': 'Employment', 'correlation': 'positive'},
            'CCSA': {'name': 'Continued Jobless Claims', 'category': 'Employment', 'correlation': 'positive'},
            
            # Monetary Policy & Interest Rates
            'FEDFUNDS': {'name': 'Federal Funds Rate', 'category': 'Monetary Policy', 'correlation': 'negative'},
            'DFF': {'name': 'Daily Federal Funds Rate', 'category': 'Monetary Policy', 'correlation': 'negative'},
            'DTB3': {'name': '3-Month Treasury Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'DTB6': {'name': '6-Month Treasury Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'DGS1': {'name': '1-Year Treasury Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'DGS2': {'name': '2-Year Treasury Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'DGS5': {'name': '5-Year Treasury Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'DGS10': {'name': '10-Year Treasury Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'DGS20': {'name': '20-Year Treasury Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'DGS30': {'name': '30-Year Treasury Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'MORTGAGE30US': {'name': '30-Year Fixed Mortgage Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'MORTGAGE15US': {'name': '15-Year Fixed Mortgage Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'DPRIME': {'name': 'Bank Prime Loan Rate', 'category': 'Interest Rates', 'correlation': 'negative'},
            'DAAA': {'name': 'AAA Corporate Bond Yield', 'category': 'Interest Rates', 'correlation': 'negative'},
            'DBAA': {'name': 'BAA Corporate Bond Yield', 'category': 'Interest Rates', 'correlation': 'negative'},
            
            # Economic Growth & Output
            'GDP': {'name': 'Gross Domestic Product', 'category': 'Economic Growth', 'correlation': 'negative'},
            'GDPC1': {'name': 'Real GDP', 'category': 'Economic Growth', 'correlation': 'negative'},
            'GDPPOT': {'name': 'Real Potential GDP', 'category': 'Economic Growth', 'correlation': 'negative'},
            'NYGDPMKTPCDWLD': {'name': 'World GDP Per Capita', 'category': 'Economic Growth', 'correlation': 'negative'},
            'INDPRO': {'name': 'Industrial Production Index', 'category': 'Economic Growth', 'correlation': 'negative'},
            'CAPUTLG2211S': {'name': 'Capacity Utilization: Manufacturing', 'category': 'Economic Growth', 'correlation': 'negative'},
            'TCU': {'name': 'Total Capacity Utilization', 'category': 'Economic Growth', 'correlation': 'negative'},
            'HOUST': {'name': 'Housing Starts', 'category': 'Real Estate', 'correlation': 'negative'},
            'PERMIT': {'name': 'Building Permits', 'category': 'Real Estate', 'correlation': 'negative'},
            'CSUSHPISA': {'name': 'Case-Shiller Home Price Index', 'category': 'Real Estate', 'correlation': 'positive'},
            
            # Money Supply & Credit
            'M1SL': {'name': 'M1 Money Supply', 'category': 'Money Supply', 'correlation': 'positive'},
            'M2SL': {'name': 'M2 Money Supply', 'category': 'Money Supply', 'correlation': 'positive'},
            'BOGMBASE': {'name': 'Monetary Base', 'category': 'Money Supply', 'correlation': 'positive'},
            'WALCL': {'name': 'Fed Balance Sheet', 'category': 'Money Supply', 'correlation': 'positive'},
            'TOTRESNS': {'name': 'Total Bank Reserves', 'category': 'Money Supply', 'correlation': 'positive'},
            'EXCSRESNS': {'name': 'Excess Bank Reserves', 'category': 'Money Supply', 'correlation': 'positive'},
            'LOANS': {'name': 'Total Bank Loans', 'category': 'Credit', 'correlation': 'negative'},
            'CONSUMER': {'name': 'Consumer Loans', 'category': 'Credit', 'correlation': 'negative'},
            'REALLN': {'name': 'Real Estate Loans', 'category': 'Credit', 'correlation': 'negative'},
            
            # Consumer & Business Sentiment
            'UMCSENT': {'name': 'Consumer Sentiment', 'category': 'Sentiment', 'correlation': 'negative'},
            'UMCSENT1': {'name': 'Consumer Sentiment: Current Conditions', 'category': 'Sentiment', 'correlation': 'negative'},
            'UMCSENT2': {'name': 'Consumer Sentiment: Expectations', 'category': 'Sentiment', 'correlation': 'negative'},
            'RSAFS': {'name': 'Retail Sales', 'category': 'Consumer Spending', 'correlation': 'negative'},
            'RSFSN': {'name': 'Retail Sales Ex Auto', 'category': 'Consumer Spending', 'correlation': 'negative'},
            'PCE': {'name': 'Personal Consumption Expenditures', 'category': 'Consumer Spending', 'correlation': 'negative'},
            'PCEDG': {'name': 'PCE: Durable Goods', 'category': 'Consumer Spending', 'correlation': 'negative'},
            'PCEND': {'name': 'PCE: Non-Durable Goods', 'category': 'Consumer Spending', 'correlation': 'negative'},
            'PCES': {'name': 'PCE: Services', 'category': 'Consumer Spending', 'correlation': 'negative'},
            
            # Trade & International
            'BOPGSTB': {'name': 'Trade Balance: Goods and Services', 'category': 'Trade', 'correlation': 'positive'},
            'BOPGTB': {'name': 'Trade Balance: Goods', 'category': 'Trade', 'correlation': 'positive'},
            'EXPGS': {'name': 'Exports: Goods and Services', 'category': 'Trade', 'correlation': 'negative'},
            'IMPGS': {'name': 'Imports: Goods and Services', 'category': 'Trade', 'correlation': 'positive'},
            'DEXUSEU': {'name': 'USD/EUR Exchange Rate', 'category': 'Currency', 'correlation': 'negative'},
            'DEXJPUS': {'name': 'JPY/USD Exchange Rate', 'category': 'Currency', 'correlation': 'positive'},
            'DEXCHUS': {'name': 'CNY/USD Exchange Rate', 'category': 'Currency', 'correlation': 'positive'},
            'DEXUSUK': {'name': 'USD/GBP Exchange Rate', 'category': 'Currency', 'correlation': 'negative'},
            
            # Government Finance
            'FYFSGDA188S': {'name': 'Federal Surplus/Deficit as % of GDP', 'category': 'Fiscal Policy', 'correlation': 'negative'},
            'GFDEBTN': {'name': 'Federal Debt: Total Public Debt', 'category': 'Fiscal Policy', 'correlation': 'positive'},
            'GFDEGDQ188S': {'name': 'Federal Debt as % of GDP', 'category': 'Fiscal Policy', 'correlation': 'positive'},
            'FGRECPT': {'name': 'Federal Government Receipts', 'category': 'Fiscal Policy', 'correlation': 'negative'},
            'FGEXPND': {'name': 'Federal Government Expenditures', 'category': 'Fiscal Policy', 'correlation': 'positive'},
            
            # Commodity & Energy Prices
            'DCOILWTICO': {'name': 'WTI Crude Oil Price', 'category': 'Commodities', 'correlation': 'positive'},
            'DCOILBRENTEU': {'name': 'Brent Crude Oil Price', 'category': 'Commodities', 'correlation': 'positive'},
            'DHHNGSP': {'name': 'Natural Gas Price', 'category': 'Commodities', 'correlation': 'positive'},
            'DPROPANEMBTX': {'name': 'Propane Price', 'category': 'Commodities', 'correlation': 'positive'},
            'DJFUELUSGULF': {'name': 'Jet Fuel Price', 'category': 'Commodities', 'correlation': 'positive'},
            'DEXBZUS': {'name': 'Brazil/US Exchange Rate', 'category': 'Currency', 'correlation': 'positive'},
            
            # Financial Stress & Risk Indicators
            'NFCI': {'name': 'National Financial Conditions Index', 'category': 'Financial Stress', 'correlation': 'positive'},
            'ANFCI': {'name': 'Adjusted National Financial Conditions Index', 'category': 'Financial Stress', 'correlation': 'positive'},
            'STLFSI': {'name': 'St. Louis Fed Financial Stress Index', 'category': 'Financial Stress', 'correlation': 'positive'},
            'TEDRATE': {'name': 'TED Spread', 'category': 'Financial Stress', 'correlation': 'positive'},
            'T10Y2Y': {'name': '10-Year Treasury Minus 2-Year Treasury', 'category': 'Yield Curve', 'correlation': 'positive'},
            'T10Y3M': {'name': '10-Year Treasury Minus 3-Month Treasury', 'category': 'Yield Curve', 'correlation': 'positive'},
            
            # Global Economic Indicators
            'NYGDPMKTPCDWLD': {'name': 'World GDP Per Capita', 'category': 'Global Economy', 'correlation': 'negative'},
            'CPILFESL': {'name': 'Core CPI', 'category': 'Inflation', 'correlation': 'positive'},
            'CPALTT01USM657N': {'name': 'CPI: All Items Less Food and Energy', 'category': 'Inflation', 'correlation': 'positive'},
            'CPILFESL': {'name': 'Consumer Price Index for All Urban Consumers: All Items Less Food and Energy', 'category': 'Inflation', 'correlation': 'positive'}
        }
        
        self.factor_data = {}
        self.correlations = {}
        self.influence_scores = {}
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def fetch_market_factors(self, period='1mo') -> Dict:
        """
        Fetch market-based factors using yfinance
        """
        logger.info("Fetching market factors...")
        market_data = {}
        
        for symbol, info in self.market_factors.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    # Calculate volatility (20-day rolling std)
                    volatility = hist['Close'].pct_change().rolling(20).std().iloc[-1] * 100
                    
                    # Get last 30 data points for correlation analysis
                    close_data = hist['Close'].tail(30)
                    timestamps = close_data.index
                    
                    market_data[symbol] = {
                        'name': info['name'],
                        'category': info['category'],
                        'current_value': current_price,
                        'change_pct': change_pct,
                        'volatility': volatility if not pd.isna(volatility) else 0,
                        'correlation_type': info['correlation'],
                        'data': close_data.tolist(),  # Last 30 days for correlation
                        'timestamps': timestamps.strftime('%Y-%m-%d').tolist()
                    }
                    
                logger.info(f"✓ Fetched {info['name']}")
                
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                market_data[symbol] = {
                    'name': info['name'],
                    'category': info['category'],
                    'current_value': None,
                    'change_pct': 0,
                    'volatility': 0,
                    'correlation_type': info['correlation'],
                    'error': str(e)
                }
        
        return market_data
    
    def fetch_fred_indicators(self) -> Dict:
        """
        Fetch economic indicators from FRED API with enhanced processing
        """
        if not self.fred_api_key or self.fred_api_key == 'your_fred_api_key_here':
            logger.warning("FRED API key not configured, skipping economic indicators")
            return {}
        
        logger.info(f"Fetching {len(self.fred_indicators)} FRED economic indicators...")
        fred_data = {}
        successful_fetches = 0
        failed_fetches = 0
        
        # Group indicators by category for better logging
        categories = {}
        for indicator, info in self.fred_indicators.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((indicator, info))
        
        # Process each category
        for category, indicators in categories.items():
            logger.info(f"Processing {category} indicators ({len(indicators)} items)...")
            
            for indicator, info in indicators:
                try:
                    # FRED API endpoint with enhanced parameters
                    url = f"https://api.stlouisfed.org/fred/series/observations"
                    params = {
                        'series_id': indicator,
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'limit': 50,  # Reduced for efficiency
                        'sort_order': 'desc',
                        'observation_start': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')  # Last year only
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        observations = data.get('observations', [])
                        
                        if observations:
                            # Enhanced data processing
                            valid_observations = [obs for obs in observations if obs['value'] not in ['.', '', None]]
                            
                            if len(valid_observations) >= 1:
                                latest_obs = valid_observations[0]
                                current_value = float(latest_obs['value'])
                                
                                # Calculate change percentage with multiple periods
                                prev_value = None
                                change_pct = 0
                                
                                if len(valid_observations) >= 2:
                                    prev_obs = valid_observations[1]
                                    prev_value = float(prev_obs['value'])
                                    change_pct = ((current_value - prev_value) / prev_value) * 100 if prev_value != 0 else 0
                                
                                # Calculate additional metrics
                                values = [float(obs['value']) for obs in valid_observations[:12]]  # Last 12 observations
                                avg_value = np.mean(values) if len(values) > 1 else current_value
                                volatility = np.std(values) if len(values) > 2 else 0
                                
                                # Get series metadata for better context
                                series_info = self._get_fred_series_info(indicator)
                                
                                fred_data[indicator] = {
                                    'name': info['name'],
                                    'category': info['category'],
                                    'current_value': current_value,
                                    'previous_value': prev_value,
                                    'change_pct': change_pct,
                                    'correlation_type': info['correlation'],
                                    'last_updated': latest_obs['date'],
                                    'unit': series_info.get('units', 'N/A'),
                                    'frequency': series_info.get('frequency', 'N/A'),
                                    'avg_12_period': avg_value,
                                    'volatility': volatility,
                                    'data_quality': 'high' if len(valid_observations) >= 10 else 'medium' if len(valid_observations) >= 5 else 'low',
                                    'observations_count': len(valid_observations)
                                }
                                
                                successful_fetches += 1
                                
                            else:
                                logger.warning(f"No valid data for {indicator} ({info['name']})")
                                failed_fetches += 1
                        else:
                            logger.warning(f"No observations returned for {indicator}")
                            failed_fetches += 1
                            
                    elif response.status_code == 429:
                        logger.warning(f"Rate limit hit for {indicator}, waiting...")
                        time.sleep(1)  # Brief pause for rate limiting
                        failed_fetches += 1
                    else:
                        logger.error(f"HTTP {response.status_code} for {indicator}")
                        failed_fetches += 1
                        
                except requests.exceptions.Timeout:
                    logger.error(f"Timeout fetching {indicator}")
                    failed_fetches += 1
                except ValueError as e:
                    logger.error(f"Data parsing error for {indicator}: {e}")
                    failed_fetches += 1
                except Exception as e:
                    logger.error(f"Unexpected error fetching {indicator}: {e}")
                    failed_fetches += 1
                    
                # Add failed indicator with placeholder data
                if indicator not in fred_data:
                    fred_data[indicator] = {
                        'name': info['name'],
                        'category': info['category'],
                        'current_value': None,
                        'change_pct': 0,
                        'correlation_type': info['correlation'],
                        'error': 'Failed to fetch data',
                        'data_quality': 'unavailable'
                    }
        
        logger.info(f"✓ FRED data fetch complete: {successful_fetches} successful, {failed_fetches} failed")
        
        # Log summary by category
        for category in categories.keys():
            category_data = [v for k, v in fred_data.items() if v['category'] == category and v.get('current_value') is not None]
            logger.info(f"  {category}: {len(category_data)} indicators with data")
        
        return fred_data
    
    def _get_fred_series_info(self, series_id: str) -> Dict:
        """
        Get additional series metadata from FRED API
        """
        try:
            url = f"https://api.stlouisfed.org/fred/series"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                series_data = data.get('seriess', [])
                if series_data:
                    return {
                        'units': series_data[0].get('units', 'N/A'),
                        'frequency': series_data[0].get('frequency', 'N/A'),
                        'seasonal_adjustment': series_data[0].get('seasonal_adjustment', 'N/A'),
                        'last_updated': series_data[0].get('last_updated', 'N/A')
                    }
        except Exception as e:
            logger.debug(f"Could not fetch series info for {series_id}: {e}")
        
        return {}
    
    def calculate_correlations_with_gold(self, gold_data: pd.Series) -> Dict:
        """
        Calculate robust correlations using multiple models for highest accuracy
        """
        correlations = {}
        
        logger.info(f"Calculating robust correlations with gold data length: {len(gold_data)}")
        
        # Calculate correlations with both market and economic factors
        all_factors = {}
        all_factors.update(self.factor_data.get('market', {}))
        all_factors.update(self.factor_data.get('economic', {}))
        
        for symbol, data in all_factors.items():
            if 'data' in data and len(data['data']) > 3:  # More lenient minimum
                try:
                    factor_series = pd.Series(data['data'])
                    
                    # Multiple correlation approaches for robustness
                    correlation_results = self._calculate_multiple_correlations(
                        gold_data, factor_series, symbol, data['name']
                    )
                    
                    if correlation_results:
                        correlations[symbol] = correlation_results
                        logger.info(f"✓ {data['name']}: {correlation_results['correlation']:.3f} "
                                  f"({correlation_results['method']}, {correlation_results['data_points']} points)")
                        
                except Exception as e:
                    logger.error(f"Error calculating correlation for {symbol}: {e}")
        
        logger.info(f"Calculated {len(correlations)} robust correlations")
        return correlations
    
    def _calculate_multiple_correlations(self, gold_data: pd.Series, factor_data: pd.Series, 
                                       symbol: str, name: str) -> Dict:
        """
        Calculate correlation using multiple methods and return the most reliable one
        """
        from scipy.stats import pearsonr, spearmanr
        from sklearn.preprocessing import StandardScaler
        
        try:
            # Data alignment and cleaning
            min_length = min(len(gold_data), len(factor_data))
            if min_length < 3:
                return None
            
            # Take most recent overlapping data
            gold_aligned = gold_data.iloc[-min_length:].reset_index(drop=True)
            factor_aligned = factor_data.iloc[-min_length:].reset_index(drop=True)
            
            # Create combined dataframe and remove NaN
            combined_df = pd.DataFrame({
                'gold': gold_aligned,
                'factor': factor_aligned
            }).dropna()
            
            if len(combined_df) < 3:
                return None
            
            gold_clean = combined_df['gold'].values
            factor_clean = combined_df['factor'].values
            
            # Method 1: Pearson correlation (linear relationships)
            try:
                pearson_corr, pearson_p = pearsonr(gold_clean, factor_clean)
                pearson_valid = not np.isnan(pearson_corr) and pearson_p < 0.1  # 90% confidence
            except:
                pearson_corr, pearson_valid = 0, False
            
            # Method 2: Spearman correlation (monotonic relationships)
            try:
                spearman_corr, spearman_p = spearmanr(gold_clean, factor_clean)
                spearman_valid = not np.isnan(spearman_corr) and spearman_p < 0.1
            except:
                spearman_corr, spearman_valid = 0, False
            
            # Method 3: Rolling correlation (recent trend)
            try:
                if len(combined_df) >= 5:
                    rolling_corr = combined_df['gold'].rolling(window=min(5, len(combined_df))).corr(
                        combined_df['factor']
                    ).iloc[-1]
                    rolling_valid = not np.isnan(rolling_corr)
                else:
                    rolling_corr, rolling_valid = 0, False
            except:
                rolling_corr, rolling_valid = 0, False
            
            # Method 4: Standardized correlation (normalized data)
            try:
                scaler_gold = StandardScaler()
                scaler_factor = StandardScaler()
                gold_scaled = scaler_gold.fit_transform(gold_clean.reshape(-1, 1)).flatten()
                factor_scaled = scaler_factor.fit_transform(factor_clean.reshape(-1, 1)).flatten()
                
                standardized_corr = np.corrcoef(gold_scaled, factor_scaled)[0, 1]
                standardized_valid = not np.isnan(standardized_corr)
            except:
                standardized_corr, standardized_valid = 0, False
            
            # Select the best correlation method
            correlations_methods = [
                ('Pearson', pearson_corr, pearson_valid, abs(pearson_corr)),
                ('Spearman', spearman_corr, spearman_valid, abs(spearman_corr)),
                ('Rolling', rolling_corr, rolling_valid, abs(rolling_corr)),
                ('Standardized', standardized_corr, standardized_valid, abs(standardized_corr))
            ]
            
            # Filter valid correlations and sort by absolute strength
            valid_correlations = [(method, corr, valid, strength) for method, corr, valid, strength 
                                in correlations_methods if valid and abs(corr) > 0.01]
            
            if not valid_correlations:
                # If no statistically significant correlations, use the strongest absolute correlation
                valid_correlations = [(method, corr, valid, strength) for method, corr, valid, strength 
                                    in correlations_methods if not np.isnan(corr)]
            
            if not valid_correlations:
                return None
            
            # Select the method with highest absolute correlation
            best_method, best_corr, _, _ = max(valid_correlations, key=lambda x: x[3])
            
            # Get expected direction from config
            factor_info = self.factor_data.get('market', {}).get(symbol) or \
                         self.factor_data.get('economic', {}).get(symbol, {})
            expected_direction = factor_info.get('correlation_type', 'unknown')
            
            return {
                'correlation': float(best_corr),
                'strength': self._classify_correlation_strength(abs(best_corr)),
                'direction': 'positive' if best_corr > 0 else 'negative',
                'expected_direction': expected_direction,
                'data_points': len(combined_df),
                'method': best_method,
                'confidence': 'high' if abs(best_corr) > 0.3 else 'medium' if abs(best_corr) > 0.1 else 'low',
                'all_methods': {
                    'pearson': float(pearson_corr) if not np.isnan(pearson_corr) else 0,
                    'spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0,
                    'rolling': float(rolling_corr) if not np.isnan(rolling_corr) else 0,
                    'standardized': float(standardized_corr) if not np.isnan(standardized_corr) else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error in multiple correlation calculation for {symbol}: {e}")
            return None
    
    def _classify_correlation_strength(self, abs_correlation: float) -> str:
        """Classify correlation strength"""
        if abs_correlation >= 0.7:
            return 'Strong'
        elif abs_correlation >= 0.4:
            return 'Moderate'
        elif abs_correlation >= 0.2:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def calculate_influence_scores(self) -> Dict:
        """
        Calculate current influence scores for each factor
        """
        influence_scores = {}
        
        for category in ['market', 'economic']:
            factors = self.factor_data.get(category, {})
            
            for symbol, data in factors.items():
                try:
                    # Base influence on volatility and recent change
                    volatility_score = min(abs(data.get('volatility', 0)) / 5, 1.0)  # Normalize to 0-1
                    change_score = min(abs(data.get('change_pct', 0)) / 10, 1.0)  # Normalize to 0-1
                    
                    # Get correlation strength
                    correlation_data = self.correlations.get(symbol, {})
                    correlation_strength = correlation_data.get('correlation', 0)
                    correlation_score = abs(correlation_strength)
                    
                    # Combined influence score (0-100)
                    influence_score = (volatility_score * 0.3 + change_score * 0.4 + correlation_score * 0.3) * 100
                    
                    influence_scores[symbol] = {
                        'score': influence_score,
                        'components': {
                            'volatility': volatility_score * 100,
                            'recent_change': change_score * 100,
                            'correlation': correlation_score * 100
                        },
                        'category': data['category'],
                        'name': data['name']
                    }
                    
                except Exception as e:
                    logger.error(f"Error calculating influence for {symbol}: {e}")
        
        return influence_scores
    
    def get_top_influencing_factors(self, limit: int = 5) -> List[Dict]:
        """
        Get top factors currently influencing gold prices
        """
        # Ensure influence scores are calculated
        if not hasattr(self, 'influence_scores') or not self.influence_scores:
            self.influence_scores = self.calculate_influence_scores()
        
        # Sort by influence score
        sorted_factors = sorted(
            self.influence_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        top_factors = []
        for symbol, data in sorted_factors[:limit]:
            factor_info = self.factor_data.get('market', {}).get(symbol, 
                         self.factor_data.get('economic', {}).get(symbol, {}))
            
            if factor_info:
                top_factors.append({
                    'symbol': symbol,
                    'name': data['name'],
                    'category': data['category'],
                    'influence_score': data['score'],
                    'current_value': factor_info.get('current_value'),
                    'change_pct': factor_info.get('change_pct', 0),
                    'correlation': self.correlations.get(symbol, {}).get('correlation', 0),
                    'expected_impact': self._determine_expected_impact(symbol, factor_info)
                })
        
        return top_factors
    
    def _determine_expected_impact(self, symbol: str, factor_data: Dict) -> str:
        """
        Determine expected impact on gold based on factor movement
        """
        change_pct = factor_data.get('change_pct', 0)
        correlation_type = factor_data.get('correlation_type', 'mixed')
        
        if abs(change_pct) < 0.1:
            return 'Neutral'
        
        if correlation_type == 'positive':
            return 'Bullish' if change_pct > 0 else 'Bearish'
        elif correlation_type == 'negative':
            return 'Bearish' if change_pct > 0 else 'Bullish'
        else:
            return 'Mixed'
    
    def generate_macro_summary(self) -> Dict:
        """
        Generate comprehensive macroeconomic summary
        """
        top_factors = self.get_top_influencing_factors(10)
        
        # Categorize impacts
        bullish_factors = [f for f in top_factors if f['expected_impact'] == 'Bullish']
        bearish_factors = [f for f in top_factors if f['expected_impact'] == 'Bearish']
        
        # Calculate overall macro sentiment
        bullish_score = sum(f['influence_score'] for f in bullish_factors)
        bearish_score = sum(f['influence_score'] for f in bearish_factors)
        
        if bullish_score > bearish_score * 1.2:
            overall_sentiment = 'Bullish'
        elif bearish_score > bullish_score * 1.2:
            overall_sentiment = 'Bearish'
        else:
            overall_sentiment = 'Mixed'
        
        return {
            'overall_sentiment': overall_sentiment,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'top_factors': top_factors,
            'bullish_factors': bullish_factors,
            'bearish_factors': bearish_factors,
            'summary': self._generate_narrative_summary(overall_sentiment, top_factors)
        }
    
    def _generate_narrative_summary(self, sentiment: str, top_factors: List[Dict]) -> str:
        """
        Generate narrative summary of macro conditions
        """
        if not top_factors:
            return "Insufficient data to generate macro analysis."
        
        top_factor = top_factors[0]
        
        summary = f"Current macro environment is **{sentiment.lower()}** for gold. "
        summary += f"The most influential factor is **{top_factor['name']}** "
        summary += f"(influence score: {top_factor['influence_score']:.1f}), "
        summary += f"which is currently {top_factor['expected_impact'].lower()} for gold prices. "
        
        if len(top_factors) > 1:
            summary += f"Other key factors include {', '.join([f['name'] for f in top_factors[1:3]])}. "
        
        return summary
    
    def update_all_factors(self, gold_data: pd.Series = None):
        """
        Update all macroeconomic factors and calculate relationships
        """
        logger.info("Updating all macroeconomic factors...")
        
        # Fetch all data
        self.factor_data['market'] = self.fetch_market_factors()
        self.factor_data['economic'] = self.fetch_fred_indicators()
        
        # Calculate correlations if gold data provided
        if gold_data is not None and not gold_data.empty:
            self.correlations = self.calculate_correlations_with_gold(gold_data)
        
        # Calculate influence scores
        self.influence_scores = self.calculate_influence_scores()
        
        logger.info("✓ All macroeconomic factors updated successfully")
        
        return self.generate_macro_summary()
    
    def get_macro_factors_by_category(self) -> Dict:
        """
        Get organized macro factors data by category for dashboard display
        """
        categories = {}
        
        # Combine market and economic data
        all_factors = {}
        if 'market' in self.factor_data:
            all_factors.update(self.factor_data['market'])
        if 'economic' in self.factor_data:
            all_factors.update(self.factor_data['economic'])
        
        for symbol, data in all_factors.items():
            category = data.get('category', 'Other')
            if category not in categories:
                categories[category] = []
            
            # Get influence score safely - handle both numeric and dict types
            influence_score = self.influence_scores.get(symbol, 0)
            if isinstance(influence_score, dict):
                # If it's a dict, try to get a numeric value from it
                influence_score = influence_score.get('score', 0) or influence_score.get('influence_score', 0) or 0
            elif not isinstance(influence_score, (int, float)):
                influence_score = 0
            
            categories[category].append({
                'symbol': symbol,
                'name': data.get('name', symbol),
                'current_value': data.get('current_value'),
                'change_pct': data.get('change_pct', 0),
                'correlation_type': data.get('correlation_type', 'mixed'),
                'last_updated': data.get('last_updated', 'N/A'),
                'unit': data.get('unit', 'N/A'),
                'frequency': data.get('frequency', 'N/A'),
                'data_quality': data.get('data_quality', 'unknown'),
                'influence_score': influence_score
            })
        
        # Sort each category by influence score (now guaranteed to be numeric)
        for category in categories:
            categories[category].sort(key=lambda x: abs(x.get('influence_score', 0)), reverse=True)
        
        return categories
    
    def get_macro_dashboard_summary(self) -> Dict:
        """
        Get comprehensive summary for dashboard display
        """
        summary = self.generate_macro_summary()
        categories = self.get_macro_factors_by_category()
        
        # Calculate category-wise statistics
        category_stats = {}
        for category, factors in categories.items():
            valid_factors = [f for f in factors if f['current_value'] is not None]
            category_stats[category] = {
                'total_indicators': len(factors),
                'valid_indicators': len(valid_factors),
                'avg_change': np.mean([f['change_pct'] for f in valid_factors]) if valid_factors else 0,
                'bullish_count': len([f for f in valid_factors if self._determine_expected_impact(f['symbol'], f) == 'Bullish']),
                'bearish_count': len([f for f in valid_factors if self._determine_expected_impact(f['symbol'], f) == 'Bearish']),
                'top_factor': valid_factors[0] if valid_factors else None
            }
        
        return {
            'overall_summary': summary,
            'categories': categories,
            'category_stats': category_stats,
            'total_indicators': sum(len(factors) for factors in categories.values()),
            'data_freshness': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_dashboard_data(self) -> Dict:
        """
        Alias for get_macro_dashboard_summary() for backward compatibility
        Used by diagnostic scripts and dashboard components
        """
        try:
            # Try to get fresh data
            dashboard_data = self.get_macro_dashboard_summary()
            
            # Ensure the data has the expected structure
            if not dashboard_data or 'categories' not in dashboard_data:
                logger.warning("Dashboard data incomplete, generating fallback")
                return self._get_fallback_dashboard_data()
            
            # Transform to expected format for dashboard
            factors = []
            for category, factor_list in dashboard_data['categories'].items():
                for factor in factor_list:
                    factors.append({
                        'symbol': factor['symbol'],
                        'name': factor['name'],
                        'current_value': factor['current_value'],
                        'change_pct': factor['change_pct'],
                        'category': category,
                        'correlation_type': factor['correlation_type'],
                        'influence_score': factor['influence_score'],
                        'last_updated': factor['last_updated'],
                        'unit': factor['unit'],
                        'frequency': factor['frequency'],
                        'data_quality': factor['data_quality']
                    })
            
            return {
                'factors': factors,
                'summary': dashboard_data.get('overall_summary', {}),
                'category_stats': dashboard_data.get('category_stats', {}),
                'total_indicators': dashboard_data.get('total_indicators', 0),
                'data_freshness': dashboard_data.get('data_freshness', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return self._get_fallback_dashboard_data()
    
    def _get_fallback_dashboard_data(self) -> Dict:
        """
        Provide fallback data when main data fetch fails
        """
        return {
            'factors': [
                {
                    'symbol': 'USD_INDEX',
                    'name': 'US Dollar Index (Simulated)',
                    'current_value': 103.5,
                    'change_pct': -0.2,
                    'category': 'Currency',
                    'correlation_type': 'negative',
                    'influence_score': 0.8,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'unit': 'Index',
                    'frequency': 'Real-time',
                    'data_quality': 'simulated'
                },
                {
                    'symbol': 'TREASURY_10Y',
                    'name': '10-Year Treasury Yield (Simulated)',
                    'current_value': 4.25,
                    'change_pct': 0.1,
                    'category': 'Interest Rates',
                    'correlation_type': 'negative',
                    'influence_score': 0.7,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'unit': '%',
                    'frequency': 'Daily',
                    'data_quality': 'simulated'
                },
                {
                    'symbol': 'INFLATION_CPI',
                    'name': 'Consumer Price Index (Simulated)',
                    'current_value': 3.2,
                    'change_pct': 0.05,
                    'category': 'Inflation',
                    'correlation_type': 'positive',
                    'influence_score': 0.9,
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'unit': '% YoY',
                    'frequency': 'Monthly',
                    'data_quality': 'simulated'
                }
            ],
            'summary': {
                'sentiment': 'NEUTRAL',
                'bullish_factors': 1,
                'bearish_factors': 1,
                'neutral_factors': 1,
                'narrative': 'Using simulated data due to API connectivity issues. Real data will be restored when connection is available.'
            },
            'category_stats': {
                'Currency': {'total_indicators': 1, 'bullish_count': 0, 'bearish_count': 1},
                'Interest Rates': {'total_indicators': 1, 'bullish_count': 0, 'bearish_count': 1},
                'Inflation': {'total_indicators': 1, 'bullish_count': 1, 'bearish_count': 0}
            },
            'total_indicators': 3,
            'data_freshness': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'fallback'
        }

if __name__ == "__main__":
    # Test the macro factors analyzer
    analyzer = MacroFactorsAnalyzer()
    
    # Create sample gold data for testing
    gold_data = pd.Series([2000, 2010, 2005, 2020, 2015] * 10)
    
    summary = analyzer.update_all_factors(gold_data)
    print("Macro Summary:", summary)
