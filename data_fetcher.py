"""
BTC 多因子研究系统 - 数据获取模块
统一的多源数据获取接口
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import pandas_datareader.data as web
from abc import ABC, abstractmethod
import json
import warnings
warnings.filterwarnings('ignore')


class DataFetcher(ABC):
    """数据获取基类"""
    
    @abstractmethod
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """获取数据的抽象方法"""
        pass
    
    def _handle_missing_data(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """处理缺失数据"""
        if method == 'forward_fill':
            return df.ffill()
        elif method == 'interpolate':
            return df.interpolate(method='linear')
        elif method == 'drop':
            return df.dropna()
        return df


class YahooFetcher(DataFetcher):
    """Yahoo Finance数据获取器"""
    
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """获取Yahoo Finance数据"""
        try:
            # 使用curl_cffi session避免rate limit
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"Warning: No data fetched for {symbol} from Yahoo")
                return pd.DataFrame()
            
            # 只返回收盘价
            result = df[['Close']].copy()
            result.columns = [symbol]
            return result
            
        except Exception as e:
            print(f"Error fetching {symbol} from Yahoo: {e}")
            return pd.DataFrame()


class FredFetcher(DataFetcher):
    """FRED数据获取器"""
    
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """获取FRED数据"""
        try:
            df = web.DataReader(symbol, 'fred', start_date, end_date)
            df.columns = [symbol]
            return df
        except Exception as e:
            print(f"Error fetching {symbol} from FRED: {e}")
            return pd.DataFrame()


class BinanceFetcher(DataFetcher):
    """币安数据获取器"""
    
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """获取币安数据"""
        try:
            # 币安API endpoint
            url = "https://api.binance.com/api/v3/klines"
            
            # 转换日期为时间戳
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            
            all_data = []
            current_start = start_ts
            
            while current_start < end_ts:
                params = {
                    "symbol": symbol,
                    "interval": "1d",
                    "startTime": current_start,
                    "endTime": end_ts,
                    "limit": 1000
                }
                
                response = requests.get(url, params=params)
                if response.status_code != 200:
                    break
                    
                data = response.json()
                if not data:
                    break
                    
                all_data.extend(data)
                
                # 更新起始时间
                current_start = int(data[-1][0]) + 1
                time.sleep(0.1)  # Rate limiting
            
            if not all_data:
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)
            
            result = df[['close']].astype(float)
            result.columns = [symbol]
            
            return result
            
        except Exception as e:
            print(f"Error fetching {symbol} from Binance: {e}")
            return pd.DataFrame()


class CoinGeckoFetcher(DataFetcher):
    """CoinGecko数据获取器"""
    
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """获取CoinGecko数据"""
        try:
            # 计算天数
            days = (end_date - start_date).days
            
            if symbol == "total_market_cap":
                url = "https://api.coingecko.com/api/v3/global"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    total_mcap = data['data']['total_market_cap']['usd']
                    # 返回单点数据，需要历史数据API（付费）
                    df = pd.DataFrame({'total_market_cap': [total_mcap]}, 
                                    index=[datetime.now()])
                    return df
                    
            elif symbol == "btc_dominance":
                url = "https://api.coingecko.com/api/v3/global"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    dominance = data['data']['market_cap_percentage']['btc']
                    df = pd.DataFrame({'btc_dominance': [dominance]}, 
                                    index=[datetime.now()])
                    return df
                    
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching {symbol} from CoinGecko: {e}")
            return pd.DataFrame()


class CryptoCompareFetcher(DataFetcher):
    """CryptoCompare数据获取器"""
    
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """获取CryptoCompare数据"""
        try:
            # 计算天数
            days = min((end_date - start_date).days, 2000)
            
            url = "https://min-api.cryptocompare.com/data/v2/histoday"
            params = {
                "fsym": "BTC",
                "tsym": "USD",
                "limit": days
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('Response') == 'Success':
                    prices = data.get('Data', {}).get('Data', [])
                    if prices:
                        df = pd.DataFrame(prices)
                        df['date'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('date', inplace=True)
                        result = df[['close']]
                        result.columns = [symbol]
                        return result
                        
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching {symbol} from CryptoCompare: {e}")
            return pd.DataFrame()


class AlternativeMeFetcher(DataFetcher):
    """Alternative.me数据获取器（恐慌贪婪指数）"""
    
    def fetch(self, symbol: str, start_date: datetime, end_date: datetime, **kwargs) -> pd.DataFrame:
        """获取恐慌贪婪指数"""
        try:
            # 计算天数
            days = (end_date - start_date).days
            
            url = f"https://api.alternative.me/fng/?limit={days}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                    df.set_index('date', inplace=True)
                    df['value'] = pd.to_numeric(df['value'])
                    result = df[['value']]
                    result.columns = ['fear_greed_index']
                    return result.sort_index()
                    
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching fear-greed index: {e}")
            return pd.DataFrame()


class MultiSourceDataFetcher:
    """多源数据获取管理器"""
    
    def __init__(self):
        """初始化各数据源获取器"""
        self.fetchers = {
            'yahoo': YahooFetcher(),
            'fred': FredFetcher(),
            'binance': BinanceFetcher(),
            'coingecko': CoinGeckoFetcher(),
            'cryptocompare': CryptoCompareFetcher(),
            'alternative_me': AlternativeMeFetcher()
        }
        
    def fetch_factor(self, factor_info: Dict, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """获取单个因子数据"""
        source = factor_info.get('source')
        symbol = factor_info.get('symbol')
        
        if source in self.fetchers:
            return self.fetchers[source].fetch(symbol, start_date, end_date)
        elif source == 'calculated':
            # 技术指标需要单独计算
            return pd.DataFrame()
        else:
            print(f"Unsupported data source: {source}")
            return pd.DataFrame()
    
    def fetch_btc_price(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """获取BTC价格数据（作为基准）"""
        # 优先使用CryptoCompare
        fetcher = self.fetchers['cryptocompare']
        df = fetcher.fetch("BTC", start_date, end_date)
        
        if df.empty:
            # 备用Binance
            fetcher = self.fetchers['binance']
            df = fetcher.fetch("BTCUSDT", start_date, end_date)
            
        if not df.empty:
            df.columns = ['BTC_Price']
            
        return df
    
    def fetch_all_factors(self, factor_pool: Dict, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """获取所有因子数据"""
        all_data = {}
        
        # 首先获取BTC价格作为基准
        btc_price = self.fetch_btc_price(start_date, end_date)
        if not btc_price.empty:
            all_data['BTC_Price'] = btc_price
        
        # 获取各类因子
        for category, factors in factor_pool.items():
            print(f"\nFetching {category} factors...")
            
            for factor_id, factor_info in factors.items():
                print(f"  - {factor_id}: {factor_info['name']}...", end=' ')
                
                df = self.fetch_factor(factor_info, start_date, end_date)
                
                if not df.empty:
                    all_data[factor_id] = df
                    print(f"✓ ({len(df)} records)")
                else:
                    print("✗ (no data)")
                
                time.sleep(0.2)  # Rate limiting
        
        # 合并所有数据
        if all_data:
            # 对齐所有数据到相同的日期索引
            combined = pd.concat(all_data.values(), axis=1, join='outer')
            combined = combined.sort_index()
            
            # 前向填充缺失值（某些数据源周末无数据）
            combined = combined.ffill()
            
            return combined
        
        return pd.DataFrame()


def calculate_technical_indicators(price_data: pd.Series) -> pd.DataFrame:
    """计算技术指标"""
    tech_indicators = pd.DataFrame(index=price_data.index)
    
    # RSI
    delta = price_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    tech_indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = price_data.ewm(span=12, adjust=False).mean()
    exp2 = price_data.ewm(span=26, adjust=False).mean()
    tech_indicators['MACD'] = exp1 - exp2
    tech_indicators['MACD_Signal'] = tech_indicators['MACD'].ewm(span=9, adjust=False).mean()
    tech_indicators['MACD_Histogram'] = tech_indicators['MACD'] - tech_indicators['MACD_Signal']
    
    # Bollinger Bands
    sma = price_data.rolling(window=20).mean()
    std = price_data.rolling(window=20).std()
    tech_indicators['BB_Upper'] = sma + (std * 2)
    tech_indicators['BB_Lower'] = sma - (std * 2)
    tech_indicators['BB_Width'] = tech_indicators['BB_Upper'] - tech_indicators['BB_Lower']
    tech_indicators['BB_Position'] = (price_data - tech_indicators['BB_Lower']) / tech_indicators['BB_Width']
    
    # Moving Averages
    for period in [7, 14, 30, 50, 100, 200]:
        tech_indicators[f'SMA_{period}'] = price_data.rolling(window=period).mean()
        tech_indicators[f'EMA_{period}'] = price_data.ewm(span=period, adjust=False).mean()
    
    # Volatility
    tech_indicators['Volatility_7d'] = price_data.pct_change().rolling(window=7).std() * np.sqrt(365)
    tech_indicators['Volatility_30d'] = price_data.pct_change().rolling(window=30).std() * np.sqrt(365)
    
    # Returns
    for period in [1, 7, 14, 30, 60, 90]:
        tech_indicators[f'Return_{period}d'] = price_data.pct_change(periods=period)
    
    return tech_indicators


if __name__ == "__main__":
    # 测试数据获取
    from btc_factor_config import FACTOR_POOL
    
    fetcher = MultiSourceDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print("Testing data fetchers...")
    print("="*60)
    
    # 测试获取BTC价格
    btc = fetcher.fetch_btc_price(start_date, end_date)
    print(f"BTC Price: {len(btc)} records")
    if not btc.empty:
        print(f"  Latest: ${btc.iloc[-1].values[0]:,.2f}")
    
    # 测试获取部分因子
    test_factors = {
        'macro': {
            'DXY': FACTOR_POOL['macro']['DXY'],
            'GOLD': FACTOR_POOL['macro']['GOLD'],
            'VIX': FACTOR_POOL['macro']['VIX']
        }
    }
    
    data = fetcher.fetch_all_factors(test_factors, start_date, end_date)
    print(f"\nCombined data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Missing values:\n{data.isnull().sum()}")
