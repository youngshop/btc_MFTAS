"""
BTC 多因子研究系统 - 因子池配置
包含宏观经济、市场情绪、链上数据、技术指标等多维度因子
"""

# 因子池定义
FACTOR_POOL = {
    # 1. 宏观经济因子
    "macro": {
        "DXY": {
            "name": "美元指数",
            "source": "yahoo",
            "symbol": "DX-Y.NYB",
            "description": "美元强弱对BTC的影响",
            "hypothesis": "负相关",
            "frequency": "daily"
        },
        "GOLD": {
            "name": "黄金价格",
            "source": "yahoo",
            "symbol": "GC=F",
            "description": "避险资产联动",
            "hypothesis": "正相关",
            "frequency": "daily"
        },
        "VIX": {
            "name": "恐慌指数",
            "source": "yahoo",
            "symbol": "^VIX",
            "description": "市场风险情绪",
            "hypothesis": "负相关",
            "frequency": "daily"
        },
        "US10Y": {
            "name": "美国10年期国债收益率",
            "source": "yahoo",
            "symbol": "^TNX",
            "description": "无风险利率变化",
            "hypothesis": "负相关",
            "frequency": "daily"
        },
        "SPY": {
            "name": "标普500指数",
            "source": "yahoo",
            "symbol": "SPY",
            "description": "股市风险偏好",
            "hypothesis": "正相关",
            "frequency": "daily"
        },
        "FED_RATE": {
            "name": "联邦基金利率",
            "source": "fred",
            "symbol": "DFF",
            "description": "货币政策立场",
            "hypothesis": "负相关",
            "frequency": "daily"
        },
        "M2": {
            "name": "M2货币供应量",
            "source": "fred",
            "symbol": "M2SL",
            "description": "流动性供给",
            "hypothesis": "正相关",
            "frequency": "monthly"
        },
        "CPI": {
            "name": "美国CPI",
            "source": "fred",
            "symbol": "CPIAUCSL",
            "description": "通胀水平",
            "hypothesis": "复杂关系",
            "frequency": "monthly"
        }
    },
    
    # 2. 加密市场因子
    "crypto_market": {
        "ETH_BTC": {
            "name": "ETH/BTC汇率",
            "source": "binance",
            "symbol": "ETHBTC",
            "description": "山寨币市场情绪",
            "hypothesis": "市场风险指标",
            "frequency": "daily"
        },
        "TOTAL_MCAP": {
            "name": "加密货币总市值",
            "source": "coingecko",
            "symbol": "total_market_cap",
            "description": "整体市场规模",
            "hypothesis": "正相关",
            "frequency": "daily"
        },
        "BTC_DOMINANCE": {
            "name": "BTC市值占比",
            "source": "coingecko",
            "symbol": "btc_dominance",
            "description": "资金聚集度",
            "hypothesis": "复杂关系",
            "frequency": "daily"
        },
        "STABLE_MCAP": {
            "name": "稳定币总市值",
            "source": "coingecko",
            "symbol": "stable_market_cap",
            "description": "市场流动性",
            "hypothesis": "正相关",
            "frequency": "daily"
        }
    },
    
    # 3. 链上数据因子
    "on_chain": {
        "HASH_RATE": {
            "name": "算力",
            "source": "blockchain_com",
            "symbol": "hash-rate",
            "description": "网络安全性",
            "hypothesis": "正相关",
            "frequency": "daily"
        },
        "ACTIVE_ADDRESSES": {
            "name": "活跃地址数",
            "source": "glassnode",
            "symbol": "active_addresses",
            "description": "网络活跃度",
            "hypothesis": "正相关",
            "frequency": "daily"
        },
        "EXCHANGE_BALANCE": {
            "name": "交易所余额",
            "source": "glassnode",
            "symbol": "exchange_balance",
            "description": "卖压指标",
            "hypothesis": "负相关",
            "frequency": "daily"
        },
        "MINER_REVENUE": {
            "name": "矿工收入",
            "source": "blockchain_com",
            "symbol": "miners-revenue",
            "description": "矿工盈利状况",
            "hypothesis": "正相关",
            "frequency": "daily"
        },
        "NVT": {
            "name": "NVT比率",
            "source": "glassnode",
            "symbol": "nvt_ratio",
            "description": "估值指标",
            "hypothesis": "负相关",
            "frequency": "daily"
        }
    },
    
    # 4. 技术指标
    "technical": {
        "RSI": {
            "name": "相对强弱指数",
            "source": "calculated",
            "period": 14,
            "description": "超买超卖",
            "hypothesis": "反向指标",
            "frequency": "daily"
        },
        "MACD": {
            "name": "MACD",
            "source": "calculated",
            "fast": 12,
            "slow": 26,
            "signal": 9,
            "description": "趋势动量",
            "hypothesis": "趋势跟随",
            "frequency": "daily"
        },
        "BOLLINGER": {
            "name": "布林带",
            "source": "calculated",
            "period": 20,
            "std": 2,
            "description": "波动率通道",
            "hypothesis": "均值回归",
            "frequency": "daily"
        },
        "VOLUME": {
            "name": "成交量",
            "source": "exchange",
            "description": "市场参与度",
            "hypothesis": "正相关",
            "frequency": "daily"
        }
    },
    
    # 5. 市场情绪因子
    "sentiment": {
        "FEAR_GREED": {
            "name": "恐慌贪婪指数",
            "source": "alternative_me",
            "symbol": "fear-greed-index",
            "description": "市场情绪综合指标",
            "hypothesis": "反向指标",
            "frequency": "daily"
        },
        "FUNDING_RATE": {
            "name": "永续合约资金费率",
            "source": "binance",
            "symbol": "funding_rate",
            "description": "多空情绪",
            "hypothesis": "反向指标",
            "frequency": "8hours"
        },
        "LONG_SHORT_RATIO": {
            "name": "多空比",
            "source": "binance",
            "symbol": "long_short_ratio",
            "description": "散户情绪",
            "hypothesis": "反向指标",
            "frequency": "daily"
        },
        "GOOGLE_TRENDS": {
            "name": "谷歌搜索趋势",
            "source": "google",
            "keyword": "bitcoin",
            "description": "公众关注度",
            "hypothesis": "正相关",
            "frequency": "weekly"
        }
    },
    
    # 6. 事件驱动因子
    "event": {
        "HALVING": {
            "name": "减半周期",
            "source": "calculated",
            "description": "供给冲击",
            "hypothesis": "正向影响",
            "frequency": "event"
        },
        "REGULATION": {
            "name": "监管事件",
            "source": "news",
            "description": "政策影响",
            "hypothesis": "双向影响",
            "frequency": "event"
        },
        "ADOPTION": {
            "name": "机构采用",
            "source": "news",
            "description": "需求增长",
            "hypothesis": "正向影响",
            "frequency": "event"
        }
    }
}

# 数据源配置
DATA_SOURCES = {
    "yahoo": {
        "api": "yfinance",
        "rate_limit": 2000,  # 每日请求限制
        "requires_key": False
    },
    "fred": {
        "api": "pandas_datareader",
        "rate_limit": None,
        "requires_key": False
    },
    "binance": {
        "api": "https://api.binance.com",
        "rate_limit": 1200,  # 每分钟
        "requires_key": False
    },
    "coingecko": {
        "api": "https://api.coingecko.com/api/v3",
        "rate_limit": 50,  # 每分钟
        "requires_key": False
    },
    "cryptocompare": {
        "api": "https://min-api.cryptocompare.com",
        "rate_limit": 100000,  # 每月
        "requires_key": True
    },
    "glassnode": {
        "api": "https://api.glassnode.com",
        "rate_limit": 10,  # 每分钟(免费版)
        "requires_key": True
    },
    "blockchain_com": {
        "api": "https://api.blockchain.info",
        "rate_limit": None,
        "requires_key": False
    },
    "alternative_me": {
        "api": "https://api.alternative.me",
        "rate_limit": None,
        "requires_key": False
    }
}

# 分析参数配置
ANALYSIS_CONFIG = {
    "correlation": {
        "methods": ["pearson", "spearman", "kendall"],
        "lags": [-30, -21, -14, -7, -3, -1, 0, 1, 3, 7, 14, 21, 30],  # 领先/滞后天数
        "rolling_windows": [30, 60, 90, 180, 365],  # 滚动相关性窗口
        "min_observations": 30  # 最小观测数
    },
    "preprocessing": {
        "outlier_method": "iqr",  # 异常值处理: iqr, zscore, none
        "outlier_threshold": 3,
        "missing_method": "interpolate",  # 缺失值: forward_fill, interpolate, drop
        "normalization": "zscore",  # 标准化: zscore, minmax, robust
        "detrending": False  # 是否去趋势
    },
    "evaluation": {
        "stability_window": 90,  # 稳定性评估窗口
        "significance_level": 0.05,  # 显著性水平
        "bootstrap_iterations": 1000,  # Bootstrap迭代次数
        "out_of_sample_ratio": 0.2  # 样本外测试比例
    }
}

# 策略参数
STRATEGY_CONFIG = {
    "factor_selection": {
        "min_correlation": 0.3,  # 最小相关性阈值
        "max_correlation": 0.9,  # 最大相关性（避免多重共线性）
        "min_stability": 0.7,  # 最小稳定性评分
        "max_factors": 10  # 最大因子数量
    },
    "signal_generation": {
        "method": "zscore",  # 信号生成方法
        "entry_threshold": 2,  # 入场阈值
        "exit_threshold": 0,  # 出场阈值
        "lookback": 60  # 回看周期
    },
    "risk_management": {
        "max_position": 1.0,  # 最大仓位
        "stop_loss": 0.05,  # 止损
        "take_profit": 0.15,  # 止盈
        "max_drawdown": 0.2  # 最大回撤
    }
}
