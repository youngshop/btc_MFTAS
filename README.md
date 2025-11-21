# 📈 BTC Multi-Factor Trading Analysis System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

## 🚀 Overview

Professional cryptocurrency quantitative analysis platform based on 22 production-verified factors with deep correlation analysis.

### ✨ Features

- **Multi-Factor Analysis**: 22+ verified factors including macro, technical, and sentiment indicators
- **Real-time Data**: Live BTC price and indicator updates
- **Trading Signals**: Automated signal generation based on factor analysis  
- **Correlation Analysis**: Deep correlation and lead/lag analysis
- **Professional Visualization**: Interactive charts with Plotly

## 🔬 Core Factors

Based on deep analysis results:

| Factor | Score | Correlation | Type |
|--------|-------|-------------|------|
| BB_Width | 78.2 | 0.305 | Volatility |
| ETH/BTC | 76.7 | -0.727 | Market Rotation |
| DFF | 56.7 | -0.887 | Macro |
| Return_90d | 67.5 | 0.094 | Momentum |

## 📊 Live Demo

🔗 [View Live App](https://your-app-name.streamlit.app)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Data Source**: CryptoCompare API

## 💻 Local Development

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/btc-factor-trading.git
cd btc-factor-trading

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app_cloud.py
```

## 📁 Project Structure

```
btc-factor-trading/
├── app_cloud.py           # Main cloud application
├── realtime_trading_monitor.py  # Full trading monitor (local)
├── btc_factor_app.py      # Complete factor analysis
├── trading_executor.py    # Trading execution engine
├── data_fetcher.py        # Data fetching modules
├── factor_analyzer.py     # Factor analysis engine
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## 📈 Trading Strategies

### Strategy A: Volatility Breakout
- Signal: BB_Width > 2σ + ETH/BTC < -1σ
- Expected Return: 5-8% monthly
- Max Drawdown: <15%

### Strategy B: Macro Hedge
- Signal: Based on Federal Funds Rate (DFF)
- Correlation: -88.7%
- Confidence: High

## ⚠️ Disclaimer

- This analysis is for reference only
- Cryptocurrency markets are highly volatile
- Not financial advice
- Test with small amounts first

## ⚠️ 免责声明
- 此分析仅供参考
- 加密货币市场波动性极大
- 不构成任何财务建议
- 请先小额交易测试，避免风险

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

MIT License

## 📧 Contact

- Email: youngshop@qq.com

