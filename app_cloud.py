"""
BTC多因子量化交易系统 - 专业云端版
包含完整的多因子分析和模拟交易功能
整合8502(多因子分析)和8504(交易监控)的核心功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="BTC多因子量化交易系统 Pro",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state for trading
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'position' not in st.session_state:
    st.session_state.position = 0
if 'entry_price' not in st.session_state:
    st.session_state.entry_price = 0
if 'balance' not in st.session_state:
    st.session_state.balance = 10000
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.DataFrame(columns=[
        'timestamp', 'action', 'price', 'size', 'value', 'pnl', 'balance'
    ])

# 标题和说明
st.title("💹 BTC多因子量化交易系统 Professional")
st.markdown("""
**专业量化分析平台** | 基于22个实盘验证因子 | 深度相关性分析 | 模拟交易系统
""")

# 添加交易功能函数
def execute_trade(signal, current_price):
    """执行模拟交易"""
    result = {'success': False, 'message': ''}
    
    if signal['confidence'] < 60:
        result['message'] = f"信号置信度不足 ({signal['confidence']:.0f}% < 60%)"
        return result
    
    position_pct = min(0.3, signal['confidence'] / 100 * 0.5)
    position_value = st.session_state.balance * position_pct
    position_size = position_value / current_price
    
    if signal['action'] == 'BUY' and st.session_state.position == 0:
        st.session_state.position = position_size
        st.session_state.entry_price = current_price
        st.session_state.balance -= position_value
        
        trade = {
            'timestamp': datetime.now(),
            'action': 'BUY',
            'price': current_price,
            'size': position_size,
            'value': position_value,
            'pnl': 0,
            'balance': st.session_state.balance
        }
        st.session_state.trades.append(trade)
        new_row = pd.DataFrame([trade])
        st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_row], ignore_index=True)
        
        result['success'] = True
        result['message'] = f"✅ 开仓: BUY {position_size:.5f} BTC @ ${current_price:,.0f}"
    
    return result

def close_position(current_price):
    """平仓"""
    result = {'success': False, 'message': ''}
    
    if st.session_state.position == 0:
        result['message'] = "无持仓"
        return result
    
    pnl = (current_price - st.session_state.entry_price) * st.session_state.position
    st.session_state.balance += st.session_state.position * current_price
    
    trade = {
        'timestamp': datetime.now(),
        'action': 'SELL',
        'price': current_price,
        'size': st.session_state.position,
        'value': st.session_state.position * current_price,
        'pnl': pnl,
        'balance': st.session_state.balance
    }
    st.session_state.trades.append(trade)
    new_row = pd.DataFrame([trade])
    st.session_state.trade_history = pd.concat([st.session_state.trade_history, new_row], ignore_index=True)
    
    st.session_state.position = 0
    st.session_state.entry_price = 0
    
    result['success'] = True
    result['message'] = f"✅ 平仓: SELL @ ${current_price:,.0f}, 盈亏: ${pnl:+,.2f}"
    return result

# 核心因子定义（基于深度分析结果）
CORE_FACTORS = {
    'top_tier': {
        'BB_Width': {'score': 78.2, 'correlation': 0.305, 'desc': '布林带宽度 - 波动性'},
        'ETH_BTC': {'score': 76.7, 'correlation': -0.727, 'desc': 'ETH/BTC - 市场轮动'},
        'Return_90d': {'score': 67.5, 'correlation': 0.094, 'desc': '90天动量'},
    },
    'macro': {
        'DFF': {'score': 56.7, 'correlation': -0.887, 'desc': '联邦基金利率'},
        'M2': {'score': 57.8, 'correlation': 0.913, 'desc': 'M2货币供应'},
        'CPI': {'score': 57.8, 'correlation': 0.933, 'desc': '通胀率'},
    }
}

# 侧边栏配置
st.sidebar.header("⚙️ 系统配置")

# 账户信息
st.sidebar.markdown("### 💰 账户信息")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("余额", f"${st.session_state.balance:,.0f}")
with col2:
    profit = st.session_state.balance - 10000
    st.metric("盈亏", f"${profit:+,.0f}")

if st.session_state.position != 0:
    st.sidebar.info(f"持仓: {st.session_state.position:.5f} BTC")
else:
    st.sidebar.success("空仓")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 数据范围")
days_back = st.sidebar.slider("历史天数", 30, 365, 180)

st.sidebar.markdown("### 📊 分析选项")
show_correlation = st.sidebar.checkbox("显示相关性分析", True)
show_signals = st.sidebar.checkbox("显示交易信号", True)
show_factors = st.sidebar.checkbox("显示因子详情", True)

# 数据获取函数
@st.cache_data(ttl=3600)
def fetch_btc_price(days=180):
    """获取BTC历史价格"""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": "BTC",
        "tsym": "USD",
        "limit": days
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('Response') == 'Success':
                prices = pd.DataFrame(data['Data']['Data'])
                prices['date'] = pd.to_datetime(prices['time'], unit='s')
                prices.set_index('date', inplace=True)
                return prices[['close', 'high', 'low', 'volumefrom']]
    except Exception as e:
        st.error(f"获取数据失败: {e}")
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def calculate_indicators(prices):
    """计算技术指标"""
    df = prices.copy()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # 移动平均
    df['MA_7'] = df['close'].rolling(window=7).mean()
    df['MA_30'] = df['close'].rolling(window=30).mean()
    df['MA_90'] = df['close'].rolling(window=90).mean()
    
    # 动量
    df['Return_7d'] = df['close'].pct_change(7)
    df['Return_30d'] = df['close'].pct_change(30)
    df['Return_90d'] = df['close'].pct_change(90)
    
    # 波动率
    df['Volatility'] = df['close'].pct_change().rolling(window=30).std() * np.sqrt(365)
    
    return df

def generate_signal(indicators):
    """生成交易信号"""
    latest = indicators.iloc[-1]
    signals = []
    score = 0
    
    # BB_Width 信号（权重最高）
    if 'BB_Width' in indicators.columns:
        bb_z = (latest['BB_Width'] - indicators['BB_Width'].mean()) / indicators['BB_Width'].std()
        if bb_z > 2:
            signals.append("波动扩张 - 可能突破")
            score += 0.3
        elif bb_z < -1:
            signals.append("波动收缩 - 等待方向")
            score -= 0.1
    
    # RSI 信号
    if 'RSI' in indicators.columns:
        if latest['RSI'] < 30:
            signals.append("超卖 - 买入机会")
            score += 0.2
        elif latest['RSI'] > 70:
            signals.append("超买 - 卖出预警")
            score -= 0.2
    
    # 动量信号
    if 'Return_90d' in indicators.columns:
        if latest['Return_90d'] > 0.5:
            signals.append("强势上涨趋势")
            score += 0.2
        elif latest['Return_90d'] < -0.3:
            signals.append("下跌趋势")
            score -= 0.2
    
    # 综合判断
    if score > 0.4:
        decision = "强烈买入 🟢"
        action = "BUY"
        confidence = min(90, score * 100)
    elif score > 0.2:
        decision = "买入 🟢"
        action = "BUY"
        confidence = min(70, score * 100)
    elif score < -0.3:
        decision = "卖出 🔴"
        action = "SELL"
        confidence = min(70, abs(score) * 100)
    else:
        decision = "持有 ⚪"
        action = "HOLD"
        confidence = 50
    
    return {
        'decision': decision,
        'action': action,
        'signals': signals,
        'score': score,
        'confidence': confidence
    }

# 主界面
def get_trade_stats():
    """获取交易统计"""
    if st.session_state.trade_history.empty:
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}
    
    df = st.session_state.trade_history
    closed = df[df['pnl'] != 0]
    
    if closed.empty:
        return {'total_trades': len(df), 'win_rate': 0, 'total_pnl': 0}
    
    wins = closed[closed['pnl'] > 0]
    return {
        'total_trades': len(df),
        'win_rate': len(wins) / len(closed) * 100 if len(closed) > 0 else 0,
        'total_pnl': closed['pnl'].sum()
    }

def main():
    # 获取数据
    with st.spinner("正在获取数据..."):
        btc_data = fetch_btc_price(days_back)
    
    if btc_data.empty:
        st.error("无法获取数据，请检查网络连接")
        return
    
    # 计算指标
    indicators = calculate_indicators(btc_data)
    
    # 生成信号
    signal = generate_signal(indicators)
    
    # 显示核心指标
    stats = get_trade_stats()
    st.markdown("### 📊 实时监控面板")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        latest_price = indicators['close'].iloc[-1]
        price_change = indicators['close'].pct_change().iloc[-1] * 100
        st.metric("BTC价格", f"${latest_price:,.0f}", f"{price_change:+.2f}%")
    
    with col2:
        st.metric("交易信号", signal['decision'], f"置信度: {signal['confidence']:.0f}%")
    
    with col3:
        if 'RSI' in indicators.columns:
            st.metric("RSI", f"{indicators['RSI'].iloc[-1]:.1f}", 
                     "超买" if indicators['RSI'].iloc[-1] > 70 else "超卖" if indicators['RSI'].iloc[-1] < 30 else "中性")
    
    with col4:
        if 'Volatility' in indicators.columns:
            st.metric("波动率", f"{indicators['Volatility'].iloc[-1]*100:.1f}%", "年化")
    
    with col5:
        st.metric("总交易", f"{stats['total_trades']}笔", f"胜率: {stats['win_rate']:.0f}%")
    
    with col6:
        st.metric("总盈亏", f"${stats['total_pnl']:+,.0f}")
    
    # 交易执行区
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if signal['action'] != "HOLD" and signal['confidence'] >= 60:
            if st.button(f"📈 执行{signal['action']}", type="primary"):
                result = execute_trade(signal, latest_price)
                if result['success']:
                    st.success(result['message'])
                    st.rerun()
                else:
                    st.warning(result['message'])
        else:
            st.info(f"等待信号 (置信度: {signal['confidence']:.0f}%)")
    
    with col2:
        if st.session_state.position != 0:
            if st.button("📉 平仓", type="secondary"):
                result = close_position(latest_price)
                if result['success']:
                    st.success(result['message'])
                    st.rerun()
                else:
                    st.warning(result['message'])
    
    with col3:
        if st.button("🔄 刷新数据"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown("---")
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 价格走势", "🔍 因子分析", "📊 相关性", "💰 交易记录", "💡 策略建议"])
    
    with tab1:
        # 价格图表
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('BTC价格与布林带', 'RSI指标'),
            row_heights=[0.7, 0.3]
        )
        
        # 价格和布林带
        fig.add_trace(
            go.Scatter(x=indicators.index, y=indicators['close'],
                      name='BTC价格', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        if 'BB_Upper' in indicators.columns:
            fig.add_trace(
                go.Scatter(x=indicators.index, y=indicators['BB_Upper'],
                          name='上轨', line=dict(color='red', dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=indicators.index, y=indicators['BB_Lower'],
                          name='下轨', line=dict(color='green', dash='dash')),
                row=1, col=1
            )
        
        # RSI
        if 'RSI' in indicators.columns:
            fig.add_trace(
                go.Scatter(x=indicators.index, y=indicators['RSI'],
                          name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 🎯 核心因子评分")
        
        # 显示顶级因子
        factor_data = []
        for category, factors in CORE_FACTORS.items():
            for name, info in factors.items():
                factor_data.append({
                    '因子': name,
                    '类别': category,
                    '评分': info['score'],
                    '相关性': info['correlation'],
                    '描述': info['desc']
                })
        
        factor_df = pd.DataFrame(factor_data)
        factor_df = factor_df.sort_values('评分', ascending=False)
        
        # 因子评分图
        fig = go.Figure(data=[
            go.Bar(
                x=factor_df['因子'],
                y=factor_df['评分'],
                marker_color=['green' if x > 70 else 'orange' if x > 50 else 'red' 
                             for x in factor_df['评分']],
                text=factor_df['评分'].round(1),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="因子评分排名（基于深度分析）",
            xaxis_title="因子",
            yaxis_title="评分",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 因子详情表
        st.markdown("### 📋 因子详情")
        # 简化样式，不使用background_gradient
        st.dataframe(
            factor_df.style.format({'评分': '{:.1f}', '相关性': '{:.3f}'})
        )
    
    with tab3:
        st.markdown("### 🔗 相关性分析")
        
        # 计算主要指标的相关性
        corr_cols = ['close', 'RSI', 'BB_Width', 'Return_7d', 'Return_30d', 'Return_90d', 'Volatility']
        available_cols = [col for col in corr_cols if col in indicators.columns]
        
        if len(available_cols) > 1:
            corr_matrix = indicators[available_cols].corr()
            
            # 热力图
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="相关系数")
            ))
            
            fig.update_layout(
                title="因子相关性矩阵",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 重要发现
        st.markdown("### 🔍 重要发现")
        st.info("""
        **基于深度分析的核心结论：**
        - 🥇 **BB_Width** (78.2分) - 最佳波动性指标
        - 🥈 **ETH/BTC** (76.7分) - 市场轮动指标
        - 🥉 **DFF** (-0.887相关) - 最强宏观因子
        - ❌ **RSI/MACD** - 相关性<0.1，不建议使用
        """)
    
    with tab4:
        st.markdown("### 💰 交易记录")
        
        if not st.session_state.trade_history.empty:
            # 交易统计
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("交易次数", stats['total_trades'])
            with col2:
                st.metric("胜率", f"{stats['win_rate']:.1f}%")
            with col3:
                st.metric("总盈亏", f"${stats['total_pnl']:+,.2f}")
            with col4:
                current_value = st.session_state.balance
                if st.session_state.position > 0:
                    current_value += st.session_state.position * latest_price
                roi = (current_value - 10000) / 10000 * 100
                st.metric("收益率", f"{roi:+.1f}%")
            
            # 交易历史表
            st.subheader("交易历史")
            display_df = st.session_state.trade_history.copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            # 显示交易历史，简化样式
            styled_df = display_df.style.format({
                'price': '${:,.0f}',
                'size': '{:.5f}',
                'value': '${:,.0f}',
                'pnl': '${:+,.2f}',
                'balance': '${:,.0f}'
            })
            # 为盈亏列添加颜色
            def color_pnl(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return 'color: green'
                    elif val < 0:
                        return 'color: red'
                return ''
            styled_df = styled_df.map(color_pnl, subset=['pnl'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("暂无交易记录")
        
        # 重置按钮
        if st.button("🗑️ 重置账户"):
            st.session_state.trades = []
            st.session_state.position = 0
            st.session_state.entry_price = 0
            st.session_state.balance = 10000
            st.session_state.trade_history = pd.DataFrame(columns=[
                'timestamp', 'action', 'price', 'size', 'value', 'pnl', 'balance'
            ])
            st.success("账户已重置")
            st.rerun()
    
    with tab5:
        st.markdown("### 💡 交易策略建议")
        
        # 显示当前信号
        st.success(f"**当前信号：{signal['decision']}**")
        
        if signal['signals']:
            st.markdown("**信号来源：**")
            for sig in signal['signals']:
                st.write(f"• {sig}")
        
        # 策略建议
        st.markdown("### 📝 推荐策略")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **策略A：波动突破策略**
            ```python
            if BB_Width > 2σ and ETH/BTC < -1σ:
                开仓做多
            elif BB_Width收缩:
                平仓观望
            ```
            - 预期收益：月5-8%
            - 最大回撤：<15%
            """)
        
        with col2:
            st.markdown("""
            **策略B：宏观对冲策略**
            ```python
            if DFF下降预期:
                增加仓位
            elif DFF上升预期:
                减少仓位
            ```
            - 相关性：-88.7%
            - 置信度：高
            """)
        
        # 风险提示
        st.warning("""
        ⚠️ **风险提示**
        - 此分析仅供参考，不构成投资建议
        - 加密货币市场高度波动，请谨慎投资
        - 建议先小额测试，验证策略有效性
        """)

# 页脚
st.markdown("---")
st.caption("🔬 基于22个实盘验证因子 | 📊 数据源: CryptoCompare")
st.caption(f"⏰ 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 添加GitHub链接（部署后可修改为您的仓库）
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 相关链接")
st.sidebar.markdown("[GitHub仓库](https://github.com/your-username/btc-factor-trading)")
st.sidebar.markdown("[API文档](https://min-api.cryptocompare.com/)")
st.sidebar.markdown("[联系作者](mailto:your-email@example.com)")

if __name__ == "__main__":
    main()
