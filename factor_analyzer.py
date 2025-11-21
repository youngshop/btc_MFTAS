"""
BTC 多因子研究系统 - 因子分析引擎
包含相关性分析、领先/滞后关系、稳定性评估等
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Dict, List, Tuple, Optional
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


class FactorPreprocessor:
    """因子预处理器"""
    
    def __init__(self, config: Dict):
        """初始化预处理器"""
        self.config = config
        self.scalers = {}
        
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 3) -> pd.DataFrame:
        """去除异常值"""
        df_clean = df.copy()
        
        if method == 'iqr':
            for col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df_clean[col] = df[col].clip(lower, upper)
                
        elif method == 'zscore':
            for col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df_clean[col] = df[col].where(z_scores < threshold, df[col].median())
                
        return df_clean
    
    def normalize(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """数据标准化"""
        df_norm = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            if method == 'zscore':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                df_norm[col] = df[col]
                continue
                
            # Fit and transform
            values = df[col].values.reshape(-1, 1)
            df_norm[col] = scaler.fit_transform(values).flatten()
            self.scalers[col] = scaler
            
        return df_norm
    
    def handle_missing(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """处理缺失值"""
        if method == 'forward_fill':
            return df.ffill().bfill()
        elif method == 'interpolate':
            return df.interpolate(method='linear')
        elif method == 'drop':
            return df.dropna()
        else:
            return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整的预处理流程"""
        # 1. 处理缺失值
        df = self.handle_missing(df, self.config.get('missing_method', 'forward_fill'))
        
        # 2. 去除异常值
        df = self.remove_outliers(
            df, 
            self.config.get('outlier_method', 'iqr'),
            self.config.get('outlier_threshold', 3)
        )
        
        # 3. 标准化
        df = self.normalize(df, self.config.get('normalization', 'zscore'))
        
        return df


class CorrelationAnalyzer:
    """相关性分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.results = {}
        
    def calculate_correlation(self, x: pd.Series, y: pd.Series, method: str = 'pearson') -> Tuple[float, float]:
        """计算两个序列的相关性"""
        # 删除缺失值
        valid_idx = x.notna() & y.notna()
        x_clean = x[valid_idx]
        y_clean = y[valid_idx]
        
        if len(x_clean) < 30:  # 最小样本数
            return np.nan, np.nan
            
        if method == 'pearson':
            corr, pval = pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr, pval = spearmanr(x_clean, y_clean)
        elif method == 'kendall':
            corr, pval = kendalltau(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
            
        return corr, pval
    
    def calculate_lagged_correlation(self, x: pd.Series, y: pd.Series, 
                                   lags: List[int], method: str = 'pearson') -> pd.DataFrame:
        """计算领先/滞后相关性"""
        results = []
        
        for lag in lags:
            if lag < 0:
                # x领先y
                x_shifted = x.shift(-lag)
                corr, pval = self.calculate_correlation(x_shifted, y, method)
                interpretation = f"Factor leads by {-lag} days"
            elif lag > 0:
                # x滞后y
                y_shifted = y.shift(lag)
                corr, pval = self.calculate_correlation(x, y_shifted, method)
                interpretation = f"Factor lags by {lag} days"
            else:
                # 同期
                corr, pval = self.calculate_correlation(x, y, method)
                interpretation = "Concurrent"
                
            results.append({
                'lag': lag,
                'correlation': corr,
                'p_value': pval,
                'significant': pval < 0.05 if not np.isnan(pval) else False,
                'interpretation': interpretation
            })
            
        return pd.DataFrame(results)
    
    def calculate_rolling_correlation(self, x: pd.Series, y: pd.Series, 
                                    windows: List[int], method: str = 'pearson') -> Dict:
        """计算滚动相关性（评估稳定性）"""
        rolling_corrs = {}
        
        for window in windows:
            corrs = []
            dates = []
            
            for i in range(window, len(x)):
                x_window = x.iloc[i-window:i]
                y_window = y.iloc[i-window:i]
                
                corr, _ = self.calculate_correlation(x_window, y_window, method)
                if not np.isnan(corr):
                    corrs.append(corr)
                    dates.append(x.index[i])
                    
            rolling_corrs[f'window_{window}'] = pd.Series(corrs, index=dates)
            
        return rolling_corrs
    
    def analyze_all_factors(self, data: pd.DataFrame, target_col: str = 'BTC_Price',
                           lags: List[int] = None, methods: List[str] = None) -> Dict:
        """分析所有因子"""
        if lags is None:
            lags = [-30, -14, -7, -3, -1, 0, 1, 3, 7, 14, 30]
        if methods is None:
            methods = ['pearson', 'spearman']
            
        results = {}
        target = data[target_col]
        
        for col in data.columns:
            if col == target_col:
                continue
                
            factor_results = {}
            
            # 1. 基础相关性
            for method in methods:
                corr, pval = self.calculate_correlation(data[col], target, method)
                factor_results[f'{method}_corr'] = corr
                factor_results[f'{method}_pval'] = pval
                
            # 2. 领先/滞后分析
            lag_df = self.calculate_lagged_correlation(data[col], target, lags, methods[0])
            factor_results['lag_analysis'] = lag_df
            
            # 找出最优lag
            best_lag_idx = lag_df['correlation'].abs().idxmax()
            if not pd.isna(best_lag_idx):
                factor_results['best_lag'] = lag_df.loc[best_lag_idx, 'lag']
                factor_results['best_lag_corr'] = lag_df.loc[best_lag_idx, 'correlation']
                
            # 3. 滚动相关性（稳定性）
            rolling = self.calculate_rolling_correlation(
                data[col], target, [30, 60, 90], methods[0]
            )
            
            # 计算稳定性指标
            if 'window_60' in rolling and len(rolling['window_60']) > 0:
                stability = 1 - rolling['window_60'].std()
                factor_results['stability_score'] = max(0, stability)
            else:
                factor_results['stability_score'] = 0
                
            factor_results['rolling_correlation'] = rolling
            
            results[col] = factor_results
            
        return results


class CausalityAnalyzer:
    """因果关系分析器"""
    
    def __init__(self):
        """初始化"""
        self.results = {}
        
    def granger_causality(self, data: pd.DataFrame, target: str, factor: str, 
                         max_lag: int = 10) -> Dict:
        """格兰杰因果检验"""
        try:
            # 准备数据
            df = data[[target, factor]].dropna()
            
            # 平稳性检验
            adf_target = adfuller(df[target])
            adf_factor = adfuller(df[factor])
            
            # 如果非平稳，进行差分
            if adf_target[1] > 0.05:
                df[target] = df[target].diff().dropna()
            if adf_factor[1] > 0.05:
                df[factor] = df[factor].diff().dropna()
                
            df = df.dropna()
            
            # 格兰杰因果检验
            results = grangercausalitytests(df[[target, factor]], maxlag=max_lag, verbose=False)
            
            # 提取p值
            p_values = {}
            for lag in range(1, max_lag + 1):
                p_values[lag] = results[lag][0]['ssr_ftest'][1]
                
            # 找出显著的lag
            significant_lags = [lag for lag, pval in p_values.items() if pval < 0.05]
            
            return {
                'p_values': p_values,
                'significant_lags': significant_lags,
                'is_causal': len(significant_lags) > 0,
                'min_pvalue': min(p_values.values())
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'is_causal': False
            }
    
    def analyze_causality_network(self, data: pd.DataFrame, target: str = 'BTC_Price') -> Dict:
        """分析因果关系网络"""
        results = {}
        
        for col in data.columns:
            if col == target:
                continue
                
            # Factor → BTC
            factor_to_btc = self.granger_causality(data, target, col)
            
            # BTC → Factor (反向)
            btc_to_factor = self.granger_causality(data, col, target)
            
            results[col] = {
                'factor_causes_btc': factor_to_btc,
                'btc_causes_factor': btc_to_factor,
                'bidirectional': factor_to_btc.get('is_causal', False) and btc_to_factor.get('is_causal', False)
            }
            
        return results


class FactorEvaluator:
    """因子评估器"""
    
    def __init__(self):
        """初始化"""
        self.scores = {}
        
    def calculate_information_coefficient(self, factor: pd.Series, returns: pd.Series) -> float:
        """计算信息系数IC"""
        try:
            # 确保有足够的数据
            if len(factor) < 2 or len(returns) < 2:
                return 0
            
            # 计算因子值与未来收益的相关性
            factor_clean = factor[:-1].dropna()
            returns_clean = returns[1:].dropna()
            
            # 对齐索引
            common_idx = factor_clean.index.intersection(returns_clean.index)
            if len(common_idx) < 2:
                return 0
                
            corr, _ = spearmanr(factor_clean[common_idx], returns_clean[common_idx])
            return corr if not np.isnan(corr) else 0
        except Exception:
            return 0
    
    def calculate_factor_return(self, factor: pd.Series, prices: pd.Series, 
                               quantiles: int = 5) -> pd.DataFrame:
        """计算因子收益（分组回测）"""
        # 根据因子值分组
        factor_quantiles = pd.qcut(factor, q=quantiles, labels=False)
        
        # 计算每组的平均收益
        returns = prices.pct_change()
        group_returns = []
        
        for q in range(quantiles):
            mask = factor_quantiles == q
            group_return = returns[mask].mean()
            group_returns.append({
                'quantile': q + 1,
                'mean_return': group_return,
                'count': mask.sum()
            })
            
        return pd.DataFrame(group_returns)
    
    def calculate_multicollinearity(self, factors: pd.DataFrame) -> pd.DataFrame:
        """计算多重共线性（VIF）"""
        vif_data = pd.DataFrame()
        vif_data["Factor"] = factors.columns
        vif_data["VIF"] = [variance_inflation_factor(factors.values, i) 
                          for i in range(len(factors.columns))]
        return vif_data
    
    def evaluate_factors(self, data: pd.DataFrame, correlation_results: Dict) -> pd.DataFrame:
        """综合评估所有因子"""
        evaluation = []
        
        btc_returns = data['BTC_Price'].pct_change()
        
        for factor, results in correlation_results.items():
            if factor == 'BTC_Price':
                continue
                
            eval_dict = {
                'factor': factor,
                'pearson_corr': results.get('pearson_corr', np.nan),
                'spearman_corr': results.get('spearman_corr', np.nan),
                'best_lag': results.get('best_lag', 0),
                'best_lag_corr': results.get('best_lag_corr', np.nan),
                'stability_score': results.get('stability_score', 0)
            }
            
            # 计算IC
            if factor in data.columns:
                ic = self.calculate_information_coefficient(data[factor], btc_returns)
                if not np.isnan(ic):
                    eval_dict['information_coefficient'] = ic
                else:
                    eval_dict['information_coefficient'] = 0
                
            # 综合评分
            scores = []
            
            # 相关性强度（40%）
            corr_score = abs(eval_dict['best_lag_corr']) if not np.isnan(eval_dict['best_lag_corr']) else 0
            scores.append(corr_score * 0.4)
            
            # 稳定性（30%）
            stability = eval_dict['stability_score'] if not np.isnan(eval_dict['stability_score']) else 0
            scores.append(stability * 0.3)
            
            # IC（20%）
            if 'information_coefficient' in eval_dict:
                ic_value = eval_dict['information_coefficient']
                ic_score = abs(ic_value) if not np.isnan(ic_value) else 0
                scores.append(ic_score * 0.2)
            else:
                scores.append(0)
                
            # 领先性（10%）- 领先指标加分
            if eval_dict['best_lag'] < 0:
                scores.append(0.1)
            else:
                scores.append(0)
                
            eval_dict['composite_score'] = sum(scores)
            
            evaluation.append(eval_dict)
            
        return pd.DataFrame(evaluation).sort_values('composite_score', ascending=False)


class FactorVisualizer:
    """因子可视化器"""
    
    @staticmethod
    def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = "Factor Correlation Heatmap"):
        """绘制相关性热力图"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   vmin=-1, vmax=1, fmt='.2f')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_lag_analysis(lag_results: pd.DataFrame, factor_name: str):
        """绘制领先/滞后分析图"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 相关性 vs Lag
        axes[0].bar(lag_results['lag'], lag_results['correlation'], 
                   color=['red' if x < 0 else 'blue' for x in lag_results['lag']])
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_xlabel('Lag (days)')
        axes[0].set_ylabel('Correlation')
        axes[0].set_title(f'{factor_name} - Lead/Lag Analysis')
        axes[0].grid(True, alpha=0.3)
        
        # P-value vs Lag
        axes[1].bar(lag_results['lag'], -np.log10(lag_results['p_value']), 
                   color=['green' if x < 0.05 else 'gray' for x in lag_results['p_value']])
        axes[1].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        axes[1].set_xlabel('Lag (days)')
        axes[1].set_ylabel('-log10(p-value)')
        axes[1].set_title('Statistical Significance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_rolling_correlation(rolling_corr: Dict, factor_name: str):
        """绘制滚动相关性图"""
        plt.figure(figsize=(12, 6))
        
        for window, corr_series in rolling_corr.items():
            if len(corr_series) > 0:
                plt.plot(corr_series.index, corr_series.values, 
                        label=window.replace('_', ' '), alpha=0.7)
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.title(f'{factor_name} - Rolling Correlation with BTC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_factor_ranking(evaluation_df: pd.DataFrame, top_n: int = 20):
        """绘制因子排名图"""
        top_factors = evaluation_df.head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 综合得分
        axes[0].barh(range(len(top_factors)), top_factors['composite_score'])
        axes[0].set_yticks(range(len(top_factors)))
        axes[0].set_yticklabels(top_factors['factor'])
        axes[0].set_xlabel('Composite Score')
        axes[0].set_title('Top Factors by Composite Score')
        axes[0].invert_yaxis()
        
        # 相关性vs稳定性
        scatter = axes[1].scatter(top_factors['best_lag_corr'].abs(), 
                                 top_factors['stability_score'],
                                 s=top_factors['composite_score']*200,
                                 alpha=0.6, c=top_factors['best_lag'])
        axes[1].set_xlabel('|Correlation|')
        axes[1].set_ylabel('Stability Score')
        axes[1].set_title('Correlation vs Stability')
        
        # 添加标签
        for i, row in top_factors.iterrows():
            axes[1].annotate(row['factor'], 
                           (abs(row['best_lag_corr']), row['stability_score']),
                           fontsize=8, alpha=0.7)
        
        plt.colorbar(scatter, ax=axes[1], label='Best Lag')
        plt.tight_layout()
        
        return fig


if __name__ == "__main__":
    # 测试分析器
    print("Factor Analyzer Module Loaded Successfully")
    print("Available classes:")
    print("  - FactorPreprocessor: 数据预处理")
    print("  - CorrelationAnalyzer: 相关性分析")
    print("  - CausalityAnalyzer: 因果关系分析")
    print("  - FactorEvaluator: 因子评估")
    print("  - FactorVisualizer: 可视化")
