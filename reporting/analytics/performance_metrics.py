"""
Comprehensive Performance Metrics Calculator
Computes all key metrics for hedge fund-grade performance reports
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

class PerformanceMetrics:
    """Comprehensive performance metrics calculator for quantitative strategies"""
    
    def __init__(self, strategy_returns: pd.Series, benchmark_returns: pd.Series, 
                 risk_free_rate: float = 0.02, trading_days: int = 252):
        """
        Initialize performance metrics calculator
        
        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Benchmark returns series
            risk_free_rate: Annual risk-free rate (default 2%)
            trading_days: Number of trading days per year (default 252)
        """
        self.strategy_returns = strategy_returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna()
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        
        # Align returns
        self.strategy_returns, self.benchmark_returns = self._align_returns()
        
        # Calculate cumulative returns
        self.strategy_cumulative = (1 + self.strategy_returns).cumprod()
        self.benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
        
        # Calculate excess returns
        self.excess_returns = self.strategy_returns - self.benchmark_returns
        
    def _align_returns(self) -> Tuple[pd.Series, pd.Series]:
        """Align strategy and benchmark returns by date"""
        aligned = pd.concat([self.strategy_returns, self.benchmark_returns], axis=1)
        aligned.columns = ['strategy', 'benchmark']
        aligned = aligned.dropna()
        return aligned['strategy'], aligned['benchmark']
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate all performance metrics"""
        return {
            'return_metrics': self.calculate_return_metrics(),
            'risk_metrics': self.calculate_risk_metrics(),
            'risk_adjusted_metrics': self.calculate_risk_adjusted_metrics(),
            'drawdown_metrics': self.calculate_drawdown_metrics(),
            'statistical_metrics': self.calculate_statistical_metrics(),
            'benchmark_metrics': self.calculate_benchmark_metrics(),
            'rolling_metrics': self.calculate_rolling_metrics()
        }
    
    def calculate_return_metrics(self) -> Dict:
        """Calculate return-based metrics"""
        # Check if we have data
        if len(self.strategy_returns) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'monthly_returns': pd.Series(dtype=float),
                'positive_days': 0,
                'negative_days': 0,
                'positive_months': 0,
                'negative_months': 0,
                'best_month': 0.0,
                'worst_month': 0.0,
                'avg_positive_month': 0.0,
                'avg_negative_month': 0.0
            }
        
        # FIXED: Ensure total return calculation is consistent
        # Use cumulative return method: (final_value / initial_value) - 1
        total_return = self.strategy_cumulative.iloc[-1] - 1
        
        # FIXED: Ensure annualized return calculation is consistent
        # Use proper annualization formula
        if len(self.strategy_returns) > 0:
            annualized_return = (1 + total_return) ** (self.trading_days / len(self.strategy_returns)) - 1
        else:
            annualized_return = 0.0
        
        # Daily hit rate calculation
        positive_days = (self.strategy_returns > 0).sum()
        negative_days = (self.strategy_returns < 0).sum()
        
        # Monthly returns for analysis
        monthly_returns = self.strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'monthly_returns': monthly_returns,
            'positive_days': positive_days,
            'negative_days': negative_days,
            'positive_months': (monthly_returns > 0).sum(),
            'negative_months': (monthly_returns < 0).sum(),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'avg_positive_month': monthly_returns[monthly_returns > 0].mean(),
            'avg_negative_month': monthly_returns[monthly_returns < 0].mean()
        }
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate risk-based metrics"""
        # Check if we have data
        if len(self.strategy_returns) == 0:
            return {
                'volatility': 0.0,
                'downside_deviation': 0.0,
                'var_95': 0.0,
                'var_99': 0.0,
                'cvar_95': 0.0,
                'cvar_99': 0.0,
                'semi_deviation': 0.0
            }
        
        volatility = self.strategy_returns.std() * np.sqrt(self.trading_days)
        downside_returns = self.strategy_returns[self.strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.trading_days) if len(downside_returns) > 0 else 0.0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(self.strategy_returns, 5)
        var_99 = np.percentile(self.strategy_returns, 1)
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = self.strategy_returns[self.strategy_returns <= var_95].mean()
        cvar_99 = self.strategy_returns[self.strategy_returns <= var_99].mean()
        
        return {
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'semi_deviation': downside_deviation
        }
    
    def calculate_risk_adjusted_metrics(self) -> Dict:
        """Calculate risk-adjusted return metrics"""
        # FIXED: Standardize Sharpe ratio calculation
        # Calculate daily excess returns properly
        daily_rf_rate = self.risk_free_rate / self.trading_days
        excess_returns = self.strategy_returns - daily_rf_rate
        
        # Calculate annualized excess return
        annualized_excess_return = excess_returns.mean() * self.trading_days
        
        # Calculate annualized volatility
        volatility = self.strategy_returns.std() * np.sqrt(self.trading_days)
        
        # FIXED: Sharpe Ratio calculation
        sharpe_ratio = annualized_excess_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio (using downside deviation)
        downside_returns = self.strategy_returns[self.strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(self.trading_days) if len(downside_returns) > 0 else volatility
        sortino_ratio = annualized_excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar Ratio
        max_drawdown = self.calculate_max_drawdown()
        calmar_ratio = annualized_excess_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Information Ratio
        tracking_error = self.excess_returns.std() * np.sqrt(self.trading_days)
        information_ratio = self.excess_returns.mean() * self.trading_days / tracking_error if tracking_error > 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        }
    
    def calculate_drawdown_metrics(self) -> Dict:
        """Calculate drawdown-related metrics"""
        # Check if we have data
        if len(self.strategy_cumulative) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'max_drawdown_start': pd.Timestamp.now(),
                'max_drawdown_end': pd.Timestamp.now(),
                'current_drawdown': 0.0,
                'avg_drawdown': 0.0,
                'drawdown_volatility': 0.0,
                'drawdown_series': pd.Series(dtype=float)
            }
        
        max_dd, max_dd_duration, max_dd_start, max_dd_end = self._calculate_drawdown_series()
        
        # Current drawdown
        try:
            current_dd = (self.strategy_cumulative.iloc[-1] / self.strategy_cumulative.max() - 1)
        except (IndexError, ValueError):
            current_dd = 0.0
        
        # Drawdown statistics
        dd_series = self._get_drawdown_series()
        avg_drawdown = dd_series[dd_series < 0].mean() if len(dd_series[dd_series < 0]) > 0 else 0.0
        drawdown_volatility = dd_series[dd_series < 0].std() if len(dd_series[dd_series < 0]) > 0 else 0.0
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_duration': max_dd_duration,
            'max_drawdown_start': max_dd_start,
            'max_drawdown_end': max_dd_end,
            'current_drawdown': current_dd,
            'avg_drawdown': avg_drawdown,
            'drawdown_volatility': drawdown_volatility,
            'drawdown_series': dd_series
        }
    
    def _calculate_drawdown_series(self) -> Tuple[float, int, pd.Timestamp, pd.Timestamp]:
        """Calculate maximum drawdown and its characteristics"""
        # Check if we have data
        if len(self.strategy_cumulative) == 0:
            return 0.0, 0, pd.Timestamp.now(), pd.Timestamp.now()
        
        peak = self.strategy_cumulative.expanding().max()
        drawdown = (self.strategy_cumulative - peak) / peak
        
        # Check if drawdown series is empty or all NaN
        if drawdown.empty or drawdown.isna().all():
            return 0.0, 0, pd.Timestamp.now(), pd.Timestamp.now()
        
        max_dd = drawdown.min()
        
        # Check if we can find the index of minimum
        try:
            max_dd_idx = drawdown.idxmin()
            max_dd_start_idx = peak[:max_dd_idx].idxmax()
            max_dd_duration = (max_dd_idx - max_dd_start_idx).days
        except (ValueError, TypeError):
            # Fallback if idxmin fails
            max_dd_idx = pd.Timestamp.now()
            max_dd_start_idx = pd.Timestamp.now()
            max_dd_duration = 0
        
        return max_dd, max_dd_duration, max_dd_start_idx, max_dd_idx
    
    def _get_drawdown_series(self) -> pd.Series:
        """Get the complete drawdown series"""
        peak = self.strategy_cumulative.expanding().max()
        return (self.strategy_cumulative - peak) / peak
    
    def calculate_statistical_metrics(self) -> Dict:
        """Calculate statistical distribution metrics"""
        returns = self.strategy_returns
        
        # Check if we have data
        if len(returns) == 0:
            return {
                'skewness': 0.0,
                'kurtosis': 0.0,
                'jarque_bera_stat': 0.0,
                'jarque_bera_pvalue': 1.0,
                'autocorrelation_lag1': 0.0,
                'is_normal': True
            }
        
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Jarque-Bera test for normality
        from scipy import stats
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Autocorrelation (simplified calculation)
        if len(returns) > 1:
            autocorr_lag1 = returns.autocorr(lag=1)
        else:
            autocorr_lag1 = 0.0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'autocorrelation_lag1': autocorr_lag1,
            'is_normal': jb_pvalue > 0.05
        }
    
    def calculate_benchmark_metrics(self) -> Dict:
        """Calculate benchmark-specific metrics for comparison"""
        # Calculate basic benchmark metrics
        benchmark_return = (self.benchmark_cumulative.iloc[-1] - 1) if len(self.benchmark_cumulative) > 0 else 0
        benchmark_annualized_return = (1 + benchmark_return) ** (self.trading_days / len(self.benchmark_returns)) - 1 if len(self.benchmark_returns) > 0 else 0
        
        # Calculate benchmark volatility
        benchmark_volatility = self.benchmark_returns.std() * np.sqrt(self.trading_days) if len(self.benchmark_returns) > 0 else 0
        
        # Calculate benchmark drawdown
        benchmark_peak = self.benchmark_cumulative.expanding().max()
        benchmark_drawdown = (self.benchmark_cumulative - benchmark_peak) / benchmark_peak
        benchmark_max_drawdown = benchmark_drawdown.min() if len(benchmark_drawdown) > 0 else 0
        
        # Calculate benchmark Sharpe ratio
        benchmark_excess_returns = self.benchmark_returns - (self.risk_free_rate / self.trading_days)
        benchmark_sharpe_ratio = (benchmark_excess_returns.mean() * self.trading_days) / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # Calculate benchmark Sortino ratio
        benchmark_downside_returns = benchmark_excess_returns[benchmark_excess_returns < 0]
        benchmark_downside_deviation = benchmark_downside_returns.std() * np.sqrt(self.trading_days) if len(benchmark_downside_returns) > 0 else benchmark_volatility
        benchmark_sortino_ratio = (benchmark_excess_returns.mean() * self.trading_days) / benchmark_downside_deviation if benchmark_downside_deviation > 0 else 0
        
        # Calculate benchmark Calmar ratio
        benchmark_calmar_ratio = benchmark_annualized_return / abs(benchmark_max_drawdown) if benchmark_max_drawdown != 0 else 0
        
        # Calculate benchmark VaR
        benchmark_var_95 = np.percentile(self.benchmark_returns, 5) if len(self.benchmark_returns) > 0 else 0
        benchmark_var_99 = np.percentile(self.benchmark_returns, 1) if len(self.benchmark_returns) > 0 else 0
        
        # Calculate benchmark tail risk (CVaR)
        benchmark_tail_returns = self.benchmark_returns[self.benchmark_returns <= benchmark_var_95]
        benchmark_tail_risk = benchmark_tail_returns.mean() if len(benchmark_tail_returns) > 0 else 0
        
        # Calculate benchmark hit ratio (positive days)
        benchmark_positive_days = (self.benchmark_returns > 0).sum()
        benchmark_total_days = len(self.benchmark_returns)
        benchmark_hit_ratio = benchmark_positive_days / benchmark_total_days if benchmark_total_days > 0 else 0
        
        # Also calculate monthly hit ratio for reference
        try:
            benchmark_monthly_returns = self.benchmark_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            benchmark_positive_months = (benchmark_monthly_returns > 0).sum()
            benchmark_total_months = len(benchmark_monthly_returns)
            benchmark_monthly_hit_ratio = benchmark_positive_months / benchmark_total_months if benchmark_total_months > 0 else 0
        except (TypeError, ValueError):
            # Fallback if resampling fails
            benchmark_monthly_hit_ratio = 0
        
        # Strategy metrics for comparison
        strategy_return = (self.strategy_cumulative.iloc[-1] - 1) if len(self.strategy_cumulative) > 0 else 0
        strategy_annualized_return = (1 + strategy_return) ** (self.trading_days / len(self.strategy_returns)) - 1 if len(self.strategy_returns) > 0 else 0
        
        # Calculate correlation and beta
        correlation = self.strategy_returns.corr(self.benchmark_returns) if len(self.strategy_returns) > 0 and len(self.benchmark_returns) > 0 else 0
        
        # Calculate beta
        covariance = np.cov(self.strategy_returns, self.benchmark_returns)[0, 1] if len(self.strategy_returns) > 1 and len(self.benchmark_returns) > 1 else 0
        benchmark_variance = self.benchmark_returns.var() if len(self.benchmark_returns) > 1 else 1
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Calculate alpha
        alpha = strategy_annualized_return - (self.risk_free_rate + beta * (benchmark_annualized_return - self.risk_free_rate))
        
        # R-squared
        r_squared = correlation ** 2
        
        return {
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'r_squared': r_squared,
            'benchmark_return': benchmark_return,
            'benchmark_annualized_return': benchmark_annualized_return,
            'benchmark_volatility': benchmark_volatility,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'benchmark_sharpe_ratio': benchmark_sharpe_ratio,
            'benchmark_sortino_ratio': benchmark_sortino_ratio,
            'benchmark_calmar_ratio': benchmark_calmar_ratio,
            'benchmark_var_95': benchmark_var_95,
            'benchmark_var_99': benchmark_var_99,
            'benchmark_tail_risk': benchmark_tail_risk,
            'benchmark_hit_ratio': benchmark_hit_ratio,
            'excess_return': strategy_return - benchmark_return
        }
    
    def calculate_rolling_metrics(self, window: int = 60) -> Dict:
        """Calculate rolling metrics"""
        # Use a smaller window for shorter datasets
        if len(self.strategy_returns) < window:
            window = max(20, len(self.strategy_returns) // 3)  # Use 1/3 of data or minimum 20 days
        
        # Calculate rolling Sharpe with proper excess returns
        excess_returns = self.strategy_returns - (self.risk_free_rate / self.trading_days)
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = excess_returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(self.trading_days)
        
        rolling_vol = self.strategy_returns.rolling(window).std() * np.sqrt(self.trading_days)
        rolling_beta = self._calculate_rolling_beta(window)
        
        return {
            'rolling_sharpe': rolling_sharpe,
            'rolling_volatility': rolling_vol,
            'rolling_beta': rolling_beta
        }

    def calculate_rolling_drawdown(self, target_window: int = 252) -> Dict:
        """Calculate rolling drawdown with adaptive window sizing"""
        # Determine appropriate window size
        data_length = len(self.strategy_returns)
        
        if data_length >= target_window:
            # Full 12-month window available
            window = target_window
            window_type = "12-month"
        elif data_length >= 126:  # At least 6 months
            # Use 6-month window
            window = 126
            window_type = "6-month"
        elif data_length >= 63:  # At least 3 months
            # Use 3-month window
            window = 63
            window_type = "3-month"
        elif data_length >= 42:  # At least 2 months
            # Use 2-month window
            window = 42
            window_type = "2-month"
        elif data_length >= 21:  # At least 1 month
            # Use 1-month window
            window = 21
            window_type = "1-month"
        else:
            # Use minimum viable window
            window = max(10, data_length // 2)
            window_type = f"{window}-day"
        
        # Calculate rolling drawdown
        rolling_drawdown = pd.Series(index=self.strategy_returns.index, dtype=float)
        
        for i in range(window, len(self.strategy_returns)):
            # Get window of cumulative returns
            window_returns = self.strategy_returns.iloc[i-window:i]
            window_cumulative = (1 + window_returns).cumprod()
            
            # Calculate drawdown within this window
            window_peak = window_cumulative.max()
            window_drawdown = (window_cumulative.iloc[-1] - window_peak) / window_peak
            rolling_drawdown.iloc[i] = window_drawdown
        
        return {
            'rolling_drawdown': rolling_drawdown,
            'window_size': window,
            'window_type': window_type,
            'data_length': data_length,
            'target_window': target_window
        }
    
    def _calculate_rolling_beta(self, window: int) -> pd.Series:
        """Calculate rolling beta"""
        try:
            # Create a simple rolling beta calculation
            rolling_beta = pd.Series(index=self.strategy_returns.index, dtype=float)
            
            for i in range(window, len(self.strategy_returns)):
                strategy_window = self.strategy_returns.iloc[i-window:i]
                benchmark_window = self.benchmark_returns.iloc[i-window:i]
                
                if len(strategy_window) > 1 and len(benchmark_window) > 1:
                    covariance = np.cov(strategy_window, benchmark_window)[0, 1]
                    benchmark_variance = benchmark_window.var()
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else np.nan
                    rolling_beta.iloc[i] = beta
                else:
                    rolling_beta.iloc[i] = np.nan
            
            return rolling_beta
        except Exception as e:
            print(f"Warning: Rolling beta calculation failed: {e}")
            return pd.Series(index=self.strategy_returns.index, dtype=float)
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        peak = self.strategy_cumulative.expanding().max()
        drawdown = (self.strategy_cumulative - peak) / peak
        return drawdown.min()
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of key performance metrics"""
        metrics = self.calculate_all_metrics()
        
        return {
            'total_return': metrics['return_metrics']['total_return'],
            'annualized_return': metrics['return_metrics']['annualized_return'],
            'volatility': metrics['risk_metrics']['volatility'],
            'sharpe_ratio': metrics['risk_adjusted_metrics']['sharpe_ratio'],
            'max_drawdown': metrics['drawdown_metrics']['max_drawdown'],
            'calmar_ratio': metrics['risk_adjusted_metrics']['calmar_ratio'],
            'information_ratio': metrics['risk_adjusted_metrics']['information_ratio'],
            'beta': metrics['benchmark_metrics']['beta'],
            'alpha': metrics['benchmark_metrics']['alpha'],
            'win_rate': metrics['return_metrics']['positive_days'] / (metrics['return_metrics']['positive_days'] + metrics['return_metrics']['negative_days']) if (metrics['return_metrics']['positive_days'] + metrics['return_metrics']['negative_days']) > 0 else 0
        } 