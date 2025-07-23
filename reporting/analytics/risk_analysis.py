"""
Advanced Risk Analysis Module
Provides comprehensive risk metrics and stress testing for quantitative strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class RiskAnalysis:
    """Advanced risk analysis for quantitative strategies"""
    
    def __init__(self, strategy_returns: pd.Series, benchmark_returns: pd.Series,
                 confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize risk analysis
        
        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Benchmark returns series
            confidence_levels: List of confidence levels for VaR/CVaR
        """
        self.strategy_returns = strategy_returns.dropna()
        self.benchmark_returns = benchmark_returns.dropna()
        self.confidence_levels = confidence_levels
        
        # Align returns
        self.strategy_returns, self.benchmark_returns = self._align_returns()
        
    def _align_returns(self) -> Tuple[pd.Series, pd.Series]:
        """Align strategy and benchmark returns"""
        aligned = pd.concat([self.strategy_returns, self.benchmark_returns], axis=1)
        aligned.columns = ['strategy', 'benchmark']
        aligned = aligned.dropna()
        return aligned['strategy'], aligned['benchmark']
    
    def calculate_all_risk_metrics(self) -> Dict:
        """Calculate all risk metrics"""
        return {
            'var_analysis': self.calculate_var_metrics(),
            'cvar_analysis': self.calculate_cvar_metrics(),
            'tail_risk': self.calculate_tail_risk_metrics(),
            'stress_tests': self.run_stress_tests(),
            'correlation_analysis': self.calculate_correlation_metrics(),
            'volatility_analysis': self.calculate_volatility_metrics(),
            'extreme_events': self.analyze_extreme_events()
        }
    
    def calculate_var_metrics(self) -> Dict:
        """Calculate Value at Risk metrics"""
        var_metrics = {}
        
        # Check if we have any returns data
        if len(self.strategy_returns) == 0:
            # Return default values for empty returns
            for confidence in self.confidence_levels:
                var_metrics[f'var_{int(confidence*100)}'] = {
                    'historical': 0.0,
                    'parametric': 0.0,
                    'monte_carlo': 0.0
                }
            return var_metrics
        
        for confidence in self.confidence_levels:
            try:
                # Historical VaR
                var_historical = np.percentile(self.strategy_returns, (1 - confidence) * 100)
                
                # Parametric VaR (assuming normal distribution)
                mean_return = self.strategy_returns.mean()
                std_return = self.strategy_returns.std()
                var_parametric = mean_return + stats.norm.ppf(1 - confidence) * std_return
                
                # Monte Carlo VaR (simplified)
                n_simulations = 10000
                simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                var_monte_carlo = np.percentile(simulated_returns, (1 - confidence) * 100)
                
                var_metrics[f'var_{int(confidence*100)}'] = {
                    'historical': var_historical,
                    'parametric': var_parametric,
                    'monte_carlo': var_monte_carlo
                }
            except Exception as e:
                # Fallback to default values if calculation fails
                var_metrics[f'var_{int(confidence*100)}'] = {
                    'historical': 0.0,
                    'parametric': 0.0,
                    'monte_carlo': 0.0
                }
        
        return var_metrics
    
    def calculate_cvar_metrics(self) -> Dict:
        """Calculate Conditional Value at Risk (Expected Shortfall) metrics"""
        cvar_metrics = {}
        
        # Check if we have any returns data
        if len(self.strategy_returns) == 0:
            # Return default values for empty returns
            for confidence in self.confidence_levels:
                cvar_metrics[f'cvar_{int(confidence*100)}'] = {
                    'cvar': 0.0,
                    'var_threshold': 0.0,
                    'tail_probability': 0.0
                }
            return cvar_metrics
        
        for confidence in self.confidence_levels:
            try:
                var_threshold = np.percentile(self.strategy_returns, (1 - confidence) * 100)
                tail_returns = self.strategy_returns[self.strategy_returns <= var_threshold]
                cvar = tail_returns.mean() if len(tail_returns) > 0 else 0.0
                
                cvar_metrics[f'cvar_{int(confidence*100)}'] = {
                    'cvar': cvar,
                    'var_threshold': var_threshold,
                    'tail_probability': len(tail_returns) / len(self.strategy_returns)
                }
            except Exception as e:
                # Fallback to default values if calculation fails
                cvar_metrics[f'cvar_{int(confidence*100)}'] = {
                    'cvar': 0.0,
                    'var_threshold': 0.0,
                    'tail_probability': 0.0
                }
        
        return cvar_metrics
    
    def calculate_tail_risk_metrics(self) -> Dict:
        """Calculate tail risk and distribution metrics"""
        returns = self.strategy_returns
        
        # Tail risk metrics
        left_tail_5 = np.percentile(returns, 5)
        left_tail_1 = np.percentile(returns, 1)
        right_tail_95 = np.percentile(returns, 95)
        right_tail_99 = np.percentile(returns, 99)
        
        # Tail dependence
        left_tail_events = returns <= left_tail_5
        right_tail_events = returns >= right_tail_95
        
        # Expected tail loss
        expected_tail_loss = returns[returns <= left_tail_5].mean()
        expected_tail_gain = returns[returns >= right_tail_95].mean()
        
        return {
            'left_tail_5': left_tail_5,
            'left_tail_1': left_tail_1,
            'right_tail_95': right_tail_95,
            'right_tail_99': right_tail_99,
            'expected_tail_loss': expected_tail_loss,
            'expected_tail_gain': expected_tail_gain,
            'tail_asymmetry': abs(expected_tail_loss) - expected_tail_gain,
            'left_tail_frequency': left_tail_events.sum() / len(returns),
            'right_tail_frequency': right_tail_events.sum() / len(returns)
        }
    
    def run_stress_tests(self) -> Dict:
        """Run various stress tests"""
        stress_tests = {}
        
        # Market crash scenario (simulate 2008-like event)
        crash_returns = self.strategy_returns * 2  # Double the volatility
        stress_tests['market_crash'] = {
            'var_95': np.percentile(crash_returns, 5),
            'max_loss': crash_returns.min(),
            'expected_loss': crash_returns[crash_returns < 0].mean()
        }
        
        # High volatility scenario
        high_vol_returns = self.strategy_returns * 1.5
        stress_tests['high_volatility'] = {
            'var_95': np.percentile(high_vol_returns, 5),
            'volatility': high_vol_returns.std() * np.sqrt(252),
            'sharpe_ratio': high_vol_returns.mean() / high_vol_returns.std() * np.sqrt(252)
        }
        
        # Correlation breakdown scenario
        # Simulate scenario where correlation with benchmark increases
        correlation_breakdown = self.strategy_returns * 0.5 + self.benchmark_returns * 0.5
        stress_tests['correlation_breakdown'] = {
            'correlation': correlation_breakdown.corr(self.benchmark_returns),
            'var_95': np.percentile(correlation_breakdown, 5),
            'beta': np.cov(correlation_breakdown, self.benchmark_returns)[0, 1] / self.benchmark_returns.var()
        }
        
        return stress_tests
    
    def calculate_correlation_metrics(self) -> Dict:
        """Calculate correlation and dependency metrics"""
        # Rolling correlation
        rolling_corr = self.strategy_returns.rolling(60).corr(self.benchmark_returns)
        
        # Correlation in different market regimes
        market_vol = self.benchmark_returns.rolling(60).std()
        high_vol_periods = market_vol > market_vol.quantile(0.75)
        low_vol_periods = market_vol < market_vol.quantile(0.25)
        
        high_vol_corr = self.strategy_returns[high_vol_periods].corr(self.benchmark_returns[high_vol_periods])
        low_vol_corr = self.strategy_returns[low_vol_periods].corr(self.benchmark_returns[low_vol_periods])
        
        # Tail correlation
        strategy_tail = self.strategy_returns <= np.percentile(self.strategy_returns, 10)
        benchmark_tail = self.benchmark_returns <= np.percentile(self.benchmark_returns, 10)
        tail_correlation = self.strategy_returns[strategy_tail & benchmark_tail].corr(
            self.benchmark_returns[strategy_tail & benchmark_tail]
        )
        
        return {
            'overall_correlation': self.strategy_returns.corr(self.benchmark_returns),
            'rolling_correlation': rolling_corr,
            'high_vol_correlation': high_vol_corr,
            'low_vol_correlation': low_vol_corr,
            'tail_correlation': tail_correlation,
            'correlation_stability': rolling_corr.std()
        }
    
    def calculate_volatility_metrics(self) -> Dict:
        """Calculate volatility-related metrics"""
        # Rolling volatility
        rolling_vol = self.strategy_returns.rolling(60).std() * np.sqrt(252)
        
        # Volatility clustering
        vol_autocorr = self.strategy_returns.rolling(60).std().autocorr()
        
        # Volatility regimes - align with strategy returns
        vol_regimes = pd.qcut(rolling_vol.dropna(), 3, labels=['Low', 'Medium', 'High'])
        
        # Align vol_regimes with strategy_returns index
        aligned_regimes = vol_regimes.reindex(self.strategy_returns.index, method='ffill')
        
        # Returns in different volatility regimes
        regime_returns = {}
        for regime in ['Low', 'Medium', 'High']:
            regime_mask = aligned_regimes == regime
            regime_returns[regime] = {
                'mean_return': self.strategy_returns[regime_mask].mean() * 252,
                'volatility': self.strategy_returns[regime_mask].std() * np.sqrt(252),
                'sharpe': (self.strategy_returns[regime_mask].mean() * 252) / 
                         (self.strategy_returns[regime_mask].std() * np.sqrt(252))
            }
        
        return {
            'current_volatility': rolling_vol.iloc[-1],
            'avg_volatility': rolling_vol.mean(),
            'volatility_of_volatility': rolling_vol.std(),
            'vol_autocorrelation': vol_autocorr,
            'regime_returns': regime_returns,
            'volatility_skew': rolling_vol.skew()
        }
    
    def analyze_extreme_events(self) -> Dict:
        """Analyze extreme events and their characteristics"""
        returns = self.strategy_returns
        
        # Define extreme events (beyond 2 standard deviations)
        mean_return = returns.mean()
        std_return = returns.std()
        extreme_threshold = 2 * std_return
        
        extreme_negative = returns <= (mean_return - extreme_threshold)
        extreme_positive = returns >= (mean_return + extreme_threshold)
        
        # Analyze extreme negative events
        extreme_neg_events = returns[extreme_negative]
        extreme_pos_events = returns[extreme_positive]
        
        # Clustering of extreme events
        extreme_dates = returns[extreme_negative | extreme_positive].index
        if len(extreme_dates) > 1:
            time_between_extremes = extreme_dates.to_series().diff().dropna()
            avg_time_between_extremes = time_between_extremes.mean()
        else:
            avg_time_between_extremes = pd.Timedelta(days=0)
        
        return {
            'extreme_negative_count': extreme_negative.sum(),
            'extreme_positive_count': extreme_positive.sum(),
            'extreme_negative_mean': extreme_neg_events.mean(),
            'extreme_positive_mean': extreme_pos_events.mean(),
            'extreme_negative_std': extreme_neg_events.std(),
            'extreme_positive_std': extreme_pos_events.std(),
            'avg_time_between_extremes': avg_time_between_extremes,
            'extreme_event_frequency': (extreme_negative.sum() + extreme_positive.sum()) / len(returns)
        }
    
    def get_risk_summary(self) -> Dict:
        """Get a summary of key risk metrics"""
        all_metrics = self.calculate_all_risk_metrics()
        
        return {
            'var_95': all_metrics['var_analysis']['var_95']['historical'],
            'cvar_95': all_metrics['cvar_analysis']['cvar_95']['cvar'],
            'max_loss': all_metrics['tail_risk']['left_tail_1'],
            'tail_asymmetry': all_metrics['tail_risk']['tail_asymmetry'],
            'correlation': all_metrics['correlation_analysis']['overall_correlation'],
            'volatility_regime_stability': all_metrics['volatility_analysis']['volatility_of_volatility'],
            'extreme_event_frequency': all_metrics['extreme_events']['extreme_event_frequency']
        } 