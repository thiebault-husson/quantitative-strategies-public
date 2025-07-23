"""
Main Analytics Engine
Coordinates all analytics modules to generate comprehensive hedge fund reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .performance_metrics import PerformanceMetrics
from .risk_analysis import RiskAnalysis
from .trade_analytics import TradeAnalytics

class AnalyticsEngine:
    """Main analytics engine for comprehensive strategy analysis"""
    
    def __init__(self, backtest_results: Dict, strategy_name: str = "Strategy", user_start_date: str = None, user_end_date: str = None):
        """
        Initialize analytics engine
        
        Args:
            backtest_results: Results from backtest engine
            strategy_name: Name of the strategy
            user_start_date: User-specified start date for the report period
            user_end_date: User-specified end date for the report period
        """
        self.backtest_results = backtest_results
        self.strategy_name = strategy_name
        self.user_start_date = user_start_date
        self.user_end_date = user_end_date
        
        # Extract key data
        self.strategy_returns = self._extract_strategy_returns()
        self.benchmark_returns = self._extract_benchmark_returns()
        self.portfolio_values = self._extract_portfolio_values()
        
        # Initialize analytics modules
        self.performance_metrics = PerformanceMetrics(
            self.strategy_returns, 
            self.benchmark_returns
        )
        
        self.risk_analysis = RiskAnalysis(
            self.strategy_returns, 
            self.benchmark_returns
        )
        
        self.trade_analytics = TradeAnalytics(
            backtest_results, 
            self.strategy_returns
        )
        
        # Ensure data consistency with audit
        self._ensure_data_consistency_with_audit()
        
        # Validate data periods
        self._validate_data_periods()
        
    
    def _validate_data_periods(self):
        """Validate that all data components use the same period"""
        if self.user_start_date and self.user_end_date:
            expected_start = pd.to_datetime(self.user_start_date)
            expected_end = pd.to_datetime(self.user_end_date)
            
            # Validate strategy returns period
            if len(self.strategy_returns) > 0:
                actual_start = self.strategy_returns.index[0]
                actual_end = self.strategy_returns.index[-1]
                
                if actual_start != expected_start or actual_end != expected_end:
                    print(f"⚠️  WARNING: Strategy returns period mismatch")
                    print(f"   Expected: {expected_start.strftime('%Y-%m-%d')} to {expected_end.strftime('%Y-%m-%d')}")
                    print(f"   Actual: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
                    
                    # Trim to expected period
                    self.strategy_returns = self.strategy_returns[
                        (self.strategy_returns.index >= expected_start) & 
                        (self.strategy_returns.index <= expected_end)
                    ]
            
            # Validate benchmark returns period
            if len(self.benchmark_returns) > 0:
                actual_start = self.benchmark_returns.index[0]
                actual_end = self.benchmark_returns.index[-1]
                
                if actual_start != expected_start or actual_end != expected_end:
                    print(f"⚠️  WARNING: Benchmark returns period mismatch")
                    print(f"   Expected: {expected_start.strftime('%Y-%m-%d')} to {expected_end.strftime('%Y-%m-%d')}")
                    print(f"   Actual: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
                    
                    # Trim to expected period
                    self.benchmark_returns = self.benchmark_returns[
                        (self.benchmark_returns.index >= expected_start) & 
                        (self.benchmark_returns.index <= expected_end)
                    ]

    
    def _ensure_data_consistency_with_audit(self):
        """Ensure data consistency - TRUST THE ALIGNED DATA"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info("🔧 Ensuring data consistency with aligned data")
        
        # Force fresh data extraction
        self.strategy_returns = self._extract_strategy_returns()
        self.benchmark_returns = self._extract_benchmark_returns()
        self.portfolio_values = self._extract_portfolio_values()
        
        # Log the data we received
        logger.info(f"📊 Received data periods:")
        if len(self.strategy_returns) > 0:
            logger.info(f"   Strategy returns: {self.strategy_returns.index[0].strftime('%Y-%m-%d')} to {self.strategy_returns.index[-1].strftime('%Y-%m-%d')}")
        if len(self.benchmark_returns) > 0:
            logger.info(f"   Benchmark returns: {self.benchmark_returns.index[0].strftime('%Y-%m-%d')} to {self.benchmark_returns.index[-1].strftime('%Y-%m-%d')}")
        if len(self.portfolio_values) > 0:
            logger.info(f"   Portfolio values: {self.portfolio_values.index[0].strftime('%Y-%m-%d')} to {self.portfolio_values.index[-1].strftime('%Y-%m-%d')}")
        
        # Validate that data is consistent (same periods)
        self._validate_data_consistency()
        
        # Reinitialize analytics modules with consistent data
        self.performance_metrics = PerformanceMetrics(
            self.strategy_returns, 
            self.benchmark_returns
        )
        
        self.risk_analysis = RiskAnalysis(
            self.strategy_returns, 
            self.benchmark_returns
        )
        
        self.trade_analytics = TradeAnalytics(
            self.backtest_results, 
            self.strategy_returns
        )
        
        logger.info(f"✅ Data consistency ensured:")
        logger.info(f"   Strategy returns: {len(self.strategy_returns)} days")
        logger.info(f"   Benchmark returns: {len(self.benchmark_returns)} days")
        logger.info(f"   Portfolio values: {len(self.portfolio_values)} days")

    def _validate_data_consistency(self):
        """Validate that all data components have consistent periods"""
        import logging
        logger = logging.getLogger(__name__)
        
        components = []
        
        if len(self.strategy_returns) > 0:
            components.append(('strategy_returns', self.strategy_returns.index[0], self.strategy_returns.index[-1]))
        if len(self.benchmark_returns) > 0:
            components.append(('benchmark_returns', self.benchmark_returns.index[0], self.benchmark_returns.index[-1]))
        if len(self.portfolio_values) > 0:
            components.append(('portfolio_values', self.portfolio_values.index[0], self.portfolio_values.index[-1]))
        
        if len(components) > 1:
            # Check if all components have the same period
            first_start = components[0][1]
            first_end = components[0][2]
            
            for name, start, end in components[1:]:
                if start != first_start or end != first_end:
                    logger.error(f"❌ Data consistency validation failed!")
                    logger.error(f"   {components[0][0]}: {first_start.strftime('%Y-%m-%d')} to {first_end.strftime('%Y-%m-%d')}")
                    logger.error(f"   {name}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
                    raise ValueError(f"Inconsistent data periods between {components[0][0]} and {name}")
        
        logger.info("✅ Data consistency validation passed")

    def _extract_strategy_returns(self) -> pd.Series:
        """Extract strategy returns from backtest results"""
        if 'strategy_returns' in self.backtest_results:
            returns = pd.Series(self.backtest_results['strategy_returns'])
            return returns
        elif 'returns' in self.backtest_results:
            returns = pd.Series(self.backtest_results['returns'])
            return returns
        else:
            # Calculate from portfolio values
            portfolio_values = self._extract_portfolio_values()
            if len(portfolio_values) > 1:
                returns = portfolio_values.pct_change().dropna()
                return returns
            else:
                return pd.Series(dtype=float)
    
    def _extract_benchmark_returns(self) -> pd.Series:
        """Extract benchmark returns from backtest results"""
        if 'benchmark_returns' in self.backtest_results:
            return pd.Series(self.backtest_results['benchmark_returns'])
        elif 'benchmark' in self.backtest_results:
            benchmark_values = pd.Series(self.backtest_results['benchmark'])
            if len(benchmark_values) > 1:
                return benchmark_values.pct_change().dropna()
        return pd.Series(dtype=float)
    
    def _extract_portfolio_values(self) -> pd.Series:
        """Extract portfolio values from backtest results"""
        if 'portfolio_values' in self.backtest_results:
            return pd.Series(self.backtest_results['portfolio_values'])
        elif 'equity_curve' in self.backtest_results:
            return pd.Series(self.backtest_results['equity_curve'])
        else:
            # Calculate from returns
            returns = self._extract_strategy_returns()
            if len(returns) > 0:
                return (1 + returns).cumprod()
            return pd.Series(dtype=float)
    
    def generate_comprehensive_analysis(self) -> Dict:
        """Generate comprehensive analysis for hedge fund report"""
        print("🔍 Generating comprehensive analytics...")
        
        # Calculate all metrics
        performance_metrics = self.performance_metrics.calculate_all_metrics()
        risk_metrics = self.risk_analysis.calculate_all_risk_metrics()
        trade_metrics = self.trade_analytics.calculate_all_trade_metrics()
        period_metrics = self._calculate_period_metrics()
        
        # Generate charts data
        charts_data = self._generate_charts_data()
        
        # Create executive summary
        executive_summary = self._create_executive_summary()
        
        # Create strategy overview
        strategy_overview = self._create_strategy_overview()
        
        return {
            'executive_summary': executive_summary,
            'strategy_overview': strategy_overview,
            'performance_metrics': performance_metrics,
            'risk_metrics': risk_metrics,
            'trade_metrics': trade_metrics,
            'period_metrics': period_metrics,
            'charts_data': charts_data,
            'report_metadata': self._create_report_metadata()
        }
    
    def _create_executive_summary(self) -> Dict:
        """Create executive summary section"""
        perf_summary = self.performance_metrics.get_performance_summary()
        risk_summary = self.risk_analysis.get_risk_summary()
        trade_summary = self.trade_analytics.get_trade_summary()
        
        # Use actual trade win rate from trade analytics instead of daily returns
        win_rate = trade_summary.get('win_rate', 0)
        
        # Get total trades from trade analytics (which uses actual strategy data)
        total_trades = trade_summary.get('total_trades', 0)
        
        return {
            'strategy_name': self.strategy_name,
            'analysis_period': f"{self.user_start_date} to {self.user_end_date}" if self.user_start_date and self.user_end_date else f"{self.strategy_returns.index[0].strftime('%Y-%m-%d')} to {self.strategy_returns.index[-1].strftime('%Y-%m-%d')}",
            'total_return': perf_summary['total_return'],
            'annualized_return': perf_summary['annualized_return'],
            'volatility': perf_summary['volatility'],
            'sharpe_ratio': perf_summary['sharpe_ratio'],
            'max_drawdown': perf_summary['max_drawdown'],
            'information_ratio': perf_summary['information_ratio'],
            'beta': perf_summary['beta'],
            'alpha': perf_summary['alpha'],
            'var_95': risk_summary['var_95'],
            'win_rate': win_rate,
            'total_trades': total_trades,
            'key_insights': self._generate_key_insights(perf_summary, risk_summary, trade_summary)
        }
    
    def _create_strategy_overview(self) -> Dict:
        """Create strategy overview section"""
        # Calculate performance metrics for strategy description
        initial_capital = self.backtest_results.get('initial_capital', 100000)
        final_value = self.portfolio_values.iloc[-1] if len(self.portfolio_values) > 0 else initial_capital
        total_return = (final_value / initial_capital - 1) * 100
        
        return {
            'strategy_name': self.strategy_name,
            'strategy_type': self._determine_strategy_type(),
            'asset_class': self._determine_asset_class(),
            'universe': self._determine_universe(),
            'signal_frequency': self._determine_signal_frequency(),
            'holding_period': self._determine_holding_period(),
            'execution_assumptions': self._create_execution_assumptions(),
            'leverage_assumptions': self._determine_leverage_assumptions(),
            'backtest_parameters': self._extract_backtest_parameters(),
            'performance_summary': {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return
            }
        }
    
    def _generate_charts_data(self) -> Dict:
        """Generate data for all charts"""
        returns = self.strategy_returns
        benchmark_returns = self.benchmark_returns
        portfolio_values = self.portfolio_values
        
        # Equity curve data (Base 100)
        # Normalize both strategy and benchmark to start at 100
        strategy_base100 = (portfolio_values / portfolio_values.iloc[0] * 100).tolist()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_base100 = (benchmark_cumulative / benchmark_cumulative.iloc[0] * 100).tolist() if len(benchmark_returns) > 0 else []
        
        equity_curve_data = {
            'dates': [date.strftime('%Y-%m-%d') for date in portfolio_values.index],
            'strategy_values': strategy_base100,
            'benchmark_values': benchmark_base100
        }
        
        # Rolling metrics data
        rolling_metrics = self.performance_metrics.calculate_rolling_metrics()
        
        # Calculate rolling Sharpe for benchmark as well
        benchmark_rolling_sharpe = self._calculate_benchmark_rolling_sharpe()
        
        # Convert pandas Series to lists, handling nan values properly for JavaScript
        def convert_series_to_list(series):
            return [None if pd.isna(val) or np.isinf(val) else val for val in series.tolist()]
        
        rolling_data = {
            'dates': [date.strftime('%Y-%m-%d') for date in returns.index],
            'rolling_sharpe': convert_series_to_list(rolling_metrics['rolling_sharpe']),
            'rolling_volatility': convert_series_to_list(rolling_metrics['rolling_volatility']),
            'rolling_beta': convert_series_to_list(rolling_metrics['rolling_beta']),
            'benchmark_rolling_sharpe': convert_series_to_list(benchmark_rolling_sharpe)
        }
        
        # Drawdown data
        drawdown_metrics = self.performance_metrics.calculate_drawdown_metrics()
        
        # Calculate benchmark drawdown as well
        benchmark_drawdown_series = self._calculate_benchmark_drawdown_series()
        
        # Calculate rolling drawdown with adaptive window
        rolling_drawdown_data = self.performance_metrics.calculate_rolling_drawdown()
        
        drawdown_data = {
            'dates': [date.strftime('%Y-%m-%d') for date in returns.index],
            'drawdown_series': convert_series_to_list(drawdown_metrics['drawdown_series']),
            'benchmark_drawdown_series': convert_series_to_list(benchmark_drawdown_series),
            'rolling_drawdown_series': convert_series_to_list(rolling_drawdown_data['rolling_drawdown']),
            'window_type': rolling_drawdown_data['window_type'],
            'window_size': rolling_drawdown_data['window_size'],
            'data_length': rolling_drawdown_data['data_length']
        }
        
        # Monthly returns heatmap data
        monthly_returns = self.performance_metrics.calculate_return_metrics()['monthly_returns']
        heatmap_data = {
            'years': monthly_returns.index.year.unique().tolist(),
            'months': monthly_returns.index.month.unique().tolist(),
            'returns_matrix': self._create_returns_matrix(monthly_returns)
        }
        
        # Returns distribution data
        returns_dist_data = {
            'returns': convert_series_to_list(returns),
            'histogram_bins': np.histogram(returns.dropna(), bins=50)[0].tolist(),
            'histogram_edges': np.histogram(returns.dropna(), bins=50)[1].tolist()
        }
        
        return {
            'equity_curve': equity_curve_data,
            'rolling_metrics': rolling_data,
            'drawdown': drawdown_data,
            'monthly_heatmap': heatmap_data,
            'returns_distribution': returns_dist_data
        }
    
    def _create_returns_matrix(self, monthly_returns: pd.Series) -> List[List[float]]:
        """Create returns matrix for heatmap"""
        returns_df = monthly_returns.to_frame('returns')
        returns_df['year'] = returns_df.index.year
        returns_df['month'] = returns_df.index.month
        
        pivot_table = returns_df.pivot_table(
            values='returns', 
            index='year', 
            columns='month', 
            fill_value=0
        )
        
        return pivot_table.values.tolist()
    
    def _generate_key_insights(self, perf_summary: Dict, risk_summary: Dict, trade_summary: Dict) -> List[str]:
        """Generate key insights for executive summary"""
        insights = []
        
        # Performance insights
        if perf_summary['sharpe_ratio'] > 1.0:
            insights.append("Strong risk-adjusted returns with Sharpe ratio above 1.0")
        elif perf_summary['sharpe_ratio'] > 0.5:
            insights.append("Moderate risk-adjusted returns with positive Sharpe ratio")
        else:
            insights.append("Risk-adjusted returns below target levels")
        
        # Risk insights
        if abs(perf_summary['max_drawdown']) < 0.1:
            insights.append("Excellent drawdown control with maximum drawdown under 10%")
        elif abs(perf_summary['max_drawdown']) < 0.2:
            insights.append("Reasonable drawdown control with maximum drawdown under 20%")
        else:
            insights.append("High drawdown risk requiring attention")
        
        # Trading insights
        if trade_summary['win_rate'] > 0.6:
            insights.append("High win rate indicating consistent execution")
        elif trade_summary['win_rate'] > 0.5:
            insights.append("Balanced win/loss ratio")
        else:
            insights.append("Low win rate suggesting need for signal improvement")
        
        # Alpha insights
        if perf_summary['alpha'] > 0.05:
            insights.append("Strong positive alpha indicating skill-based returns")
        elif perf_summary['alpha'] > 0:
            insights.append("Positive alpha with some skill-based returns")
        else:
            insights.append("Negative alpha suggesting benchmark underperformance")
        
        return insights
    
    def _determine_strategy_type(self) -> str:
        """Determine strategy type based on characteristics"""
        returns = self.strategy_returns
        autocorr = returns.autocorr()
        
        if autocorr > 0.1:
            return "Momentum/Trend Following"
        elif autocorr < -0.1:
            return "Mean Reversion"
        else:
            return "Statistical Arbitrage"
    
    def _determine_asset_class(self) -> str:
        """Determine asset class"""
        # This would need to be enhanced based on actual data
        return "Equities"
    
    def _determine_universe(self) -> str:
        """Determine trading universe"""
        # This would need to be enhanced based on actual data
        return "S&P 500"
    
    def _determine_signal_frequency(self) -> str:
        """Determine signal frequency"""
        trade_metrics = self.trade_analytics.calculate_execution_metrics()
        frequency = trade_metrics['trade_frequency']
        
        if frequency > 1000:
            return "High Frequency (1000+ trades/year)"
        elif frequency > 100:
            return "Medium Frequency (100-1000 trades/year)"
        else:
            return "Low Frequency (<100 trades/year)"
    
    def _determine_holding_period(self) -> str:
        """Determine average holding period"""
        turnover_metrics = self.trade_analytics.calculate_turnover_metrics()
        holding_period = turnover_metrics['avg_holding_period']
        
        if holding_period < 5:
            return "Short-term (<5 days)"
        elif holding_period < 30:
            return "Medium-term (5-30 days)"
        else:
            return "Long-term (>30 days)"
    
    def _create_execution_assumptions(self) -> Dict:
        """Create execution assumptions"""
        return {
            'slippage': "N/A (Not implemented)",
            'commission': "N/A (Not implemented)",
            'liquidity_filters': "N/A (Not implemented)",
            'execution_quality': "N/A (Not implemented)"
        }
    
    def _determine_leverage_assumptions(self) -> str:
        """Determine leverage assumptions"""
        # This would need to be enhanced based on actual data
        return "No leverage (100% cash)"
    
    def _extract_backtest_parameters(self) -> Dict:
        """Extract backtest parameters"""
        # Use user-specified period if available, otherwise fall back to actual strategy period
        if self.user_start_date and self.user_end_date:
            start_date = self.user_start_date
            end_date = self.user_end_date
            # Calculate total days from user period
            from datetime import datetime
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            total_days = (end_dt - start_dt).days
        else:
            # Fallback to actual strategy period
            start_date = self.strategy_returns.index[0].strftime('%Y-%m-%d')
            end_date = self.strategy_returns.index[-1].strftime('%Y-%m-%d')
            total_days = len(self.strategy_returns)
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'total_days': total_days,
            'initial_capital': self.backtest_results.get('initial_capital', 100000),
            'data_source': self.backtest_results.get('data_source', 'Yahoo Finance')
        }
    
    def _create_report_metadata(self) -> Dict:
        """Create report metadata"""
        return {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'strategy_name': self.strategy_name,
            'analysis_period': f"{self.strategy_returns.index[0].strftime('%Y-%m-%d')} to {self.strategy_returns.index[-1].strftime('%Y-%m-%d')}",
            'total_observations': len(self.strategy_returns),
            'report_version': '1.0',
            'analytics_version': '1.0'
        }
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics for quick overview"""
        perf_summary = self.performance_metrics.get_performance_summary()
        risk_summary = self.risk_analysis.get_risk_summary()
        trade_summary = self.trade_analytics.get_trade_summary()
        
        return {
            'performance': perf_summary,
            'risk': risk_summary,
            'trading': trade_summary
        } 

    def _calculate_period_metrics(self) -> Dict:
        """Calculate returns and volatility for different time periods"""
        from datetime import datetime, timedelta
        
        # Get current date and calculate period end dates
        current_date = self.strategy_returns.index[-1] if len(self.strategy_returns) > 0 else datetime.now()
        
        # Define periods with proper date calculations
        periods = {}
        
        # MTD: From start of current month to end of data
        mtd_start = current_date.replace(day=1)
        if mtd_start <= current_date:
            periods['MTD'] = mtd_start
        
        # YTD: From start of current year to end of data
        ytd_start = current_date.replace(month=1, day=1)
        if ytd_start <= current_date:
            periods['YTD'] = ytd_start
        
        # 1Y: Last 365 calendar days from end of data (will have different trading days than YTD)
        one_year_start = current_date - timedelta(days=365)
        if one_year_start >= self.strategy_returns.index[0]:
            periods['1Y'] = one_year_start
        
        # 3Y: Last 3 years
        three_year_start = current_date - timedelta(days=3*365)
        if three_year_start >= self.strategy_returns.index[0]:
            periods['3Y'] = three_year_start
        
        # 5Y: Last 5 years
        five_year_start = current_date - timedelta(days=5*365)
        if five_year_start >= self.strategy_returns.index[0]:
            periods['5Y'] = five_year_start
        
        # All Time: From start of data
        periods['All Time'] = self.strategy_returns.index[0]
        
        # Calculate metrics for each period
        period_metrics = {}
        for period_name, start_date in periods.items():
            # Filter data for this period
            period_returns = self.strategy_returns[self.strategy_returns.index >= start_date]
            period_benchmark_returns = self.benchmark_returns[self.benchmark_returns.index >= start_date]
            
            if len(period_returns) > 0:
                # Calculate annualized return
                total_return = (1 + period_returns).prod() - 1
                days_in_period = (period_returns.index[-1] - period_returns.index[0]).days
                annualized_return = (1 + total_return) ** (365 / days_in_period) - 1 if days_in_period > 0 else 0
                
                # Calculate annualized volatility
                volatility = period_returns.std() * np.sqrt(252)
                
                # Calculate benchmark metrics
                if len(period_benchmark_returns) > 0:
                    benchmark_total_return = (1 + period_benchmark_returns).prod() - 1
                    benchmark_annualized_return = (1 + benchmark_total_return) ** (365 / days_in_period) - 1 if days_in_period > 0 else 0
                    benchmark_volatility = period_benchmark_returns.std() * np.sqrt(252)
                else:
                    benchmark_annualized_return = 0
                    benchmark_volatility = 0
                
                period_metrics[period_name] = {
                    'strategy_return': annualized_return,
                    'strategy_volatility': volatility,
                    'benchmark_return': benchmark_annualized_return,
                    'benchmark_volatility': benchmark_volatility
                }
            else:
                period_metrics[period_name] = {
                    'strategy_return': 0,
                    'strategy_volatility': 0,
                    'benchmark_return': 0,
                    'benchmark_volatility': 0
                }
        
        return period_metrics

    def _extract_signals(self) -> pd.DataFrame:
        """Extract signals from backtest results"""
        if 'signals' in self.backtest_results and self.backtest_results['signals'] is not None:
            return self.backtest_results['signals']
        return pd.DataFrame()

    def _calculate_benchmark_rolling_sharpe(self, window: int = 60) -> pd.Series:
        """Calculate rolling Sharpe ratio for benchmark"""
        if len(self.benchmark_returns) == 0:
            return pd.Series(dtype=float)
        
        # Use a smaller window for shorter datasets
        if len(self.benchmark_returns) < window:
            window = max(20, len(self.benchmark_returns) // 3)  # Use 1/3 of data or minimum 20 days
        
        # Calculate rolling Sharpe ratio for benchmark with proper excess returns
        excess_returns = self.benchmark_returns - (0.02 / 252)  # 2% risk-free rate
        rolling_mean = excess_returns.rolling(window).mean()
        rolling_std = excess_returns.rolling(window).std()
        
        # Annualize the Sharpe ratio
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        
        return rolling_sharpe

    def _calculate_benchmark_drawdown_series(self) -> pd.Series:
        """Calculate drawdown series for benchmark"""
        if len(self.benchmark_returns) == 0:
            return pd.Series(dtype=float)
        
        # Calculate cumulative returns for benchmark
        benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
        
        # Calculate drawdown series
        peak = benchmark_cumulative.expanding().max()
        drawdown_series = (benchmark_cumulative - peak) / peak
        
        return drawdown_series 