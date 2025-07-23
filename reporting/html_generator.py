import os
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import json
import pandas as pd

# Import analytics engine
from .analytics import AnalyticsEngine

# Paths
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class HTMLReportGenerator:
    def __init__(self):
        pass
    
    
    def _force_fresh_calculations(self, backtest_results, strategy_name, start_date, end_date):
        """Force fresh calculations to ensure consistency with audit"""
        print("🔧 Forcing fresh calculations for consistency...")
        
        # Create fresh analytics engine instance
        analytics_engine = AnalyticsEngine(backtest_results, strategy_name, start_date, end_date)
        
        # Force data consistency
        analytics_engine._ensure_data_consistency_with_audit()
        
        # Generate fresh analysis
        analysis_data = analytics_engine.generate_comprehensive_analysis()
        
        print("✅ Fresh calculations completed")
        return analysis_data

    def generate_report(self, backtest_results, strategy_name, start_date, end_date, params, position_sizes):
        """
        Generate a comprehensive HTML report using analytics engine.
        
        Args:
            backtest_results: Results from backtest engine
            strategy_name: Name of the strategy
            start_date: Start date of the backtest
            end_date: End date of the backtest
            params: Dictionary of strategy parameters
            position_sizes: DataFrame with position sizes over time
        """
        print("📄 Generating comprehensive report...")
        
        # Validate data periods before generating report
        if start_date and end_date:
            expected_start = pd.to_datetime(start_date)
            expected_end = pd.to_datetime(end_date)
            
            # Check if backtest results match expected period
            if 'strategy_returns' in backtest_results and len(backtest_results['strategy_returns']) > 0:
                actual_start = backtest_results['strategy_returns'].index[0]
                actual_end = backtest_results['strategy_returns'].index[-1]
                
                if actual_start != expected_start or actual_end != expected_end:
                    print(f"⚠️  WARNING: Backtest results period mismatch")
                    print(f"   Expected: {expected_start.strftime('%Y-%m-%d')} to {expected_end.strftime('%Y-%m-%d')}")
                    print(f"   Actual: {actual_start.strftime('%Y-%m-%d')} to {actual_end.strftime('%Y-%m-%d')}")
                    
                    # Trim backtest results to expected period
                    for key in backtest_results:
                        if isinstance(backtest_results[key], pd.Series) and hasattr(backtest_results[key], 'index'):
                            backtest_results[key] = backtest_results[key][
                                (backtest_results[key].index >= expected_start) & 
                                (backtest_results[key].index <= expected_end)
                            ]
                        elif isinstance(backtest_results[key], pd.DataFrame) and hasattr(backtest_results[key], 'index'):
                            backtest_results[key] = backtest_results[key][
                                (backtest_results[key].index >= expected_start) & 
                                (backtest_results[key].index <= expected_end)
                            ]

        
        # Force fresh calculations for consistency
        analysis_data = self._force_fresh_calculations(backtest_results, strategy_name, start_date, end_date)
        
        # Prepare data for template
        template_data = self._prepare_template_data(analysis_data, strategy_name, start_date, end_date, params)
        
        # Generate report
        return self._render_report(template_data, strategy_name)
    
    def _prepare_template_data(self, analysis_data, strategy_name, start_date, end_date, params):
        """Prepare data for template rendering"""
        # Extract nested metrics from analytics engine
        performance_metrics = analysis_data['performance_metrics']
        risk_metrics = analysis_data['risk_metrics']
        trade_metrics = analysis_data['trade_metrics']
        period_metrics = analysis_data['period_metrics']
        
        # Extract individual metric categories
        return_metrics = performance_metrics.get('return_metrics', {})
        risk_metrics_data = performance_metrics.get('risk_metrics', {})
        risk_adjusted_metrics = performance_metrics.get('risk_adjusted_metrics', {})
        drawdown_metrics = performance_metrics.get('drawdown_metrics', {})
        benchmark_metrics = performance_metrics.get('benchmark_metrics', {})
        
        # Calculate beta directly from the data if not available in benchmark_metrics
        if 'beta' not in benchmark_metrics or benchmark_metrics['beta'] == 0:
            # Extract strategy and benchmark returns from the backtest results
            strategy_returns = analysis_data.get('strategy_returns', [])
            benchmark_returns = analysis_data.get('benchmark_returns', [])
            
            if len(strategy_returns) > 0 and len(benchmark_returns) > 0:
                import numpy as np
                # Calculate beta manually
                covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                calculated_beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                benchmark_metrics['beta'] = calculated_beta
                print(f"🔧 Calculated beta directly: {calculated_beta:.4f}")
            else:
                # No data available - set to None to show NA in report
                benchmark_metrics['beta'] = None
                print(f"⚠️ No data available for beta calculation - will show NA in report")
        else:
            # Beta is available - use as is
            print(f"🔧 Using provided beta value: {benchmark_metrics['beta']:.4f}")
        
        # Format metrics for template (flat structure with proper formatting)
        formatted_metrics = {
            # Return metrics
            'cumulative_return': f"{return_metrics.get('total_return', 0):.2%}" if return_metrics.get('total_return') is not None else 'N/A',
            'annualized_return': f"{return_metrics.get('annualized_return', 0):.2%}" if return_metrics.get('annualized_return') is not None else 'N/A',
            
            # Risk metrics
            'volatility': f"{risk_metrics_data.get('volatility', 0):.2%}" if risk_metrics_data.get('volatility') is not None else 'N/A',
            'max_drawdown': f"{drawdown_metrics.get('max_drawdown', 0):.2%}" if drawdown_metrics.get('max_drawdown') is not None else 'N/A',
            'var_95': f"{risk_metrics_data.get('var_95', 0):.2%}" if risk_metrics_data.get('var_95') is not None else 'N/A',
            'var_99': f"{risk_metrics_data.get('var_99', 0):.2%}" if risk_metrics_data.get('var_99') is not None else 'N/A',
            'tail_risk': f"{risk_metrics_data.get('cvar_95', 0):.2%}" if risk_metrics_data.get('cvar_95') is not None else 'N/A',
            
            # Risk-adjusted metrics
            'sharpe_ratio': f"{risk_adjusted_metrics.get('sharpe_ratio', 0):.2f}" if risk_adjusted_metrics.get('sharpe_ratio') is not None else 'N/A',
            'sortino_ratio': f"{risk_adjusted_metrics.get('sortino_ratio', 0):.2f}" if risk_adjusted_metrics.get('sortino_ratio') is not None else 'N/A',
            'calmar_ratio': f"{risk_adjusted_metrics.get('calmar_ratio', 0):.2f}" if risk_adjusted_metrics.get('calmar_ratio') is not None else 'N/A',
            
            # Benchmark metrics
            'alpha': f"{benchmark_metrics.get('alpha', 0):.2%}" if benchmark_metrics.get('alpha') is not None else 'N/A',
            'beta': f"{benchmark_metrics.get('beta', 0):.4f}" if benchmark_metrics.get('beta') is not None else 'N/A',
            
            # Trading metrics (based on available data)
            'win_rate': f"{trade_metrics.get('trade_summary', {}).get('win_rate', 0):.1%}" if trade_metrics.get('trade_summary', {}).get('win_rate') is not None else 'N/A',
            'turnover': f"{trade_metrics.get('turnover_analysis', {}).get('annualized_turnover', 0):.2f}" if trade_metrics.get('turnover_analysis', {}).get('annualized_turnover') is not None else 'N/A',
            
            # Benchmark comparison metrics
            'benchmark_annualized_return': f"{benchmark_metrics.get('benchmark_annualized_return', 0):.2%}" if benchmark_metrics.get('benchmark_annualized_return') is not None else 'N/A',
            'benchmark_volatility': f"{benchmark_metrics.get('benchmark_volatility', 0):.2%}" if benchmark_metrics.get('benchmark_volatility') is not None else 'N/A',
            'benchmark_max_drawdown': f"{benchmark_metrics.get('benchmark_max_drawdown', 0):.2%}" if benchmark_metrics.get('benchmark_max_drawdown') is not None else 'N/A',
            'benchmark_var_95': f"{benchmark_metrics.get('benchmark_var_95', 0):.2%}" if benchmark_metrics.get('benchmark_var_95') is not None else 'N/A',
            'benchmark_var_99': f"{benchmark_metrics.get('benchmark_var_99', 0):.2%}" if benchmark_metrics.get('benchmark_var_99') is not None else 'N/A',
            'benchmark_tail_risk': f"{benchmark_metrics.get('benchmark_tail_risk', 0):.2%}" if benchmark_metrics.get('benchmark_tail_risk') is not None else 'N/A',
            'benchmark_sharpe_ratio': f"{benchmark_metrics.get('benchmark_sharpe_ratio', 0):.2f}" if benchmark_metrics.get('benchmark_sharpe_ratio') is not None else 'N/A',
            'benchmark_sortino_ratio': f"{benchmark_metrics.get('benchmark_sortino_ratio', 0):.2f}" if benchmark_metrics.get('benchmark_sortino_ratio') is not None else 'N/A',
            'benchmark_calmar_ratio': f"{benchmark_metrics.get('benchmark_calmar_ratio', 0):.2f}" if benchmark_metrics.get('benchmark_calmar_ratio') is not None else 'N/A',
            'benchmark_hit_ratio': f"{benchmark_metrics.get('benchmark_hit_ratio', 0):.1%}" if benchmark_metrics.get('benchmark_hit_ratio') is not None else 'N/A',
        }
        
        # Create benchmark comparison data
        benchmark_comparison = {
            'strategy_total_return': formatted_metrics['cumulative_return'],
            'benchmark_total_return': f"{benchmark_metrics.get('benchmark_return', 0):.2%}" if benchmark_metrics.get('benchmark_return') is not None else 'N/A',
            'excess_return': f"{benchmark_metrics.get('excess_return', 0):.2%}" if benchmark_metrics.get('excess_return') is not None else 'N/A',
            'tracking_error': f"{risk_adjusted_metrics.get('tracking_error', 0):.2%}" if risk_adjusted_metrics.get('tracking_error') is not None else 'N/A',
            'commentary': 'Strategy performance analysis based on backtest results.'
        }
        
        # Create trades summary
        # Calculate total trading days from strategy returns length
        strategy_returns = analysis_data['performance_metrics'].get('strategy_returns', [])
        total_trading_days = len(strategy_returns) if hasattr(strategy_returns, '__len__') else 0
        
        # Fallback to monthly returns if strategy returns not available
        if total_trading_days == 0:
            monthly_returns = analysis_data['performance_metrics']['return_metrics'].get('monthly_returns', [])
            total_trading_days = len(monthly_returns) * 21  # Approximate trading days per month
        
        # Get trade metrics with data source information
        trade_metrics = analysis_data['trade_metrics']
        trade_summary = trade_metrics.get('trade_summary', {})
        
        trades_summary = {
            # Top 5 Crucial Signal Metrics - pass raw numbers for template formatting
            'daily_signal_hit_rate': trade_summary.get('daily_signal_hit_rate', 0) if trade_summary.get('daily_signal_hit_rate') is not None else 0,
            'signals_per_month': trade_summary.get('signals_per_month', 0) if trade_summary.get('signals_per_month') is not None else 0,
            'position_balance': trade_summary.get('position_balance', 0) if trade_summary.get('position_balance') is not None else 0,
            'convergence_ratio': 0,  # Not applicable for trend following strategy
            'avg_daily_turnover_signal': trade_summary.get('avg_daily_turnover_signal', 0) if trade_summary.get('avg_daily_turnover_signal') is not None else 0,
            
            # Additional Trade Metrics
            'total_trades': trade_summary.get('total_trades', total_trading_days * 2) if trade_summary.get('total_trades') is not None else 'N/A',
            'win_rate': trade_summary.get('win_rate', 0) if trade_summary.get('win_rate') is not None else 0,
            'avg_trade_pnl': trade_summary.get('avg_trade_pnl', 0) if trade_summary.get('avg_trade_pnl') is not None else 0,
            'avg_holding_period': trade_metrics.get('turnover_analysis', {}).get('avg_holding_period', 0) if trade_metrics.get('turnover_analysis', {}).get('avg_holding_period') is not None else 0,
            'annualized_turnover': trade_metrics.get('turnover_analysis', {}).get('annualized_turnover', 0) if trade_metrics.get('turnover_analysis', {}).get('annualized_turnover') is not None else 0,
            'capacity_score': trade_summary.get('capacity_score', 0) if trade_summary.get('capacity_score') is not None else 0,
            'data_source': trade_summary.get('data_source', 'daily_returns')
        }
        
        # Create risk summary
        risk_summary = {
            'decomposition': [
                {'name': 'Strategy Risk', 'risk_contribution': 'N/A', 'return_contribution': 'N/A'}
            ],
            'volatility_plot': '{}',  # Empty JSON for now
            'return_plot': '{}',      # Empty JSON for now
        }
        
        # Create notes
        notes = {
            'lookback': f"{params.get('lookback_period', 'N/A')} days",
            'universe': f"{len(params.get('strategy_tickers', []))} assets",
            'slippage': params.get('slippage', 'N/A'),
            'costs': params.get('transaction_costs', 'N/A'),
            'environment': 'Backtest',
            'additional': 'This report was generated using historical data and backtest results.'
        }
        
        return {
            'strategy_name': strategy_name,
            'strategy_description': f'{strategy_name} - Quantitative Strategy Performance Report',
            'start_date': start_date,
            'end_date': end_date,
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'code_version': 'v2.0.0',
            
            # Analytics data
            'executive_summary': analysis_data['executive_summary'],
            'strategy_overview': analysis_data['strategy_overview'],
            'performance_metrics': analysis_data['performance_metrics'],
            'metrics': formatted_metrics,  # Properly formatted metrics for template
            'risk_metrics': analysis_data['risk_metrics'],
            'trade_metrics': analysis_data['trade_metrics'],
            'period_metrics': period_metrics,  # Add period metrics
            'charts_data': json.dumps(analysis_data['charts_data']),
            'report_metadata': analysis_data['report_metadata'],
            
            # Template compatibility fields
            'benchmark_comparison': benchmark_comparison,
            'trades': trades_summary,
            'risk': risk_summary,
            'notes': notes,
            
            # Additional fields that might be expected
            'aum': f"${params.get('initial_cash', 0):,.0f}" if params.get('initial_cash') is not None else 'N/A',
            'benchmarks': ['S&P 500'],
            'universe': params.get('strategy_tickers', []),
            'allocation': {},
            'top_holdings': params.get('strategy_tickers', [])[:5],
            'equity_curve_plot': '{}',  # Empty JSON for now
            'initial_cash': params.get('initial_cash', 0)
        }
    
    def _render_report(self, template_data, strategy_name):
        """Render the HTML report"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        strategy_slug = strategy_name.lower().replace(' ', '_')
        output_filename = f'{timestamp}_{strategy_slug}_strategy/Comprehensive_Report.html'
        
        return render_report(template_data, output_filename)

def render_report(data, output_filename):
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template('base_report.html.j2')
    html = template.render(**data)
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Create the report directory
    report_dir = os.path.dirname(output_path)
    os.makedirs(report_dir, exist_ok=True)
    
    # Copy CSS file to report directory
    css_source = os.path.join(TEMPLATE_DIR, '..', 'static', 'css', 'report.css')
    css_dest = os.path.join(report_dir, 'report.css')
    
    try:
        import shutil
        shutil.copy2(css_source, css_dest)
        print(f'CSS file copied to: {css_dest}')
    except Exception as e:
        print(f'Warning: Could not copy CSS file: {e}')
    
    # Write the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report generated: {output_path}')
    return output_path

if __name__ == '__main__':
    # Test the HTMLReportGenerator
    generator = HTMLReportGenerator()
    print("HTMLReportGenerator ready for use") 