#!/usr/bin/env python3
"""
Main entry point for quantitative strategy backtesting and reporting.
This replaces the monolithic run.py with a clean, modular architecture.
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.trend_following import TrendFollowingStrategy
from backtest.engine import BacktestEngine
from utils.align import align_returns, align_returns_trimmed
from data.index_components import get_sp500_tickers
from data.historical_data import get_historical_data
from reporting.html_generator import HTMLReportGenerator
from utils.config import get_strategy_config, print_strategy_info
from utils.logger import setup_logger
from utils.strategy_analyzer import StrategyAnalyzer
import pandas as pd

def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', length=50, fill='█'):
    """Display progress bar in terminal."""
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()

def print_step_progress(step, total_steps, step_name):
    """Print progress for individual steps."""
    print_progress_bar(step, total_steps, prefix=f'Step {step}/{total_steps}', suffix=f'{step_name}', length=40)

def load_configuration():
    """Load and validate configuration."""
    print_step_progress(1, 6, "Loading Configuration")
    config = get_strategy_config()
    print_strategy_info()
    return config

def setup_strategy(config):
    """Setup strategy based on configuration."""
    print_step_progress(2, 6, "Configuring Strategy")
    
    STRATEGY_TYPE = config['strategy_type']
    strategy_params = config['strategy_params']
    common_params = config['common_params']
    
    # Extract common parameters
    start_date = common_params['start_date']
    end_date = common_params['end_date']
    yahoofinance_field = common_params['yahoofinance_field']
    initial_cash = common_params['initial_cash']
    
    if STRATEGY_TYPE == 'trend_following':
        # FULL S&P 500: Use all S&P 500 tickers for complete backtest
        strategy_tickers = get_sp500_tickers(limit=None)
        benchmark_ticker = '^GSPC'
        strategy_name = config['description']['name']
        strategy_class = TrendFollowingStrategy
        
        # Extract strategy parameters
        lookback_period = strategy_params['lookback_period']
        volatility_lookback = strategy_params['volatility_lookback']
        risk_per_trade = strategy_params['risk_per_trade']
        max_position_size = strategy_params['max_position_size']
        
        strategy = strategy_class(
            lookback_period=lookback_period,
            volatility_lookback=volatility_lookback,
            risk_per_trade=risk_per_trade,
            max_position_size=max_position_size
        )
    else:
        raise ValueError(f"Unknown strategy type: {STRATEGY_TYPE}")
    
    # Analyze strategy for rolling windows
    strategy_analysis = StrategyAnalyzer.analyze_strategy(strategy)
    
    # Calculate extended start date if rolling windows detected
    extended_start_date = start_date
    if strategy_analysis['has_rolling_windows']:
        extended_start_date = StrategyAnalyzer.calculate_extended_start_date(
            start_date, strategy, buffer_days=None  # Auto-calculate buffer
        )
        print(f"🔍 Strategy analysis: Found rolling windows {strategy_analysis['rolling_windows']}")
        print(f"📅 Extended data loading period: {extended_start_date} to {end_date}")
        print(f"   (Original strategy period: {start_date} to {end_date})")
        print(f"   (Required buffer: {strategy_analysis['recommended_buffer_days']} days)")
    else:
        print("🔍 Strategy analysis: No rolling windows detected")
    
    return {
        'strategy': strategy,
        'strategy_tickers': strategy_tickers,
        'benchmark_ticker': benchmark_ticker,
        'strategy_name': strategy_name,
        'initial_cash': initial_cash,
        'start_date': start_date,
        'end_date': end_date,
        'extended_start_date': extended_start_date,
        'yahoofinance_field': yahoofinance_field,
        'strategy_params': strategy_params,
        'strategy_type': STRATEGY_TYPE,
        'strategy_analysis': strategy_analysis
    }

def load_data(strategy_tickers, benchmark_ticker, extended_start_date, end_date, yahoofinance_field, strategy=None, original_start_date=None):
    """Load market data with extended period for rolling windows."""
    print_step_progress(3, 6, "Loading Market Data")
    print(f"📊 Loading data for {', '.join(strategy_tickers)} from {extended_start_date} to {end_date}")
    
    strategy_components_price_data = get_historical_data(
        strategy_tickers, extended_start_date, end_date, fields=yahoofinance_field
    )
    benchmark_price_data = get_historical_data(
        [benchmark_ticker], extended_start_date, end_date, fields=yahoofinance_field
    )
    
    print(f"✅ Data loaded successfully. Shape: {strategy_components_price_data.shape}")
    
    # Validate data sufficiency if strategy and original start date are provided
    if strategy is not None and original_start_date is not None:
        actual_data_start = strategy_components_price_data.index[0].strftime('%Y-%m-%d')
        validation = StrategyAnalyzer.validate_data_sufficiency(
            actual_data_start, original_start_date, strategy
        )
        
        if not validation['is_sufficient']:
            print(f"⚠️  WARNING: Insufficient data for strategy requirements!")
            print(f"   Required: {validation['required_days']} days")
            print(f"   Available: {validation['actual_days']} days")
            print(f"   Deficit: {validation['deficit_days']} days")
            print(f"   Recommendation: {validation['recommendation']}")
        else:
            print(f"✅ Data sufficiency validated: {validation['actual_days']} days available")
    
    return strategy_components_price_data, benchmark_price_data

def run_backtest(strategy, strategy_components_price_data, initial_cash):
    """Execute backtest."""
    print_step_progress(4, 6, "Running Backtest")
    print("🔄 Running backtest...")
    
    start_time = time.time()
    engine = BacktestEngine(strategy, strategy_components_price_data, initial_cash=initial_cash)
    results = engine.run()
    end_time = time.time()
    
    print(f"✅ Backtest completed successfully! (Time: {end_time - start_time:.2f}s)")
    return results

def validate_data_period(data, start_date, end_date, data_name):
    """Validate that data is within the specified period"""
    if data is None or len(data) == 0:
        return True
    
    data_start = data.index[0].strftime('%Y-%m-%d')
    data_end = data.index[-1].strftime('%Y-%m-%d')
    
    if data_start < start_date or data_end > end_date:
        print(f"⚠️  WARNING: {data_name} period ({data_start} to {data_end}) outside user period ({start_date} to {end_date})")
        return False
    
    return True

def process_results(results, benchmark_price_data, benchmark_ticker, strategy_name, strategy_type, strategy_params, start_date, original_start_date):
    """Process backtest results - NO TRIMMING, pass through extended data."""
    print_step_progress(5, 6, "Processing Results")
    
    strategy_returns = results['returns']
    strategy_returns.name = strategy_name
    benchmark_returns = benchmark_price_data[f'{benchmark_ticker}_Open'].pct_change()
    
    # Log the data we're working with
    print(f"📊 Processing backtest results:")
    print(f"   Strategy returns: {strategy_returns.index[0].strftime('%Y-%m-%d')} to {strategy_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Benchmark returns: {benchmark_returns.index[0].strftime('%Y-%m-%d')} to {benchmark_returns.index[-1].strftime('%Y-%m-%d')}")
    print(f"   User requested: {original_start_date} to end")
    
    # Prepare additional data for alignment
    additional_data = {
        'portfolio_values': results.get('portfolio_values'),
        'signals': results.get('signals'),
        'position_sizes': results.get('position_sizes'),
        'prices': results.get('prices')
    }
    
    # Use enhanced alignment function (single point of truth)
    aligned_data = align_returns_trimmed(
        strategy_returns,
        benchmark_returns,
        strategy_type,
        strategy_params,
        user_start_date=original_start_date,
        user_end_date=results.get('end_date', '2025-12-31'),
        additional_data=additional_data
    )
    
    print(f"✅ Data alignment completed successfully")
    print(f"   Effective period: {aligned_data['effective_start_date']} to {aligned_data['effective_end_date']}")
    
    # Validate the aligned data
    from utils.data_validator import DataValidator
    validation_results = DataValidator.validate_backtest_results(
        aligned_data, 
        (aligned_data['effective_start_date'], aligned_data['effective_end_date'])
    )
    
    if not validation_results['is_valid']:
        print("❌ Data validation failed - check logs for details")
    
    return aligned_data

def generate_reports(aligned_data, strategy_name, start_date, end_date, 
                    strategy_params, strategy_tickers, yahoofinance_field, results, benchmark_ticker):
    """Generate comprehensive reports using aligned data."""
    print_step_progress(6, 6, "Generating Reports")
    print("📄 Generating comprehensive report...")
    
    # Prepare parameters for the report
    params = {
        'strategy_tickers': strategy_tickers,
        'start_date': aligned_data['effective_start_date'],  # Use effective start date
        'end_date': aligned_data['effective_end_date'],      # Use effective end date
        'yahoofinance_field': yahoofinance_field,
        'strategy_name': strategy_name,
        **strategy_params
    }
    
    # Prepare backtest results for analytics engine - use aligned data
    backtest_results = {
        'strategy_returns': aligned_data['strategy_returns'],
        'benchmark_returns': aligned_data['benchmark_returns'],
        'portfolio_values': aligned_data.get('portfolio_values'),
        'position_sizes': aligned_data.get('position_sizes'),
        'signals': aligned_data.get('signals'),
        'prices': aligned_data.get('prices'),
        'trades': results.get('trades', None),
        'initial_capital': results.get('initial_cash', 100000),
        'data_source': 'Yahoo Finance'
    }
    
    # Log the actual periods being used
    print(f"📊 Report configuration:")
    print(f"   User requested: {start_date} to {end_date}")
    print(f"   Effective period: {aligned_data['effective_start_date']} to {aligned_data['effective_end_date']}")
    print(f"   Strategy returns: {len(aligned_data['strategy_returns'])} days")
    print(f"   Benchmark returns: {len(aligned_data['benchmark_returns'])} days")
    
    # Generate comprehensive HTML report
    report_generator = HTMLReportGenerator()
    report_path = report_generator.generate_report(
        backtest_results=backtest_results,
        strategy_name=strategy_name,
        start_date=aligned_data['effective_start_date'],  # Use effective dates
        end_date=aligned_data['effective_end_date'],
        params=params,
        position_sizes=aligned_data.get('position_sizes')
    )
    
    print(f"✅ Comprehensive report generated: {report_path}")
    return report_path

def main():
    """Main execution function."""
    print("🚀 Starting Quantitative Strategy Backtest")
    print("=" * 60)
    
    try:
        # Setup logging
        logger = setup_logger()
        
        # Load configuration
        config = load_configuration()
        
        # Setup strategy
        strategy_config = setup_strategy(config)
        
        # Load data
        strategy_components_price_data, benchmark_price_data = load_data(
            strategy_config['strategy_tickers'],
            strategy_config['benchmark_ticker'],
            strategy_config['extended_start_date'],
            strategy_config['end_date'],
            strategy_config['yahoofinance_field'],
            strategy_config['strategy'],
            strategy_config['start_date']  # Original start date for validation
        )
        
        # Run backtest
        results = run_backtest(
            strategy_config['strategy'],
            strategy_components_price_data,
            strategy_config['initial_cash']
        )
        
        # Process results
        aligned_returns_trimmed = process_results(
            results,
            benchmark_price_data,
            strategy_config['benchmark_ticker'],
            strategy_config['strategy_name'],
            strategy_config['strategy_type'],
            strategy_config['strategy_params'],
            strategy_config['extended_start_date'],
            strategy_config['start_date']  # Original start date
        )
        
        # Generate reports
        report_path = generate_reports(
            aligned_returns_trimmed,
            strategy_config['strategy_name'],
            strategy_config['start_date'],
            strategy_config['end_date'],
            strategy_config['strategy_params'],
            strategy_config['strategy_tickers'],
            strategy_config['yahoofinance_field'],
            results,
            strategy_config['benchmark_ticker']
        )
        
        print(f"🎉 {strategy_config['strategy_name']} backtest completed successfully!")
        print(f"📊 Report saved to: {report_path}")
        
    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")
        logger.error(f"Backtest failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 