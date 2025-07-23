import os
import pandas as pd
from typing import Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAlignmentManager:
    """Centralized data alignment and trimming manager"""
    
    def __init__(self, user_start_date: str, user_end_date: str):
        self.user_start_date = pd.to_datetime(user_start_date)
        self.user_end_date = pd.to_datetime(user_end_date)
        self.alignment_log = []
    
    def align_and_trim_all_data(
        self, 
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        strategy_type: str,
        strategy_params: Dict,
        additional_data: Optional[Dict] = None
    ) -> Dict:
        """
        Single point of truth for all data alignment and trimming.
        
        Args:
            strategy_returns: Strategy returns series
            benchmark_returns: Benchmark returns series
            strategy_type: Type of strategy
            strategy_params: Strategy parameters
            additional_data: Additional data to align (portfolio_values, signals, etc.)
            
        Returns:
            Dict containing all aligned and trimmed data
        """
        logger.info(f"🔄 Starting comprehensive data alignment")
        logger.info(f"   User period: {self.user_start_date.strftime('%Y-%m-%d')} to {self.user_end_date.strftime('%Y-%m-%d')}")
        
        # Step 1: Align returns by index
        aligned_returns = self._align_returns(strategy_returns, benchmark_returns)
        
        # Step 2: Calculate effective start date based on strategy requirements
        effective_start = self._calculate_effective_start_date(
            aligned_returns, strategy_type, strategy_params
        )
        
        # Step 3: Trim all data to effective period
        trimmed_data = self._trim_all_data_to_period(
            aligned_returns, effective_start, additional_data
        )
        
        # Step 4: Validate alignment
        self._validate_alignment(trimmed_data)
        
        # Step 5: Log results
        self._log_alignment_results(trimmed_data)
        
        return trimmed_data
    
    def _align_returns(self, strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
        """Align strategy and benchmark returns by index"""
        logger.info("📊 Aligning strategy and benchmark returns")
        
        # Create DataFrame with both returns
        aligned = pd.DataFrame({
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns
        })
        
        # Fill missing values with 0
        aligned = aligned.fillna(0)
        
        # Save aligned returns
        csv_path = 'data/csv/returns_strategy_vs_benchmark.csv'
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        aligned.to_csv(csv_path)
        
        logger.info(f"   Aligned returns shape: {aligned.shape}")
        logger.info(f"   Period: {aligned.index[0].strftime('%Y-%m-%d')} to {aligned.index[-1].strftime('%Y-%m-%d')}")
        
        return aligned
    
    def _calculate_effective_start_date(
        self, 
        aligned_returns: pd.DataFrame, 
        strategy_type: str, 
        strategy_params: Dict
    ) -> pd.Timestamp:
        """Calculate effective start date considering strategy requirements"""
        logger.info(f"🎯 Calculating effective start date for {strategy_type}")
        
        if strategy_type == 'trend_following':
            lookback_period = strategy_params.get('lookback_period', 200)
            # Need lookback_period days before user_start_date
            required_start = self.user_start_date - pd.Timedelta(days=lookback_period)
            # Find the closest available date
            available_dates = aligned_returns.index
            effective_start = available_dates[available_dates >= required_start][0]
            
            logger.info(f"   Lookback period: {lookback_period} days")
            logger.info(f"   Required start: {required_start.strftime('%Y-%m-%d')}")
            logger.info(f"   Effective start: {effective_start.strftime('%Y-%m-%d')}")
            
        else:
            # Default: Use user start date
            effective_start = self.user_start_date
            logger.info(f"   Using user start date: {effective_start.strftime('%Y-%m-%d')}")
        
        return effective_start
    
    def _trim_all_data_to_period(
        self, 
        aligned_returns: pd.DataFrame, 
        effective_start: pd.Timestamp,
        additional_data: Optional[Dict] = None
    ) -> Dict:
        """Trim all data to the effective period"""
        logger.info(f"✂️  Trimming all data to effective period")
        
        # Use the actual data end date instead of user end date to avoid mismatches
        actual_end_date = aligned_returns.index[-1]
        effective_end = min(actual_end_date, self.user_end_date)
        
        # Trim aligned returns
        trimmed_returns = aligned_returns[
            (aligned_returns.index >= effective_start) & 
            (aligned_returns.index <= effective_end)
        ]
        
        result = {
            'strategy_returns': trimmed_returns['strategy_returns'],
            'benchmark_returns': trimmed_returns['benchmark_returns'],
            'effective_start_date': effective_start.strftime('%Y-%m-%d'),
            'effective_end_date': effective_end.strftime('%Y-%m-%d')
        }
        
        # Trim additional data if provided
        if additional_data:
            for key, data in additional_data.items():
                if data is not None and hasattr(data, 'index'):
                    if isinstance(data, pd.Series):
                        result[key] = data[
                            (data.index >= effective_start) & 
                            (data.index <= effective_end)
                        ]
                    elif isinstance(data, pd.DataFrame):
                        result[key] = data[
                            (data.index >= effective_start) & 
                            (data.index <= effective_end)
                        ]
                    logger.info(f"   Trimmed {key}: {len(result[key])} days")
        
        logger.info(f"   Final period: {effective_start.strftime('%Y-%m-%d')} to {effective_end.strftime('%Y-%m-%d')}")
        logger.info(f"   Total days: {len(trimmed_returns)}")
        
        return result
    
    def _validate_alignment(self, trimmed_data: Dict):
        """Validate that all data components are properly aligned"""
        logger.info("🔍 Validating data alignment")
        
        # Check that all components have the same period
        components = ['strategy_returns', 'benchmark_returns']
        periods = {}
        
        for component in components:
            if component in trimmed_data and len(trimmed_data[component]) > 0:
                periods[component] = (
                    trimmed_data[component].index[0].strftime('%Y-%m-%d'),
                    trimmed_data[component].index[-1].strftime('%Y-%m-%d')
                )
        
        # Validate consistency
        if len(set(periods.values())) > 1:
            logger.error("❌ Data alignment validation failed!")
            for component, period in periods.items():
                logger.error(f"   {component}: {period[0]} to {period[1]}")
            raise ValueError("Data components have inconsistent periods")
        
        logger.info("✅ Data alignment validation passed")
    
    def _log_alignment_results(self, trimmed_data: Dict):
        """Log comprehensive alignment results"""
        logger.info("📋 Data alignment summary:")
        logger.info(f"   User requested: {self.user_start_date.strftime('%Y-%m-%d')} to {self.user_end_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Effective period: {trimmed_data['effective_start_date']} to {trimmed_data['effective_end_date']}")
        logger.info(f"   Strategy returns: {len(trimmed_data['strategy_returns'])} days")
        logger.info(f"   Benchmark returns: {len(trimmed_data['benchmark_returns'])} days")
        
        # Log additional components
        for key, data in trimmed_data.items():
            if key not in ['strategy_returns', 'benchmark_returns', 'effective_start_date', 'effective_end_date']:
                if hasattr(data, '__len__'):
                    logger.info(f"   {key}: {len(data)} days")

# Backward compatibility functions
def align_series(series1, series2, fill_value=0, col1=None, col2=None, save_csv_path=None):
    """
    Align two pandas Series by index and return a DataFrame with both columns.
    Optionally save the result as a CSV.

    Args:
        series1 (pd.Series): First series
        series2 (pd.Series): Second series
        fill_value: Value to fill for missing data (default: 0)
        col1 (str): Optional name for the first column
        col2 (str): Optional name for the second column
        save_csv_path (str): Optional path to save the aligned DataFrame as CSV

    Returns:
        pd.DataFrame: DataFrame with aligned series
    """
    s1 = series1.rename(col1) if col1 else series1
    s2 = series2.rename(col2) if col2 else series2
    aligned = pd.concat([s1, s2], axis=1)
    aligned = aligned.fillna(fill_value)
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        aligned.to_csv(save_csv_path)
    return aligned

def align_returns(series1, series2, col1='strategy_returns', col2='benchmark_returns'):
    """
    Backward-compatible alias for aligning returns. Always saves to the original CSV path.
    """
    csv_path = 'data/csv/returns_strategy_vs_benchmark.csv'
    return align_series(series1, series2, fill_value=0, col1=col1, col2=col2, save_csv_path=csv_path)

def align_returns_trimmed(series1, series2, strategy_type, strategy_params, user_start_date=None, user_end_date=None, additional_data=None, col1='strategy_returns', col2='benchmark_returns'):
    """
    Enhanced alignment function with comprehensive trimming and validation.
    """
    if user_start_date and user_end_date:
        # Use new alignment manager
        alignment_manager = DataAlignmentManager(user_start_date, user_end_date)
        result = alignment_manager.align_and_trim_all_data(
            series1, series2, strategy_type, strategy_params, additional_data
        )
        return result
    else:
        # Fallback to original logic for backward compatibility
        return _legacy_align_returns_trimmed(series1, series2, strategy_type, strategy_params, col1, col2)

def _legacy_align_returns_trimmed(series1, series2, strategy_type, strategy_params, col1='strategy_returns', col2='benchmark_returns'):
    """Original alignment logic for backward compatibility"""
    aligned_returns = align_returns(series1, series2, col1=col1, col2=col2)
    
    if strategy_type == 'trend_following':
        lookback_period = strategy_params.get('lookback_period', 20)
        aligned_returns_trimmed = aligned_returns.iloc[lookback_period:]
    else:
        # Default: no trimming
        aligned_returns_trimmed = aligned_returns
    
    return aligned_returns_trimmed 