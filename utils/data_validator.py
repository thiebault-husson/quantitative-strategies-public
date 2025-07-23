"""
Data Validation Pipeline
Ensures data consistency across all components
"""

import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for backtest results"""
    
    @staticmethod
    def validate_backtest_results(results: Dict, expected_period: Tuple[str, str]) -> Dict:
        """
        Validate backtest results for consistency and correctness.
        
        Args:
            results: Backtest results dictionary
            expected_period: Tuple of (start_date, end_date)
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'period_info': {},
            'data_quality': {}
        }
        
        expected_start, expected_end = expected_period
        expected_start_dt = pd.to_datetime(expected_start)
        expected_end_dt = pd.to_datetime(expected_end)
        
        # Validate strategy returns
        if 'strategy_returns' in results:
            validation_results.update(
                DataValidator._validate_returns_series(
                    results['strategy_returns'], 'strategy_returns', expected_start_dt, expected_end_dt
                )
            )
        
        # Validate benchmark returns
        if 'benchmark_returns' in results:
            validation_results.update(
                DataValidator._validate_returns_series(
                    results['benchmark_returns'], 'benchmark_returns', expected_start_dt, expected_end_dt
                )
            )
        
        # Validate portfolio values
        if 'portfolio_values' in results:
            validation_results.update(
                DataValidator._validate_portfolio_values(
                    results['portfolio_values'], expected_start_dt, expected_end_dt
                )
            )
        
        # Validate data consistency
        validation_results.update(
            DataValidator._validate_data_consistency(results)
        )
        
        # Log validation results
        DataValidator._log_validation_results(validation_results)
        
        return validation_results
    
    @staticmethod
    def _validate_returns_series(returns: pd.Series, name: str, expected_start: pd.Timestamp, expected_end: pd.Timestamp) -> Dict:
        """Validate a returns series"""
        validation = {'warnings': [], 'errors': [], 'period_info': {}}
        
        if returns is None or len(returns) == 0:
            validation['errors'].append(f"{name}: Empty or None series")
            return validation
        
        actual_start = returns.index[0]
        actual_end = returns.index[-1]
        
        validation['period_info'][name] = {
            'expected_start': expected_start.strftime('%Y-%m-%d'),
            'expected_end': expected_end.strftime('%Y-%m-%d'),
            'actual_start': actual_start.strftime('%Y-%m-%d'),
            'actual_end': actual_end.strftime('%Y-%m-%d'),
            'days': len(returns)
        }
        
        # Check period alignment
        if actual_start != expected_start:
            validation['warnings'].append(
                f"{name}: Start date mismatch (expected {expected_start.strftime('%Y-%m-%d')}, got {actual_start.strftime('%Y-%m-%d')})"
            )
        
        if actual_end != expected_end:
            validation['warnings'].append(
                f"{name}: End date mismatch (expected {expected_end.strftime('%Y-%m-%d')}, got {actual_end.strftime('%Y-%m-%d')})"
            )
        
        # Check for NaN values
        nan_count = returns.isna().sum()
        if nan_count > 0:
            validation['warnings'].append(f"{name}: Contains {nan_count} NaN values")
        
        # Check for infinite values
        inf_count = (returns == float('inf')).sum() + (returns == float('-inf')).sum()
        if inf_count > 0:
            validation['errors'].append(f"{name}: Contains {inf_count} infinite values")
        
        return validation
    
    @staticmethod
    def _validate_portfolio_values(portfolio_values: pd.Series, expected_start: pd.Timestamp, expected_end: pd.Timestamp) -> Dict:
        """Validate portfolio values"""
        validation = {'warnings': [], 'errors': [], 'period_info': {}}
        
        if portfolio_values is None or len(portfolio_values) == 0:
            validation['errors'].append("Portfolio values: Empty or None series")
            return validation
        
        actual_start = portfolio_values.index[0]
        actual_end = portfolio_values.index[-1]
        
        validation['period_info']['portfolio_values'] = {
            'expected_start': expected_start.strftime('%Y-%m-%d'),
            'expected_end': expected_end.strftime('%Y-%m-%d'),
            'actual_start': actual_start.strftime('%Y-%m-%d'),
            'actual_end': actual_end.strftime('%Y-%m-%d'),
            'days': len(portfolio_values)
        }
        
        # Check for negative values
        negative_count = (portfolio_values < 0).sum()
        if negative_count > 0:
            validation['errors'].append(f"Portfolio values: Contains {negative_count} negative values")
        
        # Check for decreasing values (should be non-decreasing)
        decreasing_count = (portfolio_values.diff() < 0).sum()
        if decreasing_count > 0:
            validation['warnings'].append(f"Portfolio values: Contains {decreasing_count} decreasing periods")
        
        return validation
    
    @staticmethod
    def _validate_data_consistency(results: Dict) -> Dict:
        """Validate consistency between different data components"""
        validation = {'warnings': [], 'errors': []}
        
        components = []
        for key in ['strategy_returns', 'benchmark_returns', 'portfolio_values']:
            if key in results and results[key] is not None and len(results[key]) > 0:
                components.append((key, results[key].index[0], results[key].index[-1]))
        
        if len(components) > 1:
            # Check if all components have the same period
            first_component, first_start, first_end = components[0]
            
            for name, start, end in components[1:]:
                if start != first_start or end != first_end:
                    validation['errors'].append(
                        f"Period inconsistency: {first_component} ({first_start.strftime('%Y-%m-%d')} to {first_end.strftime('%Y-%m-%d')}) "
                        f"vs {name} ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})"
                    )
        
        return validation
    
    @staticmethod
    def _log_validation_results(validation_results: Dict):
        """Log validation results"""
        if validation_results['errors']:
            logger.error("❌ Data validation errors:")
            for error in validation_results['errors']:
                logger.error(f"   {error}")
        
        if validation_results['warnings']:
            logger.warning("⚠️  Data validation warnings:")
            for warning in validation_results['warnings']:
                logger.warning(f"   {warning}")
        
        if validation_results['period_info']:
            logger.info("📊 Data period information:")
            for component, info in validation_results['period_info'].items():
                logger.info(f"   {component}: {info['actual_start']} to {info['actual_end']} ({info['days']} days)")
        
        if not validation_results['errors'] and not validation_results['warnings']:
            logger.info("✅ Data validation passed successfully")
