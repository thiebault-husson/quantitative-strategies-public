"""
Strategy Analysis Utilities

This module provides utilities to analyze strategy classes and detect
rolling window parameters to optimize data loading and backtesting.
"""

import inspect
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import re


class StrategyAnalyzer:
    """Analyzes strategy classes to detect rolling windows and optimize data loading."""
    
    @staticmethod
    def detect_rolling_windows(strategy_instance) -> Dict[str, int]:
        """
        Detect rolling window parameters in a strategy instance.
        
        Args:
            strategy_instance: Instance of a strategy class
            
        Returns:
            Dict[str, int]: Dictionary mapping parameter names to window sizes
        """
        rolling_windows = {}
        
        # Method 1: Check strategy instance attributes
        rolling_windows.update(StrategyAnalyzer._detect_rolling_in_attributes(strategy_instance))
        
        # Method 2: Check strategy methods for rolling operations
        rolling_windows.update(StrategyAnalyzer._detect_rolling_in_methods(strategy_instance))
        
        # Method 3: Check strategy class definition
        rolling_windows.update(StrategyAnalyzer._detect_rolling_in_class(strategy_instance))
        
        return rolling_windows
    
    @staticmethod
    def _detect_rolling_in_attributes(strategy_instance) -> Dict[str, int]:
        """
        Detect rolling window parameters in strategy instance attributes.
        
        Args:
            strategy_instance: Instance of a strategy class
            
        Returns:
            Dict[str, int]: Dictionary mapping parameter names to window sizes
        """
        rolling_windows = {}
        
        # Get all attributes of the strategy instance
        for attr_name in dir(strategy_instance):
            if not attr_name.startswith('_'):  # Skip private attributes
                attr_value = getattr(strategy_instance, attr_name)
                
                # Check for common rolling window parameter names
                if isinstance(attr_value, int) and attr_value > 1:
                    window_keywords = [
                        'window', 'lookback', 'period', 'rolling', 'lag',
                        'volatility_lookback', 'lookback_period', 'ma_window',
                        'sma_window', 'ema_window', 'rolling_window', 'span',
                        'length', 'bars', 'days', 'sessions'
                    ]
                    
                    if any(keyword in attr_name.lower() for keyword in window_keywords):
                        rolling_windows[attr_name] = attr_value
        
        return rolling_windows
    
    @staticmethod
    def _detect_rolling_in_methods(strategy_instance) -> Dict[str, int]:
        """
        Analyze strategy methods to detect rolling window operations.
        
        Args:
            strategy_instance: Instance of a strategy class
            
        Returns:
            Dict[str, int]: Dictionary mapping detected operations to window sizes
        """
        rolling_windows = {}
        
        # Get the strategy class methods
        strategy_methods = [
            'generate_signals', 'size_positions', 'calculate_returns',
            'calculate_indicators', 'get_signals', 'calculate_volatility',
            'calculate_momentum', 'calculate_ma', 'calculate_sma', 'calculate_ema'
        ]
        
        for method_name in strategy_methods:
            if hasattr(strategy_instance, method_name):
                method = getattr(strategy_instance, method_name)
                if callable(method):
                    # Get method source code if available
                    try:
                        source = inspect.getsource(method)
                        
                        # Look for rolling operations in the source code
                        rolling_patterns = [
                            r'\.rolling\(window=(\d+)\)',
                            r'\.rolling\((\d+)\)',
                            r'rolling\(window=(\d+)\)',
                            r'rolling\((\d+)\)',
                            r'window=(\d+)',
                            r'lookback=(\d+)',
                            r'period=(\d+)',
                            r'\.mean\(\)',  # Look for mean() after rolling
                            r'\.std\(\)',   # Look for std() after rolling
                            r'\.sum\(\)',   # Look for sum() after rolling
                        ]
                        
                        for pattern in rolling_patterns:
                            matches = re.findall(pattern, source)
                            for match in matches:
                                if match.isdigit():
                                    window_size = int(match)
                                    if window_size > 1:
                                        rolling_windows[f'{method_name}_rolling_{window_size}'] = window_size
                    
                    except (OSError, TypeError):
                        # Source code not available, skip
                        pass
        
        return rolling_windows
    
    @staticmethod
    def _detect_rolling_in_class(strategy_instance) -> Dict[str, int]:
        """
        Analyze strategy class definition for rolling window parameters.
        
        Args:
            strategy_instance: Instance of a strategy class
            
        Returns:
            Dict[str, int]: Dictionary mapping detected parameters to window sizes
        """
        rolling_windows = {}
        
        try:
            # Get the class source code
            class_source = inspect.getsource(strategy_instance.__class__)
            
            # Look for rolling window patterns in the class definition
            patterns = [
                r'lookback_period\s*=\s*(\d+)',
                r'volatility_lookback\s*=\s*(\d+)',
                r'window\s*=\s*(\d+)',
                r'period\s*=\s*(\d+)',
                r'rolling_window\s*=\s*(\d+)',
                r'ma_window\s*=\s*(\d+)',
                r'sma_window\s*=\s*(\d+)',
                r'ema_window\s*=\s*(\d+)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, class_source)
                for match in matches:
                    if match.isdigit():
                        window_size = int(match)
                        if window_size > 1:
                            rolling_windows[f'class_param_{window_size}'] = window_size
        
        except (OSError, TypeError):
            # Source code not available, skip
            pass
        
        return rolling_windows
    
    @staticmethod
    def get_max_rolling_window(strategy_instance) -> int:
        """
        Get the maximum rolling window size from a strategy.
        
        Args:
            strategy_instance: Instance of a strategy class
            
        Returns:
            int: Maximum rolling window size (0 if no rolling windows detected)
        """
        rolling_windows = StrategyAnalyzer.detect_rolling_windows(strategy_instance)
        
        if not rolling_windows:
            return 0
        
        return max(rolling_windows.values())
    
    @staticmethod
    def calculate_extended_start_date(start_date: Union[str, datetime], 
                                    strategy_instance, 
                                    buffer_days: int = None) -> str:
        """
        Calculate an extended start date to accommodate rolling windows.
        
        Args:
            start_date: Original start date
            strategy_instance: Instance of a strategy class
            buffer_days: Additional buffer days to add (auto-calculated if None)
            
        Returns:
            str: Extended start date in 'YYYY-MM-DD' format
        """
        # Convert to datetime if string
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            start_dt = start_date
        
        # Get maximum rolling window
        max_window = StrategyAnalyzer.get_max_rolling_window(strategy_instance)
        
        # Auto-calculate buffer days if not provided
        if buffer_days is None:
            # Smart buffer calculation based on window size
            if max_window <= 20:
                buffer_days = 10
            elif max_window <= 50:
                buffer_days = 20
            elif max_window <= 100:
                buffer_days = 30
            elif max_window <= 200:
                buffer_days = 50
            else:
                buffer_days = max_window // 4  # 25% of window size
        
        # Calculate extended start date
        # Add buffer days to account for weekends, holidays, and data quality
        total_days_back = max_window + buffer_days
        extended_start_dt = start_dt - timedelta(days=total_days_back)
        
        return extended_start_dt.strftime('%Y-%m-%d')
    
    @staticmethod
    def analyze_strategy(strategy_instance) -> Dict:
        """
        Comprehensive strategy analysis.
        
        Args:
            strategy_instance: Instance of a strategy class
            
        Returns:
            Dict: Analysis results including rolling windows, recommendations, etc.
        """
        rolling_windows = StrategyAnalyzer.detect_rolling_windows(strategy_instance)
        max_window = max(rolling_windows.values()) if rolling_windows else 0
        
        # Calculate recommended buffer based on window size
        if max_window <= 20:
            recommended_buffer = 10
        elif max_window <= 50:
            recommended_buffer = 20
        elif max_window <= 100:
            recommended_buffer = 30
        elif max_window <= 200:
            recommended_buffer = 50
        else:
            recommended_buffer = max_window // 4
        
        return {
            'rolling_windows': rolling_windows,
            'max_rolling_window': max_window,
            'has_rolling_windows': len(rolling_windows) > 0,
            'recommended_buffer_days': recommended_buffer,
            'total_required_days': max_window + recommended_buffer,
            'strategy_class': strategy_instance.__class__.__name__,
            'analysis_timestamp': datetime.now().isoformat(),
            'detection_methods': {
                'attributes': len([k for k in rolling_windows.keys() if not k.startswith('class_param_') and not '_rolling_' in k]),
                'methods': len([k for k in rolling_windows.keys() if '_rolling_' in k]),
                'class_params': len([k for k in rolling_windows.keys() if k.startswith('class_param_')])
            }
        }
    
    @staticmethod
    def validate_data_sufficiency(data_start_date: str, 
                                strategy_start_date: str, 
                                strategy_instance) -> Dict:
        """
        Validate if the loaded data is sufficient for the strategy's rolling windows.
        
        Args:
            data_start_date: Start date of loaded data
            strategy_start_date: Desired strategy start date
            strategy_instance: Instance of a strategy class
            
        Returns:
            Dict: Validation results with recommendations
        """
        analysis = StrategyAnalyzer.analyze_strategy(strategy_instance)
        required_days = analysis['total_required_days']
        
        # Calculate actual days available
        data_start = datetime.strptime(data_start_date, '%Y-%m-%d')
        strategy_start = datetime.strptime(strategy_start_date, '%Y-%m-%d')
        actual_days = (strategy_start - data_start).days
        
        is_sufficient = actual_days >= required_days
        deficit = max(0, required_days - actual_days)
        
        return {
            'is_sufficient': is_sufficient,
            'required_days': required_days,
            'actual_days': actual_days,
            'deficit_days': deficit,
            'data_start_date': data_start_date,
            'strategy_start_date': strategy_start_date,
            'recommendation': f"Load {deficit} more days of data" if deficit > 0 else "Data is sufficient",
            'analysis': analysis
        } 