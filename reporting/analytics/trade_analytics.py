"""
Trade Analytics Module
Analyzes trade execution, PnL distribution, turnover, and implementation metrics
Uses actual strategy data when available, with fallbacks to returns-based calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TradeAnalytics:
    """Trade analytics for quantitative strategy backtests"""
    
    def __init__(self, backtest_results: Dict, strategy_returns: pd.Series):
        """
        Initialize TradeAnalytics with backtest results and strategy returns.
        
        Args:
            backtest_results: Dictionary containing backtest results
            strategy_returns: Series of strategy returns
        """
        self.backtest_results = backtest_results
        self.strategy_returns = strategy_returns
        
        # Extract actual strategy data
        self.signals = self._extract_signals()
        self.position_sizes = self._extract_position_sizes()
        self.portfolio_values = self._extract_portfolio_values()
        
        # Extract individual asset returns for proper PnL calculation
        self.asset_returns = self._extract_asset_returns()
        
        # Create trade data from actual signals
        self.trades = self._create_trades_from_signals()
        
    def _extract_signals(self) -> pd.DataFrame:
        """Extract actual strategy signals from backtest results"""
        if 'signals' in self.backtest_results and self.backtest_results['signals'] is not None:
            signals = self.backtest_results['signals']
            return signals
        else:
            # Fallback: create synthetic signals from returns
            return self._create_synthetic_signals()
    
    def _extract_position_sizes(self) -> pd.DataFrame:
        """Extract actual position sizes from backtest results"""
        if 'position_sizes' in self.backtest_results and self.backtest_results['position_sizes'] is not None:
            position_sizes = self.backtest_results['position_sizes']
            return position_sizes
        else:
            # Fallback: create synthetic position sizes
            return self._create_synthetic_position_sizes()
    
    def _extract_portfolio_values(self) -> pd.Series:
        """Extract portfolio values from backtest results"""
        if 'portfolio_values' in self.backtest_results and self.backtest_results['portfolio_values'] is not None:
            return self.backtest_results['portfolio_values']
        else:
            # Fallback: calculate from returns
            return (1 + self.strategy_returns).cumprod()
    
    def _extract_asset_returns(self) -> pd.DataFrame:
        """Extract individual asset returns from backtest results"""
        if 'asset_returns' in self.backtest_results and self.backtest_results['asset_returns'] is not None:
            return self.backtest_results['asset_returns']
        elif 'prices' in self.backtest_results and self.backtest_results['prices'] is not None:
            # Calculate asset returns from prices
            prices = self.backtest_results['prices']
            return prices.pct_change()
        else:
            # Fallback: create empty DataFrame with same columns as signals
            if not self.signals.empty:
                return pd.DataFrame(index=self.signals.index, columns=self.signals.columns).fillna(0)
            else:
                return pd.DataFrame()
    
    def _create_synthetic_signals(self) -> pd.DataFrame:
        """Create synthetic signals from returns (fallback method)"""
        # Create binary signals based on return direction
        signals = pd.DataFrame(index=self.strategy_returns.index)
        signals['strategy'] = np.where(self.strategy_returns > 0, 1, 0)
        return signals
    
    def _create_synthetic_position_sizes(self) -> pd.DataFrame:
        """Create synthetic position sizes (fallback method)"""
        # Create position sizes based on return magnitude
        position_sizes = pd.DataFrame(index=self.strategy_returns.index)
        position_sizes['strategy'] = np.abs(self.strategy_returns)
        return position_sizes
    
    def _create_trades_from_signals(self) -> pd.DataFrame:
        """Create trade data from actual strategy signals"""
        if self.signals.empty:
            return pd.DataFrame()
        
        # Find signal changes (actual trades)
        signal_changes = []
        
        for col in self.signals.columns:
            col_signals = self.signals[col]
            # Find days where signal changes
            signal_diff = col_signals.diff().fillna(0)
            trade_days = signal_diff != 0
            
            for date in col_signals.index[trade_days]:
                old_signal = col_signals.shift(1).loc[date]
                new_signal = col_signals.loc[date]
                
                if pd.notna(old_signal) and pd.notna(new_signal):
                    # Determine trade type
                    if old_signal == 0 and new_signal != 0:
                        trade_type = 'BUY' if new_signal > 0 else 'SELL'
                    elif old_signal != 0 and new_signal == 0:
                        trade_type = 'SELL' if old_signal > 0 else 'BUY'
                    elif old_signal != new_signal:
                        trade_type = 'BUY' if new_signal > old_signal else 'SELL'
                    else:
                        continue
                    
                    # Calculate PnL for this trade
                    pnl = self._calculate_trade_pnl(date, col, old_signal, new_signal)
                    
                    signal_changes.append({
                        'date': date,
                        'asset': col,
                        'type': trade_type,
                        'old_signal': old_signal,
                        'new_signal': new_signal,
                        'pnl': pnl,
                        'size': abs(new_signal - old_signal)
                    })
        
        return pd.DataFrame(signal_changes)
    
    def _calculate_trade_pnl(self, date: pd.Timestamp, asset: str, old_signal: float, new_signal: float) -> float:
        """Calculate PnL for a specific trade"""
        try:
            # Get position size for this asset
            if asset in self.position_sizes.columns:
                position_size = self.position_sizes[asset].loc[date]
            else:
                position_size = 1.0  # Default position size
            
            # Get individual asset return for this date
            if not self.asset_returns.empty and asset in self.asset_returns.columns:
                asset_return = self.asset_returns[asset].loc[date]
            else:
                asset_return = 0.0  # Fallback to zero if asset return not found
            
            # PnL = position change * return
            position_change = new_signal - old_signal
            pnl = position_change * position_size * asset_return
            
            return pnl
        except:
            return 0.0
    
    def calculate_all_trade_metrics(self) -> Dict:
        """Calculate all trade-related metrics using standardized methods"""
        return {
            'trade_summary': self.calculate_trade_summary(),
            'pnl_analysis': self.calculate_pnl_analysis(),
            'execution_metrics': self.calculate_execution_metrics(),
            'turnover_analysis': self.calculate_turnover_metrics(),
            'trade_distribution': self.calculate_trade_distribution(),
            'capacity_analysis': self.calculate_capacity_metrics()
        }
    
    def calculate_trade_summary(self) -> Dict:
        """Calculate basic trade summary metrics using actual strategy data"""
        # Method 1: Use actual trades from signals
        if not self.trades.empty:
            total_trades = len(self.trades)
            winning_trades = len(self.trades[self.trades['pnl'] > 0])
            losing_trades = len(self.trades[self.trades['pnl'] < 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_trade_pnl = self.trades['pnl'].mean()
            avg_winning_trade = self.trades[self.trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_losing_trade = self.trades[self.trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            
            # Calculate profit factor
            if losing_trades > 0 and avg_losing_trade != 0:
                profit_factor = abs(avg_winning_trade * winning_trades / (avg_losing_trade * losing_trades))
            else:
                profit_factor = float('inf') if winning_trades > 0 else 0
            
            total_pnl = self.trades['pnl'].sum()
            
        else:
            # Method 2: Fallback to daily returns-based calculation
            total_trades = len(self.strategy_returns) * 2  # Daily buy/sell pairs
            positive_days = (self.strategy_returns > 0).sum()
            total_days = len(self.strategy_returns)
            
            win_rate = positive_days / total_days if total_days > 0 else 0
            avg_trade_pnl = self.strategy_returns.mean()
            avg_winning_trade = self.strategy_returns[self.strategy_returns > 0].mean() if positive_days > 0 else 0
            avg_losing_trade = self.strategy_returns[self.strategy_returns < 0].mean() if (total_days - positive_days) > 0 else 0
            
            # Calculate profit factor
            losing_days = total_days - positive_days
            if losing_days > 0 and avg_losing_trade != 0:
                profit_factor = abs(avg_winning_trade * positive_days / (avg_losing_trade * losing_days))
            else:
                profit_factor = float('inf') if positive_days > 0 else 0
            
            total_pnl = self.strategy_returns.sum()
        
        # Add signal analysis directly to trade summary
        signal_analysis = self.calculate_signal_analysis()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades if not self.trades.empty else positive_days,
            'losing_trades': losing_trades if not self.trades.empty else (total_days - positive_days),
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl,
            'data_source': 'actual_signals' if not self.trades.empty else 'daily_returns',
            # Include signal analysis metrics
            'daily_signal_hit_rate': signal_analysis['daily_signal_hit_rate'],
            'signals_per_month': signal_analysis['signals_per_month'],
            'position_balance': signal_analysis['position_balance'],
            'avg_daily_turnover_signal': signal_analysis['avg_daily_turnover']
        }
    
    def calculate_pnl_analysis(self) -> Dict:
        """Calculate PnL distribution and analysis"""
        if not self.trades.empty:
            # Use actual trade PnL
            pnls = self.trades['pnl']
        else:
            # Fallback to daily returns
            pnls = self.strategy_returns
        
        if len(pnls) == 0:
            return {
                'pnl_statistics': {
                    'mean': 0.0, 'median': 0.0, 'std': 0.0, 'skewness': 0.0, 
                    'kurtosis': 0.0, 'min': 0.0, 'max': 0.0, 'q25': 0.0, 'q75': 0.0
                },
                'pnl_percentiles': {f'p{p}': 0.0 for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
                'largest_wins': pd.Series(),
                'largest_losses': pd.Series(),
                'pnl_series': pd.Series(),
                'data_source': 'none'
            }
        
        # PnL distribution statistics
        pnl_stats = {
            'mean': pnls.mean(),
            'median': pnls.median(),
            'std': pnls.std(),
            'skewness': pnls.skew(),
            'kurtosis': pnls.kurtosis(),
            'min': pnls.min(),
            'max': pnls.max(),
            'q25': pnls.quantile(0.25),
            'q75': pnls.quantile(0.75)
        }
        
        # PnL percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pnl_percentiles = {f'p{p}': pnls.quantile(p/100) for p in percentiles}
        
        # Largest wins and losses
        largest_wins = pnls[pnls > 0].nlargest(5)
        largest_losses = pnls[pnls < 0].nsmallest(5)
        
        return {
            'pnl_statistics': pnl_stats,
            'pnl_percentiles': pnl_percentiles,
            'largest_wins': largest_wins,
            'largest_losses': largest_losses,
            'pnl_series': pnls,
            'data_source': 'actual_trades' if not self.trades.empty else 'daily_returns'
        }
    
    def calculate_execution_metrics(self) -> Dict:
        """Calculate execution-related metrics"""
        if not self.trades.empty:
            # Use actual trade data
            total_trades = len(self.trades)
            trade_frequency = total_trades / (len(self.strategy_returns) / 252)  # trades per year
            avg_trade_size = self.trades['size'].mean()
            trade_size_volatility = self.trades['size'].std()
        else:
            # Fallback to returns-based calculation
            total_trades = len(self.strategy_returns) * 2
            trade_frequency = total_trades / (len(self.strategy_returns) / 252)
            avg_trade_size = np.abs(self.strategy_returns).mean()
            trade_size_volatility = np.abs(self.strategy_returns).std()
        
        # Execution quality (simplified)
        estimated_slippage = avg_trade_size * 0.001  # 10 basis points per trade
        total_slippage = estimated_slippage * total_trades
        
        return {
            'trade_frequency': trade_frequency,
            'avg_trade_interval': pd.Timedelta(days=252/trade_frequency) if trade_frequency > 0 else pd.Timedelta(days=0),
            'avg_trade_size': avg_trade_size,
            'trade_size_volatility': trade_size_volatility,
            'estimated_slippage': total_slippage,
            'slippage_per_trade': estimated_slippage,
            'data_source': 'actual_trades' if not self.trades.empty else 'daily_returns'
        }
    
    def calculate_turnover_metrics(self) -> Dict:
        """Calculate turnover and position change metrics using actual position data"""
        if not self.position_sizes.empty:
            # Use actual position changes
            position_changes = self.position_sizes.diff().abs()
            total_turnover = position_changes.sum().sum()  # Sum across all assets and time
            annualized_turnover = total_turnover * 252 / len(self.position_sizes)
            
            # Calculate average holding period from position changes
            position_change_days = (position_changes.sum(axis=1) > 0).sum()
            avg_holding_period = len(self.position_sizes) / (position_change_days + 1) if position_change_days > 0 else len(self.position_sizes)
            
            # Monthly turnover
            monthly_turnover = position_changes.sum(axis=1).resample('M').sum()
            
        else:
            # Fallback to returns-based calculation
            return_changes = self.strategy_returns.abs()
            total_turnover = return_changes.sum()
            annualized_turnover = total_turnover * 252 / len(self.strategy_returns)
            avg_holding_period = len(self.strategy_returns) / (total_turnover + 1)
            monthly_turnover = return_changes.resample('M').sum()
        
        return {
            'total_turnover': total_turnover,
            'annualized_turnover': annualized_turnover,
            'avg_holding_period': avg_holding_period,
            'monthly_turnover': monthly_turnover,
            'turnover_volatility': monthly_turnover.std(),
            'data_source': 'actual_positions' if not self.position_sizes.empty else 'daily_returns'
        }
    
    def calculate_trade_distribution(self) -> Dict:
        """Calculate trade distribution and patterns"""
        if not self.trades.empty:
            # Use actual trade data
            trade_sizes = self.trades['size']
            size_distribution = {
                'small_trades': len(trade_sizes[trade_sizes <= trade_sizes.quantile(0.33)]),
                'medium_trades': len(trade_sizes[(trade_sizes > trade_sizes.quantile(0.33)) & 
                                               (trade_sizes <= trade_sizes.quantile(0.67))]),
                'large_trades': len(trade_sizes[trade_sizes > trade_sizes.quantile(0.67)])
            }
            
            # Trade timing analysis
            trade_dates = pd.to_datetime(self.trades['date'])
            trade_days = trade_dates.dt.dayofweek
            trade_months = trade_dates.dt.month
            
            day_distribution = trade_days.value_counts().sort_index()
            month_distribution = trade_months.value_counts().sort_index()
            
            # Consecutive wins/losses
            pnls = self.trades['pnl']
            consecutive_wins = self._calculate_consecutive_events(pnls > 0)
            consecutive_losses = self._calculate_consecutive_events(pnls < 0)
            
        else:
            # Fallback to returns-based calculation
            return_sizes = np.abs(self.strategy_returns)
            size_distribution = {
                'small_trades': len(return_sizes[return_sizes <= return_sizes.quantile(0.33)]),
                'medium_trades': len(return_sizes[(return_sizes > return_sizes.quantile(0.33)) & 
                                                (return_sizes <= return_sizes.quantile(0.67))]),
                'large_trades': len(return_sizes[return_sizes > return_sizes.quantile(0.67)])
            }
            
            day_distribution = pd.Series()
            month_distribution = pd.Series()
            consecutive_wins = self._calculate_consecutive_events(self.strategy_returns > 0)
            consecutive_losses = self._calculate_consecutive_events(self.strategy_returns < 0)
        
        return {
            'size_distribution': size_distribution,
            'day_distribution': day_distribution,
            'month_distribution': month_distribution,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'avg_consecutive_wins': self._calculate_avg_consecutive_events(self.trades['pnl'] > 0 if not self.trades.empty else self.strategy_returns > 0),
            'avg_consecutive_losses': self._calculate_avg_consecutive_events(self.trades['pnl'] < 0 if not self.trades.empty else self.strategy_returns < 0),
            'data_source': 'actual_trades' if not self.trades.empty else 'daily_returns'
        }
    
    def _calculate_consecutive_events(self, condition: pd.Series) -> int:
        """Calculate maximum consecutive events"""
        max_consecutive = 0
        current_consecutive = 0
        
        for event in condition:
            if event:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_avg_consecutive_events(self, condition: pd.Series) -> float:
        """Calculate average consecutive events"""
        consecutive_counts = []
        current_consecutive = 0
        
        for event in condition:
            if event:
                current_consecutive += 1
            else:
                if current_consecutive > 0:
                    consecutive_counts.append(current_consecutive)
                current_consecutive = 0
        
        # Add final streak if it exists
        if current_consecutive > 0:
            consecutive_counts.append(current_consecutive)
        
        return np.mean(consecutive_counts) if consecutive_counts else 0
    
    def calculate_capacity_metrics(self) -> Dict:
        """Calculate capacity and scalability metrics"""
        if not self.trades.empty:
            # Use actual trade data
            avg_trade_size = self.trades['size'].mean()
            trade_size_volatility = self.trades['size'].std()
            trade_frequency = len(self.trades) / (len(self.strategy_returns) / 252)
        else:
            # Fallback to returns-based calculation
            avg_trade_size = np.abs(self.strategy_returns).mean()
            trade_size_volatility = np.abs(self.strategy_returns).std()
            trade_frequency = len(self.strategy_returns) * 2 / (len(self.strategy_returns) / 252)
        
        # Capacity estimation (simplified)
        capacity_score = 1 / (1 + trade_size_volatility / avg_trade_size) if avg_trade_size > 0 else 0
        
        # Liquidity requirements (simplified)
        liquidity_requirement = avg_trade_size * (1 + trade_size_volatility / avg_trade_size) if avg_trade_size > 0 else 0
        
        # Scalability factors
        scalability_score = min(1, trade_frequency / 1000)  # Normalize to 0-1
        
        return {
            'capacity_score': capacity_score,
            'liquidity_requirement': liquidity_requirement,
            'scalability_score': scalability_score,
            'trade_frequency': trade_frequency,
            'avg_trade_size': avg_trade_size,
            'trade_size_volatility': trade_size_volatility,
            'data_source': 'actual_trades' if not self.trades.empty else 'daily_returns'
        }
    
    def calculate_signal_analysis(self) -> Dict:
        """Calculate the top 5 crucial signal metrics"""
        # 1. Daily Signal Hit Rate - percentage of days with signals that were profitable
        if not self.signals.empty:
            signal_days = 0
            profitable_signal_days = 0
            
            for date in self.strategy_returns.index:
                # Check if there's a signal on this date
                daily_signals = self.signals.loc[date] if date in self.signals.index else pd.Series([0])
                has_signal = (daily_signals != 0).any()
                
                if has_signal:
                    signal_days += 1
                    # Check if the day was profitable
                    if self.strategy_returns.loc[date] > 0:
                        profitable_signal_days += 1
            
            daily_signal_hit_rate = profitable_signal_days / signal_days if signal_days > 0 else 0
        else:
            # Fallback: use positive days as signal days
            positive_days = (self.strategy_returns > 0).sum()
            total_days = len(self.strategy_returns)
            daily_signal_hit_rate = positive_days / total_days if total_days > 0 else 0
            signal_days = total_days
        
        # 2. Signal Frequency (signals per month) - count actual signal days, not changes
        if not self.signals.empty:
            # Count days with actual signals (not zero)
            signal_days_count = 0
            for date in self.signals.index:
                daily_signals = self.signals.loc[date]
                if (daily_signals != 0).any():
                    signal_days_count += 1
            
            # Calculate months in the data
            if len(self.strategy_returns) > 0:
                start_date = self.strategy_returns.index[0]
                end_date = self.strategy_returns.index[-1]
                months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
                months = max(1, months)  # At least 1 month
                signals_per_month = signal_days_count / months
            else:
                signals_per_month = 0
        else:
            # Fallback: estimate from trading days
            signals_per_month = len(self.strategy_returns) / 12 if len(self.strategy_returns) > 0 else 0
        
        # 3. Position Concentration - for 2-asset strategy, show average position size instead
        if not self.position_sizes.empty:
            # Calculate average position size across all assets
            avg_position_size = self.position_sizes.abs().mean().mean()
            # For 2-asset strategy, show position balance (how evenly distributed)
            position_balance = 1 - abs(self.position_sizes.iloc[:, 0] - self.position_sizes.iloc[:, 1]).mean() / 2
        else:
            avg_position_size = 0
            position_balance = 0
        
        if not self.position_sizes.empty:
            # Calculate daily position changes
            daily_turnover = self.position_sizes.abs().diff().abs().sum(axis=1)
            avg_daily_turnover = daily_turnover.mean()
            max_daily_turnover = daily_turnover.max()
        else:
            # Fallback: estimate from returns volatility
            avg_daily_turnover = self.strategy_returns.std() * 2  # Rough estimate
            max_daily_turnover = self.strategy_returns.abs().max()
        
        return {
            'daily_signal_hit_rate': daily_signal_hit_rate,
            'signals_per_month': signals_per_month,
            'avg_position_size': avg_position_size,
            'position_balance': position_balance,  # Changed from position_concentration
            'avg_daily_turnover': avg_daily_turnover,
            'max_daily_turnover': max_daily_turnover,
            'signal_days': signal_days
        }
    
    def get_trade_summary(self) -> Dict:
        """Get a summary of key trade metrics"""
        all_metrics = self.calculate_all_trade_metrics()
        signal_analysis = self.calculate_signal_analysis()
        
        return {
            'total_trades': all_metrics['trade_summary']['total_trades'],
            'win_rate': all_metrics['trade_summary']['win_rate'],
            'profit_factor': all_metrics['trade_summary']['profit_factor'],
            'avg_trade_pnl': all_metrics['trade_summary']['avg_trade_pnl'],
            'annualized_turnover': all_metrics['turnover_analysis']['annualized_turnover'],
            'capacity_score': all_metrics['capacity_analysis']['capacity_score'],
            'data_source': all_metrics['trade_summary']['data_source'],
            # Add the 5 crucial signal metrics
            'daily_signal_hit_rate': signal_analysis['daily_signal_hit_rate'],
            'signals_per_month': signal_analysis['signals_per_month'],
            'position_balance': signal_analysis['position_balance'],  # Changed from position_concentration
            'avg_daily_turnover_signal': signal_analysis['avg_daily_turnover']
        } 
 