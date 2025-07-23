import pandas as pd
import numpy as np
from .base import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, lookback_period, volatility_lookback, risk_per_trade, max_position_size):
        self.lookback_period = lookback_period
        self.volatility_lookback = volatility_lookback
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size

    def get_required_price_fields(self) -> list:
        """Return required price fields for this strategy."""
        return ['Open', 'Close']

    def _extract_ticker_base(self, column_name: str) -> str:
        """Extract base ticker name from column name."""
        return column_name.replace('_Open', '').replace('_Close', '')

    def _get_close_column(self, ticker_base: str, prices: pd.DataFrame) -> str:
        """Get close column name for a ticker."""
        close_column = f'{ticker_base}_Close'
        if close_column not in prices.columns:
            raise ValueError(f"Close column {close_column} not found in price data")
        return close_column

    def _get_open_column(self, ticker_base: str, prices: pd.DataFrame) -> str:
        """Get open column name for a ticker."""
        open_column = f'{ticker_base}_Open'
        if open_column not in prices.columns:
            raise ValueError(f"Open column {open_column} not found in price data")
        return open_column

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using close prices (close-to-close logic).
        Signals are generated at close and executed next day at open.
        """
        signals = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        for ticker in prices.columns:
            ticker_base = self._extract_ticker_base(ticker)
            close_column = self._get_close_column(ticker_base, prices)
            
            # Calculate moving average using close prices
            ma = prices[close_column].rolling(window=self.lookback_period).mean()
            
            # Compare yesterday's close with yesterday's moving average
            # Signal generated at close, executed next day at open
            signals[ticker] = np.where(prices[close_column].shift(1) > ma.shift(1), 1, 0)
        
        return signals

    def size_positions(self, prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Size positions using close price volatility.
        """
        position_sizes = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        for ticker in prices.columns:
            ticker_base = self._extract_ticker_base(ticker)
            close_column = self._get_close_column(ticker_base, prices)
            
            # Calculate volatility using close prices
            returns = prices[close_column].pct_change()
            volatility = returns.rolling(window=self.volatility_lookback).std()
            
            position_sizes[ticker] = self.risk_per_trade / volatility
            position_sizes[ticker] = position_sizes[ticker].clip(0, self.max_position_size)
        
        # Normalize row if the sum exceeds 100%
        row_sums = position_sizes.sum(axis=1)
        position_sizes = position_sizes.div(row_sums, axis=0).where(row_sums > 1, position_sizes)
        return position_sizes * signals

    def calculate_returns(self, prices: pd.DataFrame, signals: pd.DataFrame, position_sizes: pd.DataFrame) -> pd.Series:
        """
        Calculate open-to-close returns for realistic performance measurement.
        """
        # Calculate open-to-close returns for each ticker
        returns = pd.DataFrame(index=prices.index, columns=prices.columns)
        
        for ticker in prices.columns:
            ticker_base = self._extract_ticker_base(ticker)
            open_column = self._get_open_column(ticker_base, prices)
            close_column = self._get_close_column(ticker_base, prices)
            
            # Open-to-Close returns: (Close - Open) / Open
            returns[ticker] = (prices[close_column] - prices[open_column]) / prices[open_column]
        
        # Apply signals and position sizes (shifted by 1 to avoid look-ahead bias)
        strategy_returns = returns * signals.shift(1) * position_sizes.shift(1)
        portfolio_returns = strategy_returns.sum(axis=1)
        
        return portfolio_returns