import pandas as pd
import os

class BacktestEngine:
    def __init__(self, strategy, prices, initial_cash=100000.0):
        self.strategy = strategy
        self.prices = prices
        self.initial_cash = initial_cash

    def run(self):
        # Generate signals and position sizes
        signals = self.strategy.generate_signals(self.prices)
        position_sizes = self.strategy.size_positions(self.prices, signals)
        # Calculate returns
        returns = self.strategy.calculate_returns(self.prices, signals, position_sizes)
        # Calculate portfolio value
        portfolio_values = self.initial_cash * (1 + returns).cumprod()

        # Save outputs as CSV in data/csv
        csv_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        signals.to_csv(os.path.join(csv_dir, 'strategy_signals.csv'))
        position_sizes.to_csv(os.path.join(csv_dir, 'strategy_position_sizes.csv'))
        returns.to_csv(os.path.join(csv_dir, 'strategy_returns.csv'))
        pd.DataFrame(portfolio_values).to_csv(os.path.join(csv_dir, 'strategy_portfolio_values.csv'))

        return {
            'signals': signals,
            'position_sizes': position_sizes,
            'returns': returns,
            'portfolio_values': portfolio_values,
            'prices': self.prices,
            'initial_capital': self.initial_cash
        }
