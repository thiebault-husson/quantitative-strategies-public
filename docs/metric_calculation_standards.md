# Metric Calculation Standards

## Total Return
- Formula: (Final Value / Initial Value) - 1
- Example: (251,950 / 100,000) - 1 = 1.5195 = 151.95%

## Sharpe Ratio
- Formula: (Mean Excess Return / Standard Deviation) * √252
- Risk-free rate: 2% annualized
- Example: (0.0005 / 0.015) * √252 = 0.75

## Win Rate
- Formula: (Winning Trades / Total Trades) * 100
- Example: (781 / 1559) * 100 = 50.1%

## Volatility
- Formula: Standard Deviation * √252
- Example: 0.015 * √252 = 0.238 = 23.8%

## Max Drawdown
- Formula: (Peak Value - Trough Value) / Peak Value
- Example: (1000 - 800) / 1000 = 0.20 = 20%

## Beta
- Formula: Covariance(Strategy Returns, Benchmark Returns) / Variance(Benchmark Returns)
- Example: 0.0001 / 0.0002 = 0.5

## Alpha
- Formula: Strategy Return - (Risk-free Rate + Beta * (Benchmark Return - Risk-free Rate))
- Example: 0.15 - (0.02 + 0.5 * (0.12 - 0.02)) = 0.08 = 8%
