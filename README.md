# Quantitative Strategies Backtesting Framework

A professional-grade quantitative trading strategy backtesting and reporting framework designed for hedge fund-level analysis.

## 🚀 Features

- **Momentum Strategy Support
- **Institutional-Grade Reporting**: Comprehensive HTML reports with interactive charts
- **Modular Architecture**: Clean, maintainable codebase structure
- **Professional Logging**: Detailed execution logs and error tracking
- **Scalable Design**: Easy to add new strategies and extend functionality

## 📁 Project Structure

```
quantitative-strategies/
├── strategies/              # Strategy implementations
├── backtest/               # Backtesting engine
├── data/                   # Data loading and processing
├── reporting/              # Report generation
│   ├── html_generator.py   # Unified HTML report generator
│   ├── templates/          # HTML templates
│   └── performance_metrics/ # Performance calculations
├── utils/                  # Utilities and configuration
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── reports/                # Generated reports (auto-created)
├── logs/                   # Execution logs (auto-created)
└── main.py                 # Entry point
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd quantitative-strategies
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

1. **Configure strategy** (optional - defaults to trend following)
   ```bash
   python utils/strategy_switcher.py trend_following
   ```

2. **Run backtest**
   ```bash
   python main.py
   ```

3. **View results**
   - Check the `reports/` directory for generated HTML reports
   - Check the `logs/` directory for execution logs

## 📊 Available Strategies

### Trend Following Strategy
- **Description**: Momentum-based strategy following S&P 500 trends
- **Assets**: S&P 500 components
- **Benchmark**: S&P 500 (^GSPC)

## ⚙️ Configuration

Edit `utils/config.py` to modify:
- Strategy parameters
- Date ranges
- Risk management settings
- Data sources

## 📈 Report Features

### Institutional-Grade HTML Reports Include:
- **Executive Summary**: High-level performance overview
- **Key Metrics**: Returns, volatility, Sharpe ratio, drawdown, VaR
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Detailed Tables**: Comprehensive performance analysis
- **Strategy Summary**: Parameter documentation
- **Risk Analysis**: Exposure and attribution analysis

### Generated Assets:
- Interactive HTML reports
- Performance charts (PNG)
- CSV data exports
- Execution logs

## 🧪 Testing

Run tests:
```bash
python -m pytest tests/
```

## 📝 Logging

The framework provides comprehensive logging:
- **File logs**: Detailed logs saved to `logs/` directory
- **Console output**: Real-time progress and status updates
- **Error tracking**: Full error traces and debugging information

## 🔧 Development

### Adding New Strategies

1. Create new strategy class in `strategies/`
2. Inherit from `BaseStrategy`
3. Implement required methods:
   - `generate_signals()`
   - `size_positions()`
   - `calculate_returns()`
4. Add configuration in `utils/config.py`
5. Update `main.py` to support new strategy

### Extending Reports

1. Modify `reporting/html_generator.py`
2. Update templates in `reporting/templates/`
3. Add new performance metrics in `reporting/performance_metrics/`

## 📚 Documentation

- **Strategy Documentation**: See `docs/README_STRATEGIES.md`
- **API Reference**: Check individual module docstrings
- **Examples**: Review test files for usage examples

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Ensure all tests pass
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Use at your own risk. 