# =============================================================================
# STRATEGY CONFIGURATION FILE
# =============================================================================
# This file contains all the configuration options for running the trend following strategy

# =============================================================================
# STRATEGY SELECTION
# =============================================================================
# The framework currently supports the trend following strategy
STRATEGY_TYPE = 'trend_following'  # Default strategy

# =============================================================================
# COMMON PARAMETERS
# =============================================================================
# Data parameters
START_DATE = '2014-12-31'
END_DATE = '2025-07-17'
YAHOOFINANCE_FIELD = ['Open', 'Close']  # Support both Open and Close prices
INITIAL_CASH = 100000.0

# Strategy-specific price field requirements
STRATEGY_FIELD_MAPPING = {
    'trend_following': ['Open', 'Close']
}

# =============================================================================
# TREND FOLLOWING STRATEGY PARAMETERS
# =============================================================================
TREND_FOLLOWING_PARAMS = {
    'lookback_period': 200,
    'volatility_lookback': 20,
    'risk_per_trade': 0.01,
    'max_position_size': 0.05
}

# =============================================================================
# STRATEGY DESCRIPTIONS
# =============================================================================
STRATEGY_DESCRIPTIONS = {
    'trend_following': {
        'name': 'Trend Following Strategy (Full S&P 500)',
        'description': 'A momentum-based strategy that follows trends in S&P 500 stocks',
        'tickers': 'S&P 500 components (full index)',
        'benchmark': '^GSPC (S&P 500)'
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_strategy_config():
    """
    Get the configuration for the currently selected strategy.
    
    Returns:
        dict: Configuration dictionary for the selected strategy
    """
    if STRATEGY_TYPE not in STRATEGY_DESCRIPTIONS:
        raise ValueError(f"Unknown strategy type: {STRATEGY_TYPE}. Use 'trend_following'")
    
    config = {
        'strategy_type': STRATEGY_TYPE,
        'description': STRATEGY_DESCRIPTIONS[STRATEGY_TYPE],
        'common_params': {
            'start_date': START_DATE,
            'end_date': END_DATE,
            'yahoofinance_field': STRATEGY_FIELD_MAPPING.get(STRATEGY_TYPE, YAHOOFINANCE_FIELD),
            'initial_cash': INITIAL_CASH
        }
    }
    
    if STRATEGY_TYPE == 'trend_following':
        config['strategy_params'] = TREND_FOLLOWING_PARAMS
    
    return config

def print_strategy_info():
    """
    Print information about the currently selected strategy.
    """
    config = get_strategy_config()
    desc = config['description']
    
    print("=" * 60)
    print(f"📊 STRATEGY: {desc['name']}")
    print("=" * 60)
    print(f"📝 Description: {desc['description']}")
    print(f"📈 Tickers: {desc['tickers']}")
    print(f"🎯 Benchmark: {desc['benchmark']}")
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    print(f"💰 Initial Cash: ${INITIAL_CASH:,.0f}")
    print("=" * 60) 