#!/usr/bin/env python3
"""
Strategy Switcher - Easy way to switch between strategies
"""

import sys
import os

def switch_strategy(strategy_type):
    """
    Switch to a different strategy by updating config.py
    
    Args:
        strategy_type (str): 'trend_following'
    """
    config_file = 'utils/config.py'
    
    if not os.path.exists(config_file):
        print(f"❌ Error: {config_file} not found!")
        return False
    
    # Read the current config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Update the strategy type
    import re
    pattern = r"STRATEGY_TYPE = ['\"]([^'\"]+)['\"]"
    replacement = f"STRATEGY_TYPE = '{strategy_type}'"
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        
        # Write the updated config
        with open(config_file, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Successfully switched to {strategy_type} strategy!")
        return True
    else:
        print("❌ Error: Could not find STRATEGY_TYPE in config.py")
        return False

def show_available_strategies():
    """Show available strategies"""
    print("📊 Available Strategies:")
    print("  1. trend_following - Momentum-based S&P 500 strategy")
    print()
    print("Usage: python switch_strategy.py <strategy_name>")
    print("Example: python switch_strategy.py trend_following")

def main():
    if len(sys.argv) != 2:
        show_available_strategies()
        return
    
    strategy_type = sys.argv[1].lower()
    
    if strategy_type not in ['trend_following']:
        print(f"❌ Error: Unknown strategy '{strategy_type}'")
        show_available_strategies()
        return
    
    if switch_strategy(strategy_type):
        print(f"🚀 You can now run: python main.py")
        print(f"📊 This will execute the {strategy_type} strategy")

if __name__ == "__main__":
    main() 