from abc import ABC, abstractmethod
import pandas as pd

class BaseStrategy(ABC):
    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def size_positions(self, prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def calculate_returns(self, prices: pd.DataFrame, signals: pd.DataFrame, position_sizes: pd.DataFrame) -> pd.Series:
        pass
    
    def get_required_price_fields(self) -> list:
        """
        Return list of required price fields for this strategy.
        Default implementation for backward compatibility.
        """
        return ['Open']  # Default to Open for existing strategies
