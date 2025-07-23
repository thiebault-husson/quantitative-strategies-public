
class MetricValidator:
    def __init__(self):
        self.tolerance_levels = {
            'total_return': 0.001,
            'annualized_return': 0.001,
            'volatility': 0.001,
            'sharpe_ratio': 0.01,
            'max_drawdown': 0.001,
            'beta': 0.001,
            'alpha': 0.001,
            'var_95': 0.001,
            'win_rate': 0.001,
            'total_trades': 1,
            'annualized_turnover': 0.1
        }
    
    def validate_metrics(self, calculated_metrics, expected_metrics):
        results = {}
        for metric, calculated in calculated_metrics.items():
            if metric in expected_metrics:
                expected = expected_metrics[metric]
                tolerance = self.tolerance_levels.get(metric, 0.001)
                difference = abs(calculated - expected)
                results[metric] = {
                    'calculated': calculated,
                    'expected': expected,
                    'difference': difference,
                    'tolerance': tolerance,
                    'pass': difference <= tolerance
                }
        return results
