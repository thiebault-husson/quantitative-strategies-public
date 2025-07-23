
class DataConsistencyChecker:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
    
    def check_data_periods(self, backtest_results):
        issues = []
        for key, data in backtest_results.items():
            if isinstance(data, (pd.Series, pd.DataFrame)) and len(data) > 0:
                data_start = data.index[0].strftime('%Y-%m-%d')
                data_end = data.index[-1].strftime('%Y-%m-%d')
                if data_start != self.start_date or data_end != self.end_date:
                    issues.append(f"{key}: {data_start} to {data_end}")
        return issues
