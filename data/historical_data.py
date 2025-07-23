import pandas as pd
import yfinance as yf
from typing import List, Optional
from datetime import datetime
import time
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_historical_data(
    tickers: List[str], 
    start_date, 
    end_date, 
    interval: str = '1d', 
    fields: Optional[List[str]] = None, 
    max_retries: int = 3, 
    retry_delay: int = 2
    ) -> pd.DataFrame:
    """
    Get historical data for a list of tickers between specified dates.
    
    Args:
        tickers (List[str]): List of ticker symbols to fetch data for.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
        interval (str, optional): Data interval. Options: '1d', '1wk', '1mo'. Defaults to '1d'.
        fields (List[str], optional): List of fields to fetch. Options: 'Open', 'High', 'Low', 'Close', 
                                    'Volume'. Defaults to ['Close'].
        max_retries (int): Maximum number of retries for failed downloads
        retry_delay (int): Delay between retries in seconds
    
    Returns:
        pd.DataFrame: Multi-column DataFrame with dates as index and fields for each ticker as columns.
                     Column names are in format: 'TICKER_FIELD' (e.g., 'AAPL_Close')
    """
    # Input validation
    if not tickers:
        raise ValueError("Tickers list cannot be empty")
    
    # Set default end date if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Set default fields if not provided
    if fields is None:
        fields = ['Close']
    
    # Validate interval
    valid_intervals = ['1d', '1wk', '1mo']
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval. Must be one of {valid_intervals}")
    
    # Validate fields
    valid_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
    invalid_fields = [f for f in fields if f not in valid_fields]
    if invalid_fields:
        raise ValueError(f"Invalid fields: {invalid_fields}. Must be among {valid_fields}")
    
    logger.info(f"Fetching {', '.join(fields)} data for {len(tickers)} tickers from {start_date} to {end_date}")
    logger.info(f"Tickers to process: {tickers}")
    
    # Download data in batches to avoid timeout issues
    all_data = pd.DataFrame()
    batch_size = 5  # Reduced batch size for better reliability
    failed_tickers = []
    
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i+batch_size]
        logger.info(f"Downloading batch {i//batch_size + 1} of {(len(tickers)-1)//batch_size + 1}")
        logger.info(f"Processing tickers: {batch_tickers}")
        
        for retry in range(max_retries):
            try:
                # Download data for the batch
                data = yf.download(
                    tickers=batch_tickers,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    group_by='column',
                    auto_adjust=True,
                    actions=False  # Don't download dividends and splits
                )
                
                # Debug information
                logger.info(f"Downloaded data shape: {data.shape}")
                logger.info(f"Downloaded data columns: {data.columns}")
                
                if data.empty:
                    logger.warning(f"No data received for batch {i//batch_size + 1}")
                    continue
                
                # Process the data
                processed_data = pd.DataFrame()
                
                # Handle single ticker case
                if len(batch_tickers) == 1:
                    ticker = batch_tickers[0]
                    for field in fields:
                        if field in data.columns:
                            processed_data[f"{ticker}_{field}"] = data[field]
                else:
                    # Process multi-ticker case
                    for ticker in batch_tickers:
                        for field in fields:
                            if (field, ticker) in data.columns:
                                processed_data[f"{ticker}_{field}"] = data[(field, ticker)]
                
                # Debug information
                logger.info(f"Processed data shape: {processed_data.shape}")
                logger.info(f"Processed data columns: {processed_data.columns}")
                
                # Merge with existing data
                if not processed_data.empty:
                    if all_data.empty:
                        all_data = processed_data
                    else:
                        all_data = pd.concat([all_data, processed_data], axis=1)
                    logger.info(f"Successfully downloaded batch {i//batch_size + 1}")
                    break  # Success, exit retry loop
                else:
                    logger.warning(f"No data received for batch {i//batch_size + 1}")
                
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"Retry {retry + 1}/{max_retries} for batch starting with {batch_tickers[0]}: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to download batch starting with {batch_tickers[0]}: {str(e)}")
                    failed_tickers.extend(batch_tickers)
    
    if failed_tickers:
        logger.warning(f"\nFailed to download data for {len(failed_tickers)} tickers:")
        logger.warning(failed_tickers)
    
    if all_data.empty:
        raise ValueError("No data was successfully downloaded")
    
    # Forward fill missing values
    all_data = all_data.ffill()
    
    logger.info(f"Successfully loaded data. Shape: {all_data.shape}")
    # Ensure csv directory exists relative to this file
    csv_dir = os.path.join(os.path.dirname(__file__), 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    all_data.to_csv(os.path.join(csv_dir, 'historical_data_sample.csv'))
    return all_data
