
import logging

def setup_comprehensive_logging():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('backtest.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create specific loggers
    logger = logging.getLogger('backtest')
    logger.setLevel(logging.INFO)
    
    return logger
