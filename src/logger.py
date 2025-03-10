import logging
import os
from datetime import datetime

# Define log directory and ensure it exists
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)

# Define log file name based on the current date (ensures all logs for the day go into one file)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y')}.log"
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
