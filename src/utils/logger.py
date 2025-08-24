import logging
import os
import sys
from datetime import datetime


def setup_logging(log_dir: str,
                  log_filename: str,
                  level: int = logging.INFO,
                  console_output: bool = True) -> logging.Logger:
    """
    Logging configuration for the application.
    Logs will be written to a file and optionally to the console

    Args:
        log_dir (str):  Directory where log files will be saved.
        log_filename (str): Name of the log file.
        level (int): Logging level (logging.INFO, logging.DEBUG, etc.).
        console_output (bool): If True, logs are written to the console.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Verify log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create log file path
    log_filepath = os.path.join(log_dir, log_filename)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # Define consistent format for log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s')

    # Add FileHandler to write logs to a file
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add StreamHandler to output logs to console
    if console_output:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.info(f"Logging configured. Logs will be save to: {log_filepath}")
    return logger


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    sys.path.insert(0, project_root)

    test_log_dir = os.path.join(project_root, 'results', 'test_logs')
    test_log_filename = f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = setup_logging(test_log_dir, test_log_filename, logging.DEBUG)
    logger.debug("This is a debug message.")
    logger.info("Tis is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")
    print(f"\nCheck '{test_log_dir}/{test_log_filename}' for log output.")
