import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    """
    Formats log records as a JSON string.
    Handles the 'extra' parameter to include custom fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        # These are the standard attributes from a LogRecord
        standard_keys = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
            "taskName",
        }

        # Base log object with standard information
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger_name": record.name,
        }

        # Add any fields passed in the 'extra' parameter
        for key, value in record.__dict__.items():
            if key not in standard_keys:
                log_object[key] = value

        return json.dumps(log_object, ensure_ascii=False)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger with a JSON formatter.
    """
    logger = logging.getLogger(name)

    # Prevents adding handlers multiple times in interactive environments
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
