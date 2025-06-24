import json
import logging
import sys
from typing import Optional


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
            "logger_name": record.name,
            "message": record.getMessage(),
        }

        # Add any fields passed in the 'extra' parameter
        for key, value in record.__dict__.items():
            if key not in standard_keys:
                log_object[key] = value

        return json.dumps(log_object, ensure_ascii=False)


def setup_logger(
    level: int = logging.INFO,
    logfile_path: Optional[str] = None,
):
    """
    Sets up a logger with a JSON formatter.

    Args:
        name (str): The name of the logger.
        level (int, optional): The logging level. Defaults to logging.INFO.
        logfile_path (str, optional): The path to the log file. If None,
          no file handler will be added.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Prevents adding handlers multiple times in interactive environments
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = JsonFormatter()

    # ターミナル出力用のハンドラ
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root_logger.addHandler(stream_handler)

    if logfile_path:
        # ログディレクトリが存在しない場合は作成
        import os

        log_dir = os.path.dirname(logfile_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(logfile_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

        logging.info("Logging is also directed to the file: %s", logfile_path)
