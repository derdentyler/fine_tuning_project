import logging
from src.utils.config_loader import ConfigLoader

class LoggerLoader:
    def __init__(self):
        self.config = ConfigLoader().get_categories()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        log_level = self.config.get("log_level", "INFO").upper()
        log_format = self.config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        logger = logging.getLogger("fine_tuning_project")

        # Убираем дублирование логов
        if not logger.hasHandlers():
            logger.setLevel(getattr(logging, log_level, logging.INFO))
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(handler)

        return logger

    def get_logger(self):
        return self.logger
