import os
import yaml
from dotenv import load_dotenv
import re
from src.utils.config_model import AppConfig
from pydantic import ValidationError

class ConfigLoader:
    """Класс для загрузки конфигурации из YAML-файла с подстановкой переменных окружения."""

    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()
        self.config_path = config_path

    def _load_config(self):
        """Загружает YAML и подставляет переменные окружения."""
        # Открываем файл — если нет, бросаем FileNotFoundError
        with open(self.config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
        return self._resolve_env_vars(raw_config)

    def _resolve_env_vars(self, config):
        # (ваша рекурсия без изменений)
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(i) for i in config]
        elif isinstance(config, str):
            return re.sub(r"\$\{([^}]+)\}",
                          lambda m: os.getenv(m.group(1), m.group(0)),
                          config)
        else:
            return config

    def get_config(self) -> AppConfig:
        """Возвращает валидированный AppConfig или пробрасывает ошибку."""
        cfg_dict = self._load_config()
        try:
            return AppConfig(**cfg_dict)
        except ValidationError as e:
            print("Ошибка конфигурации:", e)
            raise

    def get_categories(self):
        return self.get_config().categories
