import os
import yaml
from dotenv import load_dotenv
import re


class ConfigLoader:
    """Класс для загрузки конфигурации из YAML-файла с подстановкой переменных окружения."""

    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()  # Загружаем .env
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """Загружает YAML и подставляет переменные окружения."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}

            return self._resolve_env_vars(raw_config)

        except FileNotFoundError:
            print(f"⚠️ Файл {self.config_path} не найден!")
            return {}
        except yaml.YAMLError as e:
            print(f"⚠️ Ошибка загрузки YAML: {e}")
            return {}

    def _resolve_env_vars(self, config):
        """Рекурсивно подставляет переменные окружения в конфиг."""
        if isinstance(config, dict):
            return {key: self._resolve_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            return re.sub(r"\$\{([^}]+)\}", lambda match: os.getenv(match.group(1), match.group(0)), config)
        else:
            return config

    def get_categories(self):
        """Возвращает словарь с категориями и ссылками."""
        return self.config.get("categories", {})

