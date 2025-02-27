import yaml


class ConfigLoader:
    """Класс для загрузки конфигурации из YAML-файла."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        """Загружает и парсит YAML-файл."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️ Файл {self.config_path} не найден!")
            return {}
        except yaml.YAMLError as e:
            print(f"⚠️ Ошибка загрузки YAML: {e}")
            return {}

    def get_categories(self):
        """Возвращает словарь с категориями и ссылками."""
        return self.config.get("categories", {})
