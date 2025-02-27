import json
import os
from src.utils.logger_loader import LoggerLoader

class DatasetSaver:
    """
    Класс для сохранения обработанных данных в JSON-файл.
    """

    def __init__(self, save_path: str):
        """
        Инициализация DatasetSaver.

        :param save_path: Путь к файлу, в который будут сохраняться данные.
        """
        self.save_path = save_path
        self.logger = LoggerLoader().get_logger()

    def save(self, data: list):
        """
        Сохраняет данные в JSON-файл.

        :param data: Список словарей, содержащих текст и его категорию.
        """
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)  # Создаем папку, если ее нет
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)  # Сохраняем в JSON с красивым форматированием
            self.logger.info(f"✅ Датасет успешно сохранен в {self.save_path}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка при сохранении датасета: {e}")
