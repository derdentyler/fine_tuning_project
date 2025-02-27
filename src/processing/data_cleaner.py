import pandas as pd
from src.utils.logger_loader import LoggerLoader


class DataCleaner:
    """
    Класс для очистки и предобработки датасета перед подачей в text_preprocessor.
    """

    def __init__(self):
        self.logger = LoggerLoader().get_logger()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Загружает датасет из CSV-файла.

        :param file_path: Путь к CSV-файлу.
        :return: DataFrame с загруженными данными.
        """
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"✅ Данные успешно загружены из {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"❌ Ошибка при загрузке данных: {e}")
            return pd.DataFrame()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Очищает датасет: удаляет дубликаты, пропущенные значения и балансирует классы.

        :param df: Исходный DataFrame.
        :return: Очищенный DataFrame.
        """
        self.logger.info("🛠️ Начало очистки данных...")

        # Удаляем строки с пропущенными значениями
        df = df.dropna()

        # Удаляем дубликаты
        df = df.drop_duplicates()

        self.logger.info("✅ Очистка данных завершена")
        return df

    def save_clean_data(self, df: pd.DataFrame, save_path: str):
        """
        Сохраняет очищенный датасет в CSV-файл.

        :param df: Очищенный DataFrame.
        :param save_path: Путь для сохранения CSV-файла.
        """
        try:
            df.to_csv(save_path, index=False)
            self.logger.info(f"✅ Очищенные данные сохранены в {save_path}")
        except Exception as e:
            self.logger.error(f"❌ Ошибка при сохранении очищенных данных: {e}")
