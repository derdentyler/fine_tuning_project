import pandas as pd
from src.utils.logger_loader import LoggerLoader


class DatasetValidator:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.logger = LoggerLoader().get_logger()

    def load_dataset(self) -> pd.DataFrame:
        """Загружает датасет и проверяет его наличие."""
        try:
            df = pd.read_csv(self.dataset_path)
            self.logger.info(f"Датасет загружен: {self.dataset_path}")
            return df
        except FileNotFoundError:
            self.logger.error(f"Файл не найден: {self.dataset_path}")
            raise
        except Exception as e:
            self.logger.error(f"Ошибка загрузки датасета: {e}")
            raise

    def check_missing_values(self, df: pd.DataFrame):
        """Проверяет пропущенные значения."""
        missing = df.isnull().sum().sum()
        if missing > 0:
            self.logger.warning(f"Обнаружено {missing} пропущенных значений")
        else:
            self.logger.info("Пропущенных значений нет")

    def check_duplicates(self, df: pd.DataFrame):
        """Проверяет дубликаты."""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"Обнаружено {duplicates} дубликатов")
        else:
            self.logger.info("Дубликатов нет")

    def check_label_consistency(self, df: pd.DataFrame):
        """Проверяет соответствие меток и текстов."""
        if 'text' not in df.columns or 'label' not in df.columns:
            self.logger.error("Отсутствуют необходимые колонки: 'text' или 'label'")
            raise ValueError("Некорректный формат датасета")

        unique_labels = df['label'].unique()
        self.logger.info(f"Найдено {len(unique_labels)} уникальных меток: {unique_labels}")

    def validate(self):
        """Запускает все проверки."""
        df = self.load_dataset()
        self.check_missing_values(df)
        self.check_duplicates(df)
        self.check_label_consistency(df)
        self.logger.info("Проверка датасета завершена")
        return df
