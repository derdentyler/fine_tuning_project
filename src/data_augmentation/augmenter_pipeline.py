import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.utils.logger_loader import LoggerLoader
from src.data_augmentation.bt_augmenter import BackTranslationAugmenter

logger = LoggerLoader().get_logger()


class DataAugmentationPipeline:
    def __init__(
        self,
        input_json: str,
        output_json: str,
        min_examples: int = 500,
        bt_rounds: int = 2,
        bt_beam_size: int = 5
    ) -> None:
        self.input_json: str = input_json
        self.output_json: str = output_json
        self.min_examples: int = min_examples

        # Инициализируем BT аугментатор
        self.bt_augmenter: BackTranslationAugmenter = BackTranslationAugmenter(
            src_lang="ru",
            mid_lang="en",
            rounds=bt_rounds,
            beam_size=bt_beam_size
        )

    def load_data(self) -> List[Dict[str, Any]]:
        """Загружаем данные из JSON."""
        try:
            logger.info(f"Загружаем данные из {self.input_json}...")
            with open(self.input_json, 'r', encoding='utf-8') as file:
                data: List[Dict[str, Any]] = json.load(file)
            logger.info(f"Данные загружены, количество записей: {len(data)}")
            return data
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            raise

    def save_data(self, data: List[Dict[str, Any]]) -> None:
        """Сохраняем данные в JSON."""
        try:
            logger.info(f"Сохраняем данные в {self.output_json}...")
            Path(self.output_json).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_json, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            logger.info(f"Данные успешно сохранены в {self.output_json}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных: {e}")
            raise

    def augment_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Аугментируем данные с помощью back-translation."""
        try:
            logger.info("Начинаем back-translation аугментацию...")
            augmented_data: List[Dict[str, Any]] = []

            for sample in data:
                category: str = sample.get('category', '')  # type: ignore
                text: str = sample.get('text', '')  # type: ignore

                # Оригинальный текст
                augmented_data.append({'category': category, 'text': text})

                # Back-translation
                try:
                    bt_texts: List[str] = self.bt_augmenter.augment(text)
                    for aug_text in bt_texts:
                        augmented_data.append({'category': category, 'text': aug_text})
                except Exception as e:
                    logger.error(f"Ошибка в back-translation аугментации: {e}")
                    logger.error(traceback.format_exc())

            logger.info("Back-translation аугментация завершена.")
            return augmented_data
        except Exception as e:
            logger.error(f"Ошибка при аугментации данных: {e}")
            logger.error(traceback.format_exc())
            raise

    def balance_data(
        self,
        augmented_data: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Проверяем, хватает ли примеров в каждой категории."""
        try:
            logger.info("Проверка на минимальное количество примеров...")

            counts: Dict[str, int] = defaultdict(int)
            for sample in augmented_data:
                counts[sample.get('category', '')] += 1  # type: ignore

            missing: Dict[str, int] = {}
            for cat, cnt in counts.items():
                if cnt < self.min_examples:
                    missing[cat] = self.min_examples - cnt

            if missing:
                for cat, miss in missing.items():
                    logger.warning(f"Для категории '{cat}' не хватает {miss} примеров.")
            else:
                logger.info("Все категории имеют достаточно примеров.")

            return missing
        except Exception as e:
            logger.error(f"Ошибка при проверке баланса данных: {e}")
            raise

    def run(self) -> None:
        """Запуск полного пайплайна."""
        try:
            data: List[Dict[str, Any]] = self.load_data()
            augmented: List[Dict[str, Any]] = self.augment_data(data)
            missing_info: Dict[str, int] = self.balance_data(augmented)

            if missing_info:
                logger.warning("Недостаточно данных для некоторых категорий:")
                for cat, miss in missing_info.items():
                    logger.warning(f"  - {cat}: {miss} примеров не хватает.")

            self.save_data(augmented)
            logger.info("Пайплайн завершен успешно.")

        except Exception as e:
            logger.error(f"Процесс завершился с ошибкой: {e}")


if __name__ == "__main__":
    pipeline = DataAugmentationPipeline(
        input_json='data/processed/val.json',
        output_json='data/processed/dataset_augmented_balanced.json',
        min_examples=500,
        bt_rounds=2,
        bt_beam_size=5
    )
    pipeline.run()