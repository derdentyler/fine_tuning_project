import os
import shutil
import pytest
from src.youtube_dataset_builder import YouTubeDatasetBuilder
from src.utils.config_loader import ConfigLoader

# Пути для тестов
TEST_CONFIG_PATH = "tests/configs/test_config.yaml"
TEST_OUTPUT_DIR = "tests/data/processed/"
TEST_SUBTITLE_DIR = "tests/data/raw/"

@pytest.mark.integration
def test_youtube_dataset_builder():
    """
    Интеграционный тест: проверяет полный цикл работы YouTubeDatasetBuilder.
    Скачиваются субтитры, обрабатываются и сохраняются в JSON.
    """
    # Убедиться, что чистая папка вывода
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # Загружаем валидированный конфиг
    cfg = ConfigLoader(config_path=TEST_CONFIG_PATH).get_config()

    # Создаем билдер и задаем тестовые пути
    builder = YouTubeDatasetBuilder(cfg)
    builder.output_dir = TEST_OUTPUT_DIR
    builder.saver.output_path = os.path.join(TEST_OUTPUT_DIR, "dataset.json")

    # Запускаем сбор датасета
    builder.build_dataset()

    # Проверяем JSON
    dataset_path = os.path.join(TEST_OUTPUT_DIR, "dataset.json")
    assert os.path.isfile(dataset_path), "❌ Датасет не был сохранен!"

    # Проверяем наличие хотя бы одного обработанного файла
    cleaned = [f for f in os.listdir(TEST_SUBTITLE_DIR) if f.endswith("_cleaned.txt")]
    assert cleaned, "❌ Нет обработанных субтитров!"

    # Финальная очистка только test data
    shutil.rmtree(TEST_OUTPUT_DIR)
