import os
import shutil
import pytest
from src.youtube_dataset_builder import YouTubeDatasetBuilder
from src.utils.config_loader import ConfigLoader

# Пути для тестов
TEST_CONFIG_PATH = "tests/configs/test_config.yaml"
TEST_OUTPUT_DIR = "tests/data/processed/"
TEST_SUBTITLE_DIR = "tests/data/raw/"
TEST_DATA = "tests/data/"

config = ConfigLoader(TEST_CONFIG_PATH).config


@pytest.mark.integration
def test_youtube_dataset_builder():
    """
    Интеграционный тест: проверяет полный цикл работы YouTubeDatasetBuilder.
    Должны скачаться субтитры, обработаться и сохраниться в JSON.
    """
    # Удаляем старые тестовые данные перед запуском
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    # Создаем объект билдера с тестовым конфигом
    builder = YouTubeDatasetBuilder(config)

    # Переназначаем директорию для тестов (чтобы не лезло в основную папку)
    builder.output_dir = TEST_OUTPUT_DIR
    builder.saver.output_path = os.path.join(TEST_OUTPUT_DIR, "dataset.json")

    # Запускаем сбор датасета
    builder.build_dataset()

    # Проверяем, что JSON-файл с результатами создан
    dataset_path = os.path.join(TEST_OUTPUT_DIR, "dataset.json")
    assert os.path.exists(dataset_path), "❌ Датасет не был сохранен!"

    # Проверяем, что хотя бы один обработанный файл субтитров есть
    processed_files = [f for f in os.listdir(TEST_SUBTITLE_DIR) if f.endswith("_cleaned.txt")]
    assert len(processed_files) > 0, "❌ Нет обработанных субтитров!"

    # Очистка тестовой папки после выполнения (чтобы файлы не копились)
    shutil.rmtree(TEST_DATA)
