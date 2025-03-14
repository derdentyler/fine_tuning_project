import os
import pytest
import shutil
from src.utils.config_loader import ConfigLoader
from src.youtube_scraper import YouTubeScraper

# Путь к тестовому конфигу
CONFIG_PATH = "tests/configs/test_config.yaml"
TEST_DATA = "tests/data/"

# Загружаем конфиг
config = ConfigLoader(config_path=CONFIG_PATH).config

# Получаем параметры из тестового конфига
TEST_SUBTITLE_DIR = config["subtitles_dir"]

# Берём первую попавшуюся ссылку из категорий
TEST_VIDEO_URL = None
for category, videos in config["categories"].items():
    if videos:  # Если список не пустой
        TEST_VIDEO_URL = videos[0]
        break

# Проверяем, что ссылка найдена
if not TEST_VIDEO_URL:
    raise ValueError("❌ В test_config.yaml нет видео для тестирования!")

@pytest.fixture
def scraper():
    """Фикстура для создания YouTubeScraper с тестовыми параметрами."""
    return YouTubeScraper(save_path=TEST_SUBTITLE_DIR)

@pytest.mark.unit
def test_download_subtitles(scraper):
    """
    Тест проверяет реальное скачивание субтитров.
    1. Запускает `download_subtitles()`
    2. Проверяет, что файл появился.
    3. Удаляет файл после теста.
    """
    video_id = TEST_VIDEO_URL.split("v=")[-1]
    expected_file = os.path.join(TEST_SUBTITLE_DIR, f"{video_id}.vtt")

    # Удаляем старый файл, если он остался от предыдущих тестов
    if os.path.exists(expected_file):
        os.remove(expected_file)

    # Запускаем скачивание субтитров
    result = scraper.download_subtitles(TEST_VIDEO_URL)

    # Проверяем, что файл создался
    assert result == expected_file, "Функция вернула неправильный путь!"
    assert os.path.exists(expected_file), "Субтитры не скачались!"

    # Очистка тестовой папки после выполнения (чтобы файлы не копились)
    shutil.rmtree(TEST_DATA)
