import os
import pytest
import shutil
from src.utils.config_loader import ConfigLoader
from src.youtube_scraper import YouTubeScraper

# Путь к тестовому конфигу
CONFIG_PATH = "tests/configs/test_config.yaml"

@pytest.fixture(scope="module")
def cfg():
    return ConfigLoader(config_path=CONFIG_PATH).get_config()

@pytest.fixture
def scraper(tmp_path, cfg):
    """
    Фикстура для YouTubeScraper.
    Создаём чистый tmp_path и передаём в него конфиг.subtitles_dir
    """
    # переопределяем путь для субтитров в tmp
    temp_dir = tmp_path / "subs"
    temp_dir.mkdir()
    cfg.subtitles_dir = str(temp_dir)
    return YouTubeScraper(save_path=str(temp_dir))

@pytest.mark.unit
def test_download_subtitles(scraper, cfg):
    """
    Тестирует download_subtitles:
    1. Удаляем старый файл.
    2. Скачиваем субтитры.
    3. Проверяем, что файл появился.
    """
    # Выбираем первую ссылку
    TEST_VIDEO_URL = next(iter(cfg.categories.values()))[0]
    video_id = TEST_VIDEO_URL.split("v=")[-1]
    expected_file = os.path.join(cfg.subtitles_dir, f"{video_id}.vtt")

    # Гарантируем, что файла нет
    if os.path.exists(expected_file):
        os.remove(expected_file)

    # Запускаем скачивание
    result = scraper.download_subtitles(TEST_VIDEO_URL)

    assert result == expected_file, "Функция вернула неправильный путь!"
    assert os.path.isfile(expected_file), "Субтитры не скачались!"

    # Чистим только созданные файлы
    shutil.rmtree(cfg.subtitles_dir)
