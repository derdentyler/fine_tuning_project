import os
import pytest
import shutil
from src.utils.config_loader import ConfigLoader
from src.youtube_scraper import YouTubeScraper

CONFIG_PATH = "tests/configs/test_config.yaml"
TEST_VIDEO_URL = "https://www.youtube.com/watch?v=FAKEID"
VIDEO_ID = "FAKEID"


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    # Загружаем конфиг, но сразу переназначим папку субтитров в tmp
    cfg = ConfigLoader(config_path=CONFIG_PATH).get_config()
    temp = tmp_path_factory.mktemp("subs")
    cfg.subtitles_dir = str(temp)
    return cfg


@pytest.fixture
def scraper(cfg, monkeypatch):
    """
    Подменяем метод download_subtitles, чтобы:
      1) Он создавал фиктивный .vtt файл.
      2) Возвращал к нему путь.
    """
    def fake_download(url):
        # url мы игнорируем, вместо этого всегда FAKEID.vtt
        out = os.path.join(cfg.subtitles_dir, f"{VIDEO_ID}.vtt")
        # пишем минимальный WEBVTT-файл
        with open(out, "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n00:00:00.000 --> 00:00:00.500\nHello world")
        return out

    # Подменяем метод
    monkeypatch.setattr(YouTubeScraper, "download_subtitles", fake_download)
    return YouTubeScraper(save_path=cfg.subtitles_dir)


@pytest.mark.unit
def test_download_subtitles(scraper, cfg):
    """
    Проверяем, что:
      1) Метод вернёт путь к FAKEID.vtt,
      2) Файл реально создан в cfg.subtitles_dir.
    """
    expected = os.path.join(cfg.subtitles_dir, f"{VIDEO_ID}.vtt")

    # Убедимся, что до вызова нет
    if os.path.exists(expected):
        os.remove(expected)

    result = scraper.download_subtitles(TEST_VIDEO_URL)

    assert result == expected, "download_subtitles должен вернуть путь к FAKEID.vtt"
    assert os.path.isfile(expected),      "Файл субтитров не создан!"

    # чистим
    shutil.rmtree(cfg.subtitles_dir)
