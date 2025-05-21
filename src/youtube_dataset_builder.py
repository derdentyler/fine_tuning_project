import os
from src.youtube_scraper import YouTubeScraper
from src.subtitle_preprocessor import SubtitlePreprocessor
from src.dataset_saver import DatasetSaver
from src.utils.logger_loader import LoggerLoader
from src.utils.config_model import AppConfig  # Pydantic‑модель


class YouTubeDatasetBuilder:
    """
    Класс для автоматического сбора датасета с YouTube на основе ссылок
    и меток из Pydantic‑конфига AppConfig.
    """

    def __init__(self, cfg: AppConfig):
        """
        :param cfg: экземпляр AppConfig с загруженными и валидированными настройками.
        """
        self.cfg = cfg
        self.logger = LoggerLoader().get_logger()

        # Проверка, что конфиг действительно есть
        if not self.cfg:
            raise ValueError("❌ Ошибка: Конфиг не загружен.")

        # Директория для загрузки субтитров
        self.subtitle_dir = cfg.subtitles_dir
        os.makedirs(self.subtitle_dir, exist_ok=True)
        self.scraper = YouTubeScraper(self.subtitle_dir)

        # Директория для выходных данных
        self.output_dir = cfg.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.saver = DatasetSaver(os.path.join(self.output_dir, "dataset.json"))

        # Статистика
        self.total_videos = 0
        self.downloaded_subtitles = 0
        self.skipped_videos = []

    def build_dataset(self):
        """
        Собирает датасет: скачивает субтитры, обрабатывает их и сохраняет в JSON.
        """
        dataset = []
        for category, urls in self.cfg.categories.items():
            for url in urls:
                self.total_videos += 1
                self.logger.info(f"🔍 Обрабатываю {url} (категория: {category})")
                try:
                    # 1) Скачиваем субтитры
                    path = self.scraper.download_subtitles(url)
                    if not path:
                        self.logger.warning(f"⚠️ Пропущено (нет сабов): {url}")
                        self.skipped_videos.append({"url": url, "reason": "Нет субтитров"})
                        continue
                    self.downloaded_subtitles += 1

                    # 2) Предобработка
                    cleaned = path.replace(".vtt", "_cleaned.txt")
                    SubtitlePreprocessor(path, cleaned).process()

                    # 3) Читаем и валидируем текст
                    try:
                        text = open(cleaned, encoding="utf-8").read()
                    except Exception as e:
                        self.logger.error(f"Ошибка чтения {cleaned}: {e}")
                        self.skipped_videos.append({"url": url, "reason": "Чтение сабов"})
                        continue
                    if not text:
                        self.logger.warning(f"⚠️ Пустой текст после обработки: {url}")
                        self.skipped_videos.append({"url": url, "reason": "Пустой текст"})
                        continue

                    dataset.append({"category": category, "text": text})

                except Exception as e:
                    self.logger.exception(f"Ошибка обработки {url}: {e}")
                    self.skipped_videos.append({"url": url, "reason": "Общая ошибка"})
                    continue

        # Сохраняем результат
        if dataset:
            try:
                self.saver.save(dataset)
                self.logger.info(f"✅ Датасет сохранён в {self.output_dir}")
            except Exception as e:
                self.logger.error(f"Ошибка сохранения датасета: {e}")
        else:
            self.logger.warning("⚠️ Датасет пуст")

        # Логгируем статистику
        self.logger.info(f"📊 Загружено {self.downloaded_subtitles}/{self.total_videos} сабов")
        if self.skipped_videos:
            self.logger.warning("⚠️ Необработанные видео:")
            for it in self.skipped_videos:
                self.logger.warning(f"   ❌ {it['url']} — {it['reason']}")
        print(f"\n📊 Статистика: {self.downloaded_subtitles}/{self.total_videos} сабов")
        if self.skipped_videos:
            print("⚠️ Необработанные видео:")
            for it in self.skipped_videos:
                print(f"   ❌ {it['url']} — {it['reason']}")