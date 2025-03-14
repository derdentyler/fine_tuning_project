import os
import yaml
from src.youtube_scraper import YouTubeScraper
from src.subtitle_preprocessor import SubtitlePreprocessor
from src.dataset_saver import DatasetSaver
from src.utils.logger_loader import LoggerLoader
from src.utils.config_loader import ConfigLoader


class YouTubeDatasetBuilder:
    """
    Класс для автоматического сбора датасета с YouTube на основе ссылок и меток из YAML-конфига.
    """

    def __init__(self, config_path: str):
        """
        :param config_path: Путь к YAML-конфигу с видео и метками.
        """
        self.config = ConfigLoader(config_path).config
        self.logger = LoggerLoader().get_logger()


        if not self.config:
            raise ValueError("❌ Ошибка: Конфиг не загружен.")

        self.subtitle_dir = self.config.get("subtitles_dir", "data/raw/")
        self.scraper = YouTubeScraper(self.subtitle_dir)

        self.output_dir = self.config.get("output_dir", "data/processed/")
        os.makedirs(self.output_dir, exist_ok=True)
        self.saver = DatasetSaver(os.path.join(self.output_dir, "dataset.json"))

        # Счетчики
        self.total_videos = 0
        self.downloaded_subtitles = 0
        self.skipped_videos = []  # Список пропущенных видео


    def build_dataset(self):
        """
        Собирает датасет: скачивает субтитры, обрабатывает их и сохраняет в JSON.
        """
        dataset = []
        categories = self.config.get("categories", {})

        for category, video_urls in categories.items():
            for url in video_urls:
                self.total_videos += 1  # Увеличиваем общее количество видео
                self.logger.info(f"🔍 Обрабатываю видео {url} (категория: {category})")

                try:
                    subtitle_path = self.scraper.download_subtitles(url)
                    if not subtitle_path:
                        self.logger.warning(f"⚠️ Пропущено: нет субтитров для {url}")
                        self.skipped_videos.append({"url": url, "reason": "Нет субтитров"})
                        continue

                    self.downloaded_subtitles += 1  # Увеличиваем количество загруженных субтитров

                    output_subtitle_path = subtitle_path.replace(".vtt", "_cleaned.txt")
                    preprocessor = SubtitlePreprocessor(subtitle_path, output_subtitle_path)
                    preprocessor.process()

                    # Читаем обработанный текст
                    try:
                        with open(output_subtitle_path, "r", encoding="utf-8") as f:
                            processed_text = f.read()
                    except Exception as e:
                        self.logger.error(f"❌ Ошибка чтения обработанного файла {output_subtitle_path}: {e}")
                        self.skipped_videos.append({"url": url, "reason": "Ошибка чтения обработанного файла"})
                        continue

                    if not processed_text:
                        self.logger.warning(f"⚠️ Пропущено: ошибка обработки субтитров {url}")
                        self.skipped_videos.append({"url": url, "reason": "Ошибка обработки субтитров"})
                        continue

                    dataset.append({"category": category, "text": processed_text})

                except Exception as e:
                    self.logger.exception(f"❌ Ошибка обработки видео {url}: {e}")
                    self.skipped_videos.append({"url": url, "reason": "Ошибка при обработке"})
                    continue  # Переходим к следующему видео, не прерывая процесс

        # Сохранение датасета с обработкой ошибок
        if dataset:
            try:
                self.saver.save(dataset)
                self.logger.info(f"✅ Датасет успешно собран и сохранен в {self.output_dir}")
            except Exception as e:
                self.logger.error(f"❌ Ошибка сохранения датасета: {e}")
        else:
            self.logger.warning("⚠️ Датасет пуст: не найдено обработанных данных")

        # Вывод статистики
        self.logger.info(f"📊 Загружено субтитров: {self.downloaded_subtitles} из {self.total_videos} видео")

        # Вывод списка необработанных видео
        if self.skipped_videos:
            self.logger.warning("⚠️ Необработанные видео:")
            for item in self.skipped_videos:
                self.logger.warning(f"   ❌ {item['url']} — {item['reason']}")

        print("\n📊 Статистика загрузки:")
        print(f"   ✅ Загружено субтитров: {self.downloaded_subtitles} из {self.total_videos}")
        if self.skipped_videos:
            print("\n⚠️ Список необработанных видео:")
            for item in self.skipped_videos:
                print(f"   ❌ {item['url']} — {item['reason']}")


if __name__ == "__main__":
    CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))

    builder = YouTubeDatasetBuilder(CONFIG_PATH)
    builder.build_dataset()
