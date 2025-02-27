import re
import os


class SubtitlePreprocessor:
    """
    Класс для очистки субтитров от временных меток, тегов и дублирования строк.
    """

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def clean_text(self, text: str) -> str:
        """
        Удаляет временные метки, теги и лишние символы.

        :param text: Исходный текст субтитров.
        :return: Очищенный текст.
        """
        # Удаляем временные метки (форматы типа "00:00:16.139 --> 00:00:17.990")
        text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", "", text)

        # Удаляем одиночные временные метки типа "<00:00:16.560>"
        text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text)

        # Убираем теги <c> (форматы типа "<c>текст</c>")
        text = re.sub(r"<c>|</c>", "", text)

        # Убираем служебные заголовки WEBVTT, Kind, Language
        text = re.sub(r"WEBVTT|Kind: captions|Language: \w+", "", text, flags=re.IGNORECASE)

        # Удаляем строки с метками типа "align:start position:0%"
        text = re.sub(r"align:start position:\d+%", "", text)

        # Очищаем пустые строки и лишние пробелы
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

        return text

    def remove_duplicates(self, text: str) -> str:
        """
        Убирает повторяющиеся подряд строки.

        :param text: Очищенный текст субтитров.
        :return: Текст без дубликатов.
        """
        lines = text.split("\n")
        unique_lines = []

        for line in lines:
            if not unique_lines or line != unique_lines[-1]:  # Добавляем, если строка не повторяется
                unique_lines.append(line)

        return "\n".join(unique_lines)

    def process(self):
        """
        Запускает очистку, удаление дублей и сохраняет результат.
        """
        if not os.path.exists(self.input_path):
            print(f"❌ Файл не найден: {self.input_path}")
            return

        with open(self.input_path, "r", encoding="utf-8") as file:
            raw_text = file.read()

        cleaned_text = self.clean_text(raw_text)
        final_text = self.remove_duplicates(cleaned_text)

        with open(self.output_path, "w", encoding="utf-8") as file:
            file.write(final_text)

        print(f"✅ Очистка завершена. Результат сохранен в {self.output_path}")
