import re
import os
from typing import List


class SubtitlePreprocessor:
    """
    Класс для очистки субтитров от временных меток, тегов и дублирования строк.
    """

    def __init__(self, input_path: str, output_path: str):
        self.input_path: str = input_path
        self.output_path: str = output_path

    def clean_text(self, text: str) -> str:
        """Удаляет метки времени, теги и лишние символы."""
        text = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", "", text)
        text = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", text)
        text = re.sub(r"<c>|</c>", "", text)
        text = re.sub(r"WEBVTT|Kind: captions|Language: \w+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"align:start position:\d+%", "", text)
        # удаляем пустые строки и лишние пробелы
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def remove_duplicates(self, text: str) -> str:
        """Убирает повторяющиеся подряд строки."""
        unique_lines: List[str] = []
        for line in text.split("\n"):
            if not unique_lines or line != unique_lines[-1]:
                unique_lines.append(line)
        return "\n".join(unique_lines)

    def process(self) -> None:
        """
        Запускает очистку, удаление дублей и сохраняет результат в output_path.
        """
        if not os.path.exists(self.input_path):
            print(f"Файл не найден: {self.input_path}")
            return

        with open(self.input_path, "r", encoding="utf-8") as f:
            raw_text: str = f.read()

        cleaned = self.clean_text(raw_text)
        final = self.remove_duplicates(cleaned)

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(final)

        print(f"Очистка завершена. Результат сохранен в {self.output_path}")
