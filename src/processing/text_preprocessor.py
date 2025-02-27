import re
import string
from src.utils import nltk_loader
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    def __init__(self, language='russian'):
        self.stop_words = set(stopwords.words(language))

    def clean_text(self, text: str) -> str:
        """Удаляет пунктуацию и приводит текст к нижнему регистру."""
        text = text.lower()
        text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)  # Удаляем пунктуацию
        text = re.sub(r'\s+', ' ', text).strip()  # Убираем лишние пробелы
        return text

    def remove_stopwords(self, words: list) -> list:
        """Удаляет стоп-слова из списка токенов."""
        return [word for word in words if word not in self.stop_words]

    def tokenize(self, text: str) -> list:
        """Токенизирует текст, удаляя пунктуацию."""
        words = word_tokenize(text)  # Токенизация
        words = [word.lower() for word in words]  # Приводим к нижнему регистру
        words = [word for word in words if word not in string.punctuation]  # Убираем пунктуацию
        return words

    def preprocess(self, text: str) -> list:
        """Полный цикл препроцессинга текста."""
        text = self.clean_text(text)
        words = self.tokenize(text)  # Токенизируем до удаления стоп-слов
        words = self.remove_stopwords(words)
        return words
