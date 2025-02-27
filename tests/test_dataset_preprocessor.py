import unittest
from src.processing.text_preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    def setUp(self):
        """Настраиваем препроцессор перед тестами."""
        self.preprocessor = TextPreprocessor(language='russian')
        self.sample_texts = [
            "Привет! Это тестовый текст. Как дела?",
            "NLTK — это библиотека для обработки естественного языка.",
            "12345, спецсимволы: @#$%^&*()",
        ]

    def test_clean_text(self):
        """Проверяем, удаляет ли препроцессор пунктуацию и приводит текст к нижнему регистру."""
        processed = self.preprocessor.clean_text(self.sample_texts[0])
        self.assertEqual(processed, "привет это тестовый текст как дела")

    def test_remove_stopwords(self):
        """Проверяем, удаляет ли препроцессор стоп-слова."""
        processed = self.preprocessor.remove_stopwords("это очень важный тест")
        self.assertNotIn("это", processed)
        self.assertNotIn("очень", processed)

    def test_tokenize(self):
        """Проверяем, корректно ли работает токенизация."""
        tokens = self.preprocessor.tokenize(self.sample_texts[1])
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 1)  # Должно быть больше одного токена

    def test_full_preprocessing(self):
        """Проверяем полный цикл препроцессинга текста."""
        processed = self.preprocessor.preprocess(self.sample_texts[0])
        self.assertIsInstance(processed, list)
        self.assertGreater(len(processed), 1)  # После препроцессинга должны остаться токены

if __name__ == "__main__":
    unittest.main()
