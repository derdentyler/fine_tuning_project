import nltk
import os


nltk.download("punkt_tab")

# Принудительно указываем путь для NLTK
NLTK_PATH = os.path.join(os.path.expanduser("~"), "nltk_data")
if NLTK_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_PATH)

# Функция для проверки и загрузки ресурсов
def load_nltk_resources():
    for resource in ["punkt", "stopwords"]:
        try:
            nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
            print(f"✅ {resource} найден!")
        except LookupError:
            print(f"❌ {resource} не найден, скачиваем...")
            nltk.download(resource, download_dir=NLTK_PATH)

# Автоматический запуск при импорте
load_nltk_resources()
