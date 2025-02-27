from src.dataset_saver import DatasetSaver

# Тестовые данные
test_data = [
    {"category": "type1", "text": "Пример текста 1"},
    {"category": "type2", "text": "Пример текста 2"}
]

# Путь сохранения (временный файл)
test_save_path = "tests/output/test_dataset.json"

if __name__ == "__main__":
    saver = DatasetSaver(test_save_path)
    saver.save(test_data)

    # Проверяем, что файл создался
    with open(test_save_path, "r", encoding="utf-8") as f:
        saved_data = f.read()
        print("✅ Проверка сохраненного файла:")
        print(saved_data)  # Выведет JSON-файл, чтобы убедиться, что все корректно
