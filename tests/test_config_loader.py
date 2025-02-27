from src.utils.config_loader import ConfigLoader

# Тест работы конфиг-лоадера
if __name__ == "__main__":
    config_loader = ConfigLoader()
    categories = config_loader.get_categories()

    print("✅ Загруженные категории:")
    for category, links in categories.items():
        print(f"{category}: {links}")
