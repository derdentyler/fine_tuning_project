import pytest
from src.utils.config_loader import ConfigLoader

CONFIG_PATH = "tests/configs/test_config.yaml"  # Указываем путь к тестовому конфигу


@pytest.mark.unit
def test_load_valid_config():
    """Проверяет, что тестовый конфиг загружается корректно."""
    loader = ConfigLoader(config_path=CONFIG_PATH)
    assert isinstance(loader.config, dict), "Конфиг должен быть словарем"
    assert "categories" in loader.config, "Конфиг должен содержать ключ 'categories'"


@pytest.mark.unit
def test_load_missing_file():
    """Проверяет, что при отсутствии файла загружается пустой конфиг."""
    loader = ConfigLoader(config_path="non_existent.yaml")
    assert loader.config == {}, "Конфиг должен быть пустым при отсутствии файла"


@pytest.mark.unit
def test_get_categories():
    """Проверяет, что get_categories() возвращает данные из test_config.yaml."""
    loader = ConfigLoader(config_path=CONFIG_PATH)
    categories = loader.get_categories()

    assert isinstance(categories, dict), "Метод get_categories() должен возвращать словарь"
    assert categories, "Категории не должны быть пустыми"
