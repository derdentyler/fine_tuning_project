import pytest
from pydantic import ValidationError
from src.utils.config_loader import ConfigLoader
from src.utils.config_model import AppConfig

CONFIG_PATH = "tests/configs/test_config.yaml"


@pytest.mark.unit
def test_load_valid_config_returns_appconfig():
    """Проверяет, что из валидного YAML ConfigLoader.get_config() возвращает AppConfig."""
    loader = ConfigLoader(config_path=CONFIG_PATH)
    cfg = loader.get_config()

    # Проверяем тип и ключевые поля
    assert isinstance(cfg, AppConfig), "Ожидаем AppConfig, а не dict"
    assert cfg.model_name == "cointegrated/rubert-tiny2"
    assert cfg.output_dir.endswith("tests/data/processed/")
    assert isinstance(cfg.categories, dict)
    assert "delirium" in cfg.categories
    assert isinstance(cfg.categories["delirium"], list)


@pytest.mark.unit
def test_missing_file_raises_filenotfound():
    """Проверяет, что при отсутствии файла ConfigLoader.get_config() кидает FileNotFoundError."""
    loader = ConfigLoader(config_path="non_existent.yaml")
    with pytest.raises(FileNotFoundError):
        _ = loader.get_config()


@pytest.mark.unit
def test_invalid_config_raises_validation_error(tmp_path):
    """
    Проверяет, что при отсутствии обязательного поля (например, model_name)
    Pydantic внутри get_config() бросает ValidationError.
    """
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text(
        '# Убираем обязательное поле model_name\n'
        'output_dir: "out/"\n',
        encoding="utf-8"
    )

    loader = ConfigLoader(config_path=str(bad_cfg))
    with pytest.raises(ValidationError):
        _ = loader.get_config()
