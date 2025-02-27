from transformers import AutoTokenizer, AutoModel
from src.utils.config_loader import ConfigLoader
import os
import torch


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def test_model_loading():
    """Проверяет, загружается ли модель и токенизатор."""
    config = ConfigLoader("config.yaml").config
    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    assert tokenizer is not None, "Ошибка загрузки токенизатора!"
    assert model is not None, "Ошибка загрузки модели!"
    assert isinstance(model, torch.nn.Module), "Модель должна быть экземпляром nn.Module!"
    print("✅ Тест загрузки модели и токенизатора пройден!")


