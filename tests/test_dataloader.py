import torch
from src.dataset import get_dataloader

def test_dataloader():
    """Проверяет, что DataLoader загружается корректно."""
    model_name = "cointegrated/rubert-tiny2"  # Легкая модель
    dataloader = get_dataloader("data/processed/train.json", "config.yaml", model_name, batch_size=4)

    batch = next(iter(dataloader))  # Берем первый батч
    encodings, labels = batch  # Теперь batch содержит токенизированные тексты и метки

    assert isinstance(encodings, dict), "Токенизированные данные должны быть словарем!"
    assert "input_ids" in encodings, "В токенах должен быть ключ 'input_ids'!"
    assert isinstance(labels, torch.Tensor), "Метки должны быть тензором!"
    assert labels.ndim == 1, "Метки должны быть одномерным тензором!"
    assert labels.shape[0] == 4, "Размер батча должен совпадать с batch_size (4)!"

    print("✅ Тест DataLoader пройден!")
