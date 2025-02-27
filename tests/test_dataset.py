import pytest
import torch
from src.dataset import TextDataset


def test_dataset_loading():
    dataset = TextDataset("data/processed/train.json", "config.yaml", "DeepPavlov/rubert-base-cased")
    assert len(dataset) > 0, "Датасет пустой!"

    encoding, label = dataset[0]

    # Проверяем, что encoding - это словарь с нужными ключами
    assert isinstance(encoding, dict), "Токенизированные данные должны быть словарем!"
    assert "input_ids" in encoding and "attention_mask" in encoding, "В encoding отсутствуют ключи input_ids или attention_mask!"
    assert isinstance(encoding["input_ids"], torch.Tensor), "input_ids должен быть тензором!"

    # Проверяем, что метка - это тензор
    assert isinstance(label, torch.Tensor), "Метка должна быть тензором!"
    assert label.dtype == torch.long, "Метка должна быть типа long!"


def test_tokenization():
    dataset = TextDataset("data/processed/train.json", "config.yaml", "DeepPavlov/rubert-base-cased")

    tokens, label = dataset[0]  # Теперь принимаем два значения

    # Проверяем, что tokens — это словарь с тензорами (а не сам тензор!)
    assert isinstance(tokens, dict), "Токенизированный текст должен быть словарем с тензорами!"

    # Проверяем, что словарь содержит нужные ключи (input_ids, attention_mask)
    assert "input_ids" in tokens and "attention_mask" in tokens, "tokens должен содержать input_ids и attention_mask"

    # Проверяем, что input_ids и attention_mask являются тензорами
    assert isinstance(tokens["input_ids"], torch.Tensor), "input_ids должен быть тензором!"
    assert isinstance(tokens["attention_mask"], torch.Tensor), "attention_mask должен быть тензором!"

    # Проверяем, что label — это тензор типа long
    assert isinstance(label, torch.Tensor), "Метка должна быть тензором!"
    assert label.dtype == torch.long, "Метка должна иметь тип long!"

    print("✅ Тест прошел успешно!")
