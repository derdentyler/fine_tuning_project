import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.utils.config_loader import ConfigLoader
import os
from torch.utils.data import DataLoader


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class TextDataset(Dataset):
    def __init__(self, json_path: str, config_path: str, model_name: str, max_length: int = 512):
        """
        Загружает данные из JSON, токенизирует текст и преобразует категории в числовые индексы.

        :param json_path: путь к JSON-файлу с данными.
        :param config_path: путь к конфигу с категориями.
        :param model_name: имя предобученной модели (используется для загрузки соответствующего токенизатора).
        :param max_length: максимальная длина токенизированного текста (по умолчанию 512).
        """
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Загружаем список категорий из конфига и создаем маппинг категория -> индекс
        config = ConfigLoader(config_path).config
        self.label_to_idx = {cat: idx for idx, cat in enumerate(config["categories"])}
        self.idx_to_label = {idx: cat for cat, idx in self.label_to_idx.items()}

        # Загружаем токенизатор от модели
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

        # Проверяем, что JSON содержит нужные ключи
        for item in self.data:
            if "text" not in item or "category" not in item:
                raise ValueError("JSON-файл должен содержать ключи 'text' и 'category'.")

        # Извлекаем тексты и метки, преобразуем категории в индексы
        self.texts = [item["text"] for item in self.data]
        self.labels = [self.label_to_idx[item["category"]] for item in self.data]

    def __len__(self):
        """ Возвращает количество примеров в датасете. """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Возвращает токенизированный текст и числовой индекс категории.

        :param idx: индекс примера.
        :return: токенизированный текст (словарь с тензорами) и метка категории (int).
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Токенизируем текст с padding и truncation
        encoding = self.tokenizer(
            text,
            padding="max_length",  # Дополняем до max_length, чтобы все примеры были одной длины
            truncation=True,  # Обрезаем слишком длинные тексты
            max_length=self.max_length,
            return_tensors="pt"  # Возвращаем тензоры PyTorch
        )

        # `encoding` - это объект с несколькими тензорами (input_ids, attention_mask и т. д.)
        # Нам нужно убрать дополнительное измерение, так как `return_tensors="pt"` возвращает 2D-тензор.
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}

        return encoding, torch.tensor(label, dtype=torch.long)

    def get_label_mapping(self):
        """ Возвращает словарь {категория: индекс}. """
        return self.label_to_idx

    @staticmethod
    def collate_fn(batch):
        """
        Функция объединения батча для DataLoader.

        :param batch: список кортежей (токенизированный текст, метка)
        :return: батч с текстами (dict of tensors) и метками (tensor)
        """
        texts, labels = zip(*batch)  # Разделяем тексты и метки

        # Объединяем тексты в батч (создаем единую структуру с input_ids, attention_mask и т. д.)
        batch_texts = {key: torch.stack([text[key] for text in texts]) for key in texts[0]}

        # Создаем тензор для меток
        batch_labels = torch.tensor(labels, dtype=torch.long)

        return batch_texts, batch_labels


def get_dataloader(json_path, config_path, model_name, batch_size=8, shuffle=True):
    """
    Создает DataLoader для работы с батчами.

    :param json_path: путь к JSON-файлу с данными.
    :param config_path: путь к конфигу.
    :param model_name: название предобученной модели для токенизации.
    :param batch_size: размер батча (по умолчанию 8).
    :param shuffle: перемешивать данные или нет (по умолчанию True).
    :return: DataLoader
    """
    dataset = TextDataset(json_path, config_path, model_name)  # Теперь передаем model_name
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=TextDataset.collate_fn)
