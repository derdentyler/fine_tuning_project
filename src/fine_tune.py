import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.dataset import TextDataset
from src.utils.config_loader import ConfigLoader
import os

# Загружаем конфигурацию
config = ConfigLoader("config.yaml").config

# Пути к данным и параметрам
train_data_path = "data/processed/train.json"
val_data_path = "data/processed/val.json"
model_name = config["model_name"]  # Например, "cointegrated/rubert-tiny2"
save_dir = "checkpoints"

# Проверяем, есть ли checkpoints
os.makedirs(save_dir, exist_ok=True)

# Проверяем, есть ли CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загружаем датасеты
train_dataset = TextDataset(train_data_path, "config.yaml", model_name)
val_dataset = TextDataset(val_data_path, "config.yaml", model_name)

# Создаём DataLoader (не нужен для Trainer, но полезен при отладке)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Загружаем предобученную модель
num_labels = len(train_dataset.get_label_mapping())
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

# Определяем аргументы тренировки
training_args = TrainingArguments(
    output_dir=save_dir,  # Где сохранять модель
    evaluation_strategy="epoch",  # Оценивать каждую эпоху
    save_strategy="epoch",  # Сохранять чекпоинты каждую эпоху
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    save_total_limit=2,  # Хранить только 2 последних чекпоинта
    push_to_hub=False,
)

# Trainer API от Hugging Face
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Запуск обучения
trainer.train()

# Сохранение итоговой модели
model.save_pretrained(os.path.join(save_dir, "final_model"))
print("Training complete! Model saved.")
