import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.dataset import TextDataset
from src.utils.logger_loader import LoggerLoader

logger = LoggerLoader().get_logger()

def fine_tune_model(config: dict, model_name: str):
    logger.info("Starting fine-tuning process...")

    # Пути к данным из конфига
    train_data_path = config["train_data_path"]
    val_data_path = config["val_data_path"]
    save_dir = config.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Проверяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Загружаем датасеты
    train_dataset = TextDataset(train_data_path, config, model_name)
    val_dataset = TextDataset(val_data_path, config, model_name)

    # Загружаем предобученную модель
    num_labels = len(train_dataset.get_label_mapping())
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

    # Определяем аргументы тренировки
    training_args = TrainingArguments(
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=config.get("batch_size", 8),
        per_device_eval_batch_size=config.get("batch_size", 8),
        num_train_epochs=config.get("num_epochs", 3),
        weight_decay=config.get("weight_decay", 0.01),
        logging_dir="logs",
        logging_steps=config.get("logging_steps", 10),
        save_total_limit=config.get("save_total_limit", 2),
        push_to_hub=False,
    )

    # Trainer API от Hugging Face
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=TextDataset.collate_fn,
    )

    # Запуск обучения
    logger.info("Training started...")
    trainer.train()

    # Сохранение итоговой модели
    final_model_path = os.path.join(save_dir, "final_model")
    model.save_pretrained(final_model_path)
    logger.info(f"Training complete! Model saved to {final_model_path}")
