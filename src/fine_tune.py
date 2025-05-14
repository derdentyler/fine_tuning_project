import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from src.dataset import TextDataset
from src.utils.logger_loader import LoggerLoader

logger = LoggerLoader().get_logger()

def fine_tune_model(config: dict, model_name: str):
    logger.info("Starting fine-tuning process...")

    # ========== Настройки из конфига ==========
    use_lora    = config.get("use_lora", False)              # флаг включения LoRA
    lora_r      = config.get("lora_r", 8)                    # ранг адаптера (8–16) :contentReference[oaicite:3]{index=3}
    lora_alpha  = config.get("lora_alpha", 32)               # масштабирование градиента :contentReference[oaicite:4]{index=4}
    lora_dropout= config.get("lora_dropout", 0.1)            # dropout для адаптера :contentReference[oaicite:5]{index=5}
    save_dir    = config.get("save_dir", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # ========== Устройство ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========== Датасеты ==========
    train_ds = TextDataset(config["train_data_path"], config, model_name)
    val_ds   = TextDataset(config["val_data_path"],   config, model_name)
    num_labels = len(train_ds.get_label_mapping())

    # ========== Токенизатор и модель ==========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    logger.info(f"Loaded base model: {model_name}")

    # ========== Применение LoRA (PEFT) ==========
    if use_lora:
        logger.info("Applying LoRA PEFT...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,     # Sequence Classification :contentReference[oaicite:6]{index=6}
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],  # части attention :contentReference[oaicite:7]{index=7}
            bias="none"
        )
        model = get_peft_model(model, peft_config)        # оборачиваем модель в PEFT :contentReference[oaicite:8]{index=8}

    model.to(device)

    # ========== Аргументы тренировки ==========
    training_args = TrainingArguments(
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=config.get("batch_size", 8),
        per_device_eval_batch_size=config.get("batch_size", 8),
        num_train_epochs=config.get("num_epochs", 3),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        logging_dir="logs",
        logging_steps=config.get("logging_steps", 10),
        save_total_limit=config.get("save_total_limit", 2),
        push_to_hub=False,
    )

    # ========== Trainer ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=TextDataset.collate_fn,
    )

    # ========== Запуск обучения ==========
    logger.info("Training started...")
    trainer.train()

    # ========== Сохранение ==========
    final_dir = os.path.join(save_dir, "final_model")
    model.save_pretrained(final_dir)
    logger.info(f"Model saved to {final_dir}")

    if use_lora:
        logger.info("LoRA adapters saved separately.")
