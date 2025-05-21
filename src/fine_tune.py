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
from src.utils.config_model import AppConfig  # импорт Pydantic-модели

logger = LoggerLoader().get_logger()

def fine_tune_model(cfg: AppConfig, model_name: str):
    """
    cfg: AppConfig — валидированный конфиг из ConfigLoader().get_config()
    model_name: str — имя модели или путь, можно переопределить через CLI
    """
    logger.info("Starting fine-tuning process...")

    # ========== Настройки из конфига ==========
    use_lora     = cfg.use_lora
    lora_r       = cfg.lora_r
    lora_alpha   = cfg.lora_alpha
    lora_dropout = cfg.lora_dropout
    save_dir     = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ========== Устройство ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========== Датасеты ==========
    train_ds = TextDataset(cfg.train_data_path, cfg.model_dump(), model_name)
    val_ds   = TextDataset(cfg.val_data_path,   cfg.model_dump(), model_name)
    num_labels = len(train_ds.get_label_mapping())

    # ========== Токенизатор и модель ==========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    logger.info(f"Loaded base model: {model_name}")

    # ========== Применение LoRA (PEFT) ==========
    if use_lora:
        logger.info("Applying LoRA PEFT...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none"
        )
        model = get_peft_model(model, peft_config)

    model.to(device)

    # ========== Аргументы тренировки ==========
    training_args = TrainingArguments(
        output_dir=save_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        logging_dir="logs",
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        push_to_hub=False,
    )

    # ========== Trainer ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=TextDataset.collate_fn,
        tokenizer=tokenizer,
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
