from pydantic import BaseModel, Field
from typing import Dict, List


class AugmentationConfig(BaseModel):
    input_path: str
    output_path: str
    min_examples: int
    bt_rounds: int
    bt_beam_size: int

    class Config:
        extra = "ignore"  # любые поля вне этого списка будут просто проигнорированы


class AppConfig(BaseModel):
    model_name: str
    output_dir: str
    subtitles_dir: str
    train_data_path: str
    val_data_path: str
    save_dir: str
    use_lora: bool
    lora_r: int = Field(..., ge=1)
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    logging_steps: int
    save_total_limit: int

    log_level: str
    log_format: str

    categories: Dict[str, List[str]]

    augmentation: AugmentationConfig

    class Config:
        extra = "ignore"
