from pydantic import BaseModel, Field
from typing import Dict, List

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
