from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, List

class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    # ===== Hugging Face =====
    hf_token: Optional[str] = None  # Will be read from .env

    # ===== Model =====
    model_name: str = "google/gemma-2-9b-it"
    max_length: int = 96

    # ===== Dataset =====
    dataset_name: Optional[str] = "Abirate/english_quotes"
    dataset_split: str = "train"
    dataset_text_field: str = "quote"
    dataset_path: Optional[str] = None
    dataset_label_field: Optional[str] = "tags" 

    # ===== Quantization =====
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # ===== LoRA =====
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = ["k_proj", "v_proj"]

    # ===== Training =====
    batch_size: int = 1
    grad_accum_steps: int = 2
    learning_rate: float = 2e-4
    max_steps: int = 30

    attn_implementation: str = "auto"

    # ===== System =====
    device: str = "cuda"
    output_dir: str = "outputs"


settings = TrainingConfig()
