from .BaseModel import BaseModel
import logging

logger = logging.getLogger(__name__)


class Phi(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_key = "microsoft/Phi-3.5-mini-instruct"
        self.display_name = "microsoft/Phi-3.5-mini-instruct"
        self.hf_repo = "microsoft/Phi-3.5-mini-instruct"
        self.model_size_gb = 3.8
        self.vram_requirements = {
            "min": 3,
            "recommended": 5,
            "optimal": 6
        }
        self.model_params = 3.5

    def supports_vrams(self, vram, preset) -> bool:
        if not self.is_a_supported_preset(preset):
            raise ValueError(
                f"Unsupported preset '{preset}'. "
                f"Choose from ['highspeed', 'balanced', 'highquality']"
            )

        if preset == "highspeed":
            return vram >= self.vram_requirements["min"]

        elif preset == "balanced":
            return vram >= self.vram_requirements["recommended"]

        elif preset == "highquality":
            return vram >= self.vram_requirements["optimal"]

    def get_loading_config(self, preset: str) -> dict:

        if not self.is_a_supported_preset(preset):
            raise ValueError(
                f"Unsupported preset '{preset}'. "
                f"Choose from ['highspeed', 'balanced', 'highquality']"
            )

        if preset == "highspeed":
            return {
                "LOAD_IN_4BIT": True,
                "LOAD_IN_8BIT": False,
                "BNB_4BIT_COMPUTE_DTYPE": "float16",
                "BNB_4BIT_QUANT_TYPE": "nf4",
                "attn_implementation": "auto",
            }

        elif preset == "balanced":
            return {
                "LOAD_IN_4BIT": True,
                "LOAD_IN_8BIT": False,
                "BNB_4BIT_COMPUTE_DTYPE": "bfloat16",
                "BNB_4BIT_QUANT_TYPE": "nf4",
                "attn_implementation": "auto",
            }

        elif preset == "highquality":
            return {
                "LOAD_IN_4BIT": False,
                "LOAD_IN_8BIT": True,
                "BNB_4BIT_COMPUTE_DTYPE": "float16",
                "BNB_4BIT_QUANT_TYPE": None,
                "attn_implementation": "auto",
            }

    def get_lora_congif(self, preset):
        if not self.is_a_supported_preset(preset):
            raise ValueError(
                f"Unsupported preset '{preset}'. "
                f"Choose from ['highspeed', 'balanced', 'highquality']"
            )

        if preset == "highspeed":
            return {
                "LORA_R": 8,
                "LORA_ALPHA": 16,
                "LORA_DROPOUT": 0.1,
                "LORA_TARGET_MODULES": ["q_proj", "v_proj"],
            }

        elif preset == "balanced":
            return {
                "LORA_R": 16,
                "LORA_ALPHA": 32,
                "LORA_DROPOUT": 0.05,
                "LORA_TARGET_MODULES": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }

        elif preset == "highquality":
            return {
                "LORA_R": 32,
                "LORA_ALPHA": 64,
                "LORA_DROPOUT": 0.03,
                "LORA_TARGET_MODULES": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
            }

    def get_training_config(self, preset):
        if not self.is_a_supported_preset(preset):
            raise ValueError(
                f"Unsupported preset '{preset}'. "
                f"Choose from ['highspeed', 'balanced', 'highquality']"
            )

        if preset == "highspeed":
            return {
                "MAX_LENGTH": 64,
                "BATCH_SIZE": 1,
                "GRAD_ACCUM_STEPS": 4,
                "LEARNING_RATE": 2e-4,
                "MAX_STEPS": 80,
            }

        elif preset == "balanced":
            return {
                "MAX_LENGTH": 96,
                "BATCH_SIZE": 2,
                "GRAD_ACCUM_STEPS": 4,
                "LEARNING_RATE": 1e-4,
                "MAX_STEPS": 120,
            }

        elif preset == "highquality":
            return {
                "MAX_LENGTH": 128,
                "BATCH_SIZE": 2,
                "GRAD_ACCUM_STEPS": 8,
                "LEARNING_RATE": 8e-5,
                "MAX_STEPS": 200,
            }

    def get_model_params(self):
        return self.model_params