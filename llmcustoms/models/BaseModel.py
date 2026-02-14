from abc import ABC, abstractmethod
import logging


logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self):
        self.model_key = None
        self.display_name = None
        self.hf_repo = None
        self.model_size_gb = None
        self.vram_requirements = None
        self.model_params = None
        self.supported_models = ["Gemma", "phi3.5b", "qwen7b", "tinyllama1.1b"]

    @abstractmethod
    def supports_vrams(self, vram: int, preset: str) -> bool:
        pass

    @abstractmethod
    def get_loading_config(self, preset):
        pass

    @abstractmethod 
    def get_lora_congif(self, preset):
        pass

    @abstractmethod
    def get_training_config(self, preset):
        pass

    @abstractmethod
    def get_model_params(self):
        return self.model_params
    
    @staticmethod
    def is_a_supported_preset(preset) -> bool:
        if preset in ["highquality", "balanced", "highspeed"]:
            return True

        logger.error(
            "Wrong Preset. Pick from - ['highspeed', 'balanced', 'highquality']"
        )
        return False
    
    def get_supported_models(self) -> list[str]:
        return self.supported_models
    
