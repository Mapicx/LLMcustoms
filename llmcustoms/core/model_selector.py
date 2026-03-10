from ..models.BaseModel import BaseModel
from ..core.hardware_detector_new import HardwareDetector
from ..models.gemma_9b import Gemma
from ..models.qwen_7b import Qwen
from ..models.Tinyllama import TinyLlama
from ..models.phi_35 import Phi

class ModelSelector:

    def __init__(self):
        self.hardware = HardwareDetector()
        self.vram = self.hardware.detect_gpu_memory()

    def select_best_model(self, models: list[BaseModel], preset: str) -> BaseModel | None:
        supported_model_preset = {}

        for model in models:
            if model.supports_vrams(self.vram, preset):
                supported_model_preset[model] = model.get_model_params()

        if not supported_model_preset:
            return None

        sorted_models = sorted(
            supported_model_preset.items(),
            key=lambda x: x[1],  
            reverse=True          
        )

        best_model = sorted_models[0][0]

        return best_model

    def select_best_preset(self, model: BaseModel) -> str:
        if self.vram < model.vram_requirements['min']:
            raise ValueError(
                f"Insufficient VRAM ({self.vram}GB). "
                f"Model requires at least {model.vram_requirements['min']}GB"
            )
        elif self.vram < model.vram_requirements['recommended']:
            return 'highspeed'
        elif self.vram < model.vram_requirements['optimal']:
            return 'balanced'
        else:
            return 'highquality'
    
    def select_best_model_and_preset(self):
        if self.vram <= 2:
            model = TinyLlama()
        elif self.vram <= 4:
            model = Phi()
        elif self.vram <= 6:
            model = Qwen()
        else:
            model = Gemma()
        
        preset = self.select_best_preset(model)
        return {model: preset}
