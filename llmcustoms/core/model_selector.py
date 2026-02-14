from ..models.BaseModel import BaseModel
from ..core.hardware_detector_new import HardwareDetector


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
        pass
