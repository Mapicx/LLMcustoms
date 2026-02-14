"""
Hardware detection and optimization utilities for LLMCustoms.
"""

import torch
import psutil
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HardwareDetector:
    SUPPORTED_MODELS = {
        "phi35b": "phi 3.5b",
        "gemma9b": "gemma 9b",
        "tinyllama11b": "tinyllama 1.1b"
    }

    SUPPORTED_PRESETS = ["highspeed", "balanced", "highquality"]

    MODEL_VRAM_REQUIREMENTS = {
        "tinyllama11b": {"min": 2, "recommended": 3, "optimal": 4},
        "phi35b": {"min": 4, "recommended": 5, "optimal": 6},
        "gemma9b": {"min": 6, "recommended": 12, "optimal": 16}
    }

    MODEL_SIZES = {
        "tinyllama11b": 1.1,
        "phi35b": 3.8,
        "gemma9b": 9.0
    }

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.system_ram_gb = psutil.virtual_memory().total / (1024**3)

        if self.gpu_available:
            logger.info(f"Detected {self.gpu_count} GPU(s)")
        else:
            logger.warning("No CUDA-compatible GPU detected.")

    def detect_gpu_memory(self) -> int:
        if not self.gpu_available:
            return 0
        try:
            torch.cuda.empty_cache()
            gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
            return int(gpu_memory_bytes / (1024**3) - 1)
        except Exception:
            return 0

    # 🔥 NEW FUNCTION 1
    def get_best_preset_for_model(self, model_name: str) -> str:
        """
        Given a model name, returns the best safe preset
        based on available VRAM.
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")

        vram = self.detect_gpu_memory()
        requirements = self.MODEL_VRAM_REQUIREMENTS[model_name]

        if vram >= requirements["optimal"]:
            return "highquality"
        elif vram >= requirements["recommended"]:
            return "balanced"
        elif vram >= requirements["min"]:
            return "highspeed"
        else:
            return "highspeed"  # fallback

    # 🔥 NEW FUNCTION 2
    def get_best_model_for_preset(self, preset: str) -> str:
        """
        Given a preset, returns the largest runnable model
        that safely supports that preset.
        """
        if preset not in self.SUPPORTED_PRESETS:
            raise ValueError(f"Unsupported preset: {preset}")

        vram = self.detect_gpu_memory()

        # Sort models by size descending
        sorted_models = sorted(
            self.MODEL_SIZES.keys(),
            key=lambda m: self.MODEL_SIZES[m],
            reverse=True
        )

        for model in sorted_models:
            requirements = self.MODEL_VRAM_REQUIREMENTS[model]

            if preset == "highquality" and vram >= requirements["optimal"]:
                return model
            elif preset == "balanced" and vram >= requirements["recommended"]:
                return model
            elif preset == "highspeed" and vram >= requirements["min"]:
                return model

        # fallback to smallest model
        return "tinyllama11b"

    # Existing function (kept unchanged)
    def get_runnable_models(self) -> Dict[str, str]:
        """
        Returns the single best runnable model and its highest safe preset
        based on available VRAM.
        """
        vram = self.detect_gpu_memory()

        if vram >= self.MODEL_VRAM_REQUIREMENTS["gemma9b"]["optimal"]:
            return {"model": "gemma9b", "preset": "highquality"}
        elif vram >= self.MODEL_VRAM_REQUIREMENTS["gemma9b"]["recommended"]:
            return {"model": "gemma9b", "preset": "balanced"}
        elif vram >= self.MODEL_VRAM_REQUIREMENTS["gemma9b"]["min"]:
            return {"model": "gemma9b", "preset": "highspeed"}

        if vram >= self.MODEL_VRAM_REQUIREMENTS["phi35b"]["optimal"]:
            return {"model": "phi35b", "preset": "highquality"}
        elif vram >= self.MODEL_VRAM_REQUIREMENTS["phi35b"]["recommended"]:
            return {"model": "phi35b", "preset": "balanced"}
        elif vram >= self.MODEL_VRAM_REQUIREMENTS["phi35b"]["min"]:
            return {"model": "phi35b", "preset": "highspeed"}

        if vram >= self.MODEL_VRAM_REQUIREMENTS["tinyllama11b"]["optimal"]:
            return {"model": "tinyllama11b", "preset": "highquality"}
        elif vram >= self.MODEL_VRAM_REQUIREMENTS["tinyllama11b"]["recommended"]:
            return {"model": "tinyllama11b", "preset": "balanced"}
        elif vram >= self.MODEL_VRAM_REQUIREMENTS["tinyllama11b"]["min"]:
            return {"model": "tinyllama11b", "preset": "highspeed"}

        return {"model": "tinyllama11b", "preset": "highspeed"}

    def get_best_model_and_preset(self) -> Dict[str, str]:
        """(Kept for backward compatibility)"""
        return self.get_runnable_models()