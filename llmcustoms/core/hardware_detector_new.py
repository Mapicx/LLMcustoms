import torch
import psutil
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class HardwareDetector:
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
    
    def is_gpu_available(self) -> bool:
        self.gpu_available
    
    def get_gpu_count(self) -> int:
        return self.gpu_count
    
    def get_system_ram(self) -> float:
        return self.system_ram_gb