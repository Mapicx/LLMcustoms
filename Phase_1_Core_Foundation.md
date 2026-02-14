# Phase 1: Core Foundation

## Overview
Establish the basic infrastructure for LLMCustoms library with essential fine-tuning capabilities, hardware detection, and configuration management.

## Duration: 2-3 weeks

## Goals
- Create a working fine-tuning pipeline for TinyLlama
- Implement hardware detection and VRAM-based optimization
- Set up project structure and basic configuration system
- Establish model download and management system

## Deliverables

### 1. Project Structure Setup
```
llmcustoms/
├── llmcustoms/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── fine_tuner.py
│   │   ├── model_manager.py
│   │   └── hardware_detector.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── validators.py
│   └── training/
│       ├── __init__.py
│       ├── trainer.py
│       └── presets.py
├── examples/
│   └── basic_example.py
├── tests/
├── setup.py
├── requirements.txt
└── README.md
```

### 2. Core Components

#### 2.1 Hardware Detector (`hardware_detector.py`)
```python
class HardwareDetector:
    def detect_gpu_memory(self) -> int
    def get_optimal_batch_size(self, model_size: str) -> int
    def suggest_model(self, available_vram: int) -> str
    def get_training_config(self, model: str, vram: int) -> dict
```

**Features:**
- Detect NVIDIA GPU VRAM using `torch.cuda`
- Suggest optimal model based on available VRAM
- Return recommended training parameters
- Handle CPU-only fallback with warnings

#### 2.2 Model Manager (`model_manager.py`)
```python
class ModelManager:
    def download_model(self, model_name: str) -> str
    def check_local_model(self, model_path: str) -> bool
    def get_model_info(self, model_name: str) -> dict
    def cache_model(self, model_name: str) -> str
```

**Features:**
- Download models from HuggingFace Hub
- Cache models in `~/.llmcustoms/models/`
- Check if model exists locally
- Model metadata management

#### 2.3 Configuration Manager (`config.py`)
```python
class Config:
    def load_env_config(self) -> dict
    def validate_config(self, config: dict) -> bool
    def get_default_config(self) -> dict
    def merge_configs(self, user_config: dict, defaults: dict) -> dict
```

**Features:**
- Load configuration from `.env` file
- Validate required parameters (GROQ_API_KEY)
- Provide sensible defaults
- Support environment variable overrides

#### 2.4 Basic Fine Tuner (`fine_tuner.py`)
```python
class FineTuner:
    def __init__(self, data_path: str, model: str = "auto", preset: str = "quality")
    def prepare_data(self) -> Dataset
    def setup_model(self) -> tuple[AutoModel, AutoTokenizer]
    def train(self) -> str
    def test_model(self, prompts: list = None) -> None
```

**Features:**
- Basic text file processing (.txt only)
- Simple Q&A format conversion
- TinyLlama fine-tuning with LoRA
- Model saving and testing

### 3. Training Presets (`presets.py`)

#### 3.1 HighSpeed Preset
```python
HIGHSPEED_CONFIG = {
    "num_train_epochs": 1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "learning_rate": 1e-3,
    "max_steps": 100,
    "lora_rank": 4,
    "lora_alpha": 8,
}
```

#### 3.2 Quality Preset (Default)
```python
QUALITY_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "max_steps": 300,
    "lora_rank": 8,
    "lora_alpha": 16,
}
```

#### 3.3 BestAccuracy Preset
```python
BESTACCURACY_CONFIG = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "max_steps": 1000,
    "lora_rank": 16,
    "lora_alpha": 32,
}
```

### 4. Basic Data Processing

#### 4.1 Text File Processing
- Read `.txt` files from specified directory
- Basic text chunking (512 tokens max for TinyLlama)
- Simple Q&A format: `Question: ... Answer: ...`
- Convert to TinyLlama chat format

#### 4.2 Data Validation
- Check file formats and sizes
- Validate text encoding (UTF-8)
- Ensure minimum data requirements
- Warning for insufficient data

### 5. Logging and Error Handling

#### 5.1 Logger Setup (`logger.py`)
```python
class Logger:
    def setup_logger(self, log_level: str = "INFO") -> logging.Logger
    def log_training_progress(self, metrics: dict) -> None
    def log_hardware_info(self, hardware_info: dict) -> None
```

**Features:**
- Console and file logging
- Training progress tracking
- Hardware detection logging
- Error reporting with context

#### 5.2 Error Handling
- Graceful CUDA out of memory handling
- Model download failure recovery
- Invalid configuration warnings
- Data processing error reporting

### 6. Basic Example Usage

```python
# examples/basic_example.py
from llmcustoms import FineTuner

# Simple usage
tuner = FineTuner(
    data_path="./my_text_files/",
    model="tinyllama",  # or "auto"
    preset="quality"
)

# Train the model
model_path = tuner.train()
print(f"Model saved to: {model_path}")

# Test the model
tuner.test_model([
    "What is machine learning?",
    "How do I install Python?",
    "Explain neural networks simply."
])
```

## Technical Requirements

### Dependencies
```txt
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.24.0
python-dotenv>=1.0.0
tqdm>=4.66.0
```

### Environment Variables (.env)
```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional
DEFAULT_MODEL=auto
DEFAULT_PRESET=quality
OUTPUT_DIR=./models/
CACHE_DIR=~/.llmcustoms/
LOG_LEVEL=INFO
MAX_VRAM_GB=auto
FORCE_CPU=false
```

## Testing Strategy

### Unit Tests
- Hardware detection accuracy
- Configuration loading and validation
- Model download and caching
- Data processing pipeline

### Integration Tests
- End-to-end fine-tuning with sample data
- Hardware optimization validation
- Model saving and loading
- Error handling scenarios

### Manual Testing
- Test on different GPU configurations (4GB, 6GB, 8GB)
- Validate with various text file formats
- Test preset configurations
- Verify model quality with sample prompts

## Success Criteria

### Functional Requirements
- [ ] Successfully fine-tune TinyLlama on custom text data
- [ ] Automatic hardware detection suggests appropriate settings
- [ ] All three presets work without manual configuration
- [ ] Models download and cache properly
- [ ] Basic Q&A generation from text files

### Performance Requirements
- [ ] Training completes on 4GB VRAM (TinyLlama + HighSpeed preset)
- [ ] Model download < 5 minutes on average internet
- [ ] Data processing handles files up to 10MB
- [ ] Training progress visible in terminal

### Quality Requirements
- [ ] Fine-tuned model shows improvement over base model
- [ ] Generated responses are coherent and relevant
- [ ] No memory leaks during training
- [ ] Graceful error handling for common issues

## Risk Mitigation

### Technical Risks
- **CUDA compatibility issues**: Test on multiple GPU generations
- **Model download failures**: Implement retry logic and mirrors
- **Memory optimization**: Conservative default settings
- **Data quality**: Validate input data and provide warnings

### User Experience Risks
- **Complex setup**: Provide clear installation instructions
- **Unclear errors**: Implement user-friendly error messages
- **Performance expectations**: Document hardware requirements clearly

## Next Phase Preparation

### Phase 2 Prerequisites
- Stable TinyLlama fine-tuning pipeline
- Working hardware detection system
- Basic data processing framework
- Configuration management system

### Technical Debt to Address
- Add comprehensive error handling
- Implement proper logging throughout
- Create automated testing pipeline
- Document all APIs and configurations

---

**Phase 1 Completion Target**: Working fine-tuning library that can take text files and produce a custom TinyLlama model with minimal user configuration.