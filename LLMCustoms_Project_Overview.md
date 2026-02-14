# LLMCustoms - Custom Fine-Tuning Library

## Project Vision

LLMCustoms is a user-friendly Python library that democratizes fine-tuning of large language models. It allows users to easily fine-tune models on their custom data without deep ML expertise, with automatic hardware optimization and model-specific pipelines.

## Core Philosophy

- **Simplicity First**: Users should be able to fine-tune models with minimal configuration
- **Hardware Aware**: Automatically detect and optimize for user's hardware capabilities
- **Model Specific**: Each supported model has its own optimized data pipeline and training configuration
- **Production Ready**: Easy integration with FastAPI and other frameworks for deployment

## Target Users

- Developers wanting to create domain-specific AI assistants
- Researchers needing quick model customization
- Businesses wanting to fine-tune models on proprietary data
- Anyone who wants custom AI without deep ML knowledge

## Supported Models

| Model | Size | VRAM Requirement | Use Case |
|-------|------|------------------|----------|
| TinyLlama | 1.1B | 4-6GB | Fast prototyping, resource-constrained |
| Phi-3.5-Mini | 3.8B | 6-8GB | Balanced performance/efficiency |
| Mistral-7B | 7B | 12-16GB | High quality responses |
| Qwen2.5 | 7B | 12-16GB | Multilingual, coding tasks |

## Key Features

### Phase 1: Core Foundation
- ✅ Basic fine-tuning pipeline
- ✅ Hardware detection and optimization
- ✅ Model-specific data processing
- ✅ .env configuration management

### Phase 2: Data Intelligence
- 🔄 PDF text extraction with structure preservation
- 🔄 Automatic Q&A generation using Groq API
- 🔄 Smart data chunking for large documents
- 🔄 Model-specific chat format conversion

### Phase 3: Training Optimization
- ⏳ Preset configurations (HighSpeed, Quality, BestAccuracy)
- ⏳ Advanced LoRA parameter tuning
- ⏳ Training monitoring and logging
- ⏳ Checkpoint management and recovery

### Phase 4: Integration & Deployment
- ⏳ FastAPI integration helpers
- ⏳ Model serving utilities
- ⏳ Easy deployment patterns
- ⏳ Documentation and examples

## Architecture Overview

```
LLMCustoms/
├── llmcustoms/
│   ├── __init__.py
│   ├── core/
│   │   ├── fine_tuner.py          # Main FineTuner class
│   │   ├── model_manager.py       # Model download/management
│   │   └── hardware_detector.py   # VRAM detection & optimization
│   ├── data/
│   │   ├── processors/
│   │   │   ├── pdf_processor.py   # PDF text extraction
│   │   │   ├── text_processor.py  # Text file processing
│   │   │   └── qa_generator.py    # Groq Q&A generation
│   │   └── formatters/
│   │       ├── tinyllama_formatter.py
│   │       ├── phi_formatter.py
│   │       ├── mistral_formatter.py
│   │       └── qwen_formatter.py
│   ├── models/
│   │   ├── configs/               # Model-specific configurations
│   │   └── templates/             # Chat templates for each model
│   ├── training/
│   │   ├── presets.py            # Training presets
│   │   ├── trainer.py            # Training logic
│   │   └── utils.py              # Training utilities
│   └── utils/
│       ├── config.py             # .env configuration
│       ├── logger.py             # Logging utilities
│       └── validators.py         # Input validation
├── examples/
├── docs/
├── tests/
├── setup.py
└── README.md
```

## User Workflow

### Simple Usage
```python
from llmcustoms import FineTuner

# Basic usage - library handles everything
tuner = FineTuner(
    data_path="./my_documents/",
    model="auto",  # Auto-select based on VRAM
    preset="quality"
)

model_path = tuner.train()
```

### Advanced Usage
```python
from llmcustoms import FineTuner

# Advanced configuration
tuner = FineTuner(
    data_path="./documents/",
    model="phi-3.5-mini",
    preset="custom",
    config={
        "learning_rate": 5e-4,
        "batch_size": 4,
        "epochs": 3,
        "lora_rank": 16
    }
)

model_path = tuner.train()
```

### Integration with FastAPI
```python
from llmcustoms.deployment import ModelServer
from fastapi import FastAPI

app = FastAPI()
model_server = ModelServer(model_path="./finetuned_model")

@app.post("/chat")
async def chat(message: str):
    return model_server.generate(message)
```

## Configuration (.env)

```env
# Required
GROQ_API_KEY=your_groq_api_key_here

# Optional - Hardware Override
FORCE_CPU=false
MAX_VRAM_GB=8

# Optional - Training Defaults
DEFAULT_MODEL=auto
DEFAULT_PRESET=quality
OUTPUT_DIR=./models/

# Optional - Advanced
CACHE_DIR=~/.llmcustoms/
LOG_LEVEL=INFO
```

## Training Presets

### HighSpeed
- **Goal**: Fastest training time
- **Quality**: Basic
- **VRAM**: Minimal usage
- **Use Case**: Quick prototyping, testing

### Quality (Default)
- **Goal**: Balanced speed/quality
- **Quality**: Good
- **VRAM**: Moderate usage
- **Use Case**: Most production use cases

### BestAccuracy
- **Goal**: Highest quality output
- **Quality**: Excellent
- **VRAM**: Higher usage
- **Use Case**: Critical applications, research

## Model-Specific Pipelines

### TinyLlama Pipeline
- **Chat Format**: `<|user|>\n{prompt}\n<|assistant|>\n{response}<|end|>`
- **Max Context**: 512 tokens
- **LoRA Rank**: 8
- **Batch Size**: 2-4

### Phi-3.5 Pipeline
- **Chat Format**: `<|user|>\n{prompt}<|end|>\n<|assistant|>\n{response}<|end|>`
- **Max Context**: 1024 tokens
- **LoRA Rank**: 16
- **Batch Size**: 1-2

### Mistral Pipeline
- **Chat Format**: `[INST] {prompt} [/INST] {response}`
- **Max Context**: 2048 tokens
- **LoRA Rank**: 32
- **Batch Size**: 1

### Qwen Pipeline
- **Chat Format**: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>`
- **Max Context**: 2048 tokens
- **LoRA Rank**: 32
- **Batch Size**: 1

## Hardware Requirements

### Minimum
- **GPU**: 4GB VRAM (TinyLlama only)
- **RAM**: 8GB system RAM
- **Storage**: 10GB free space

### Recommended
- **GPU**: 8GB VRAM (Phi-3.5, optimized settings)
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space

### Optimal
- **GPU**: 16GB+ VRAM (All models, best settings)
- **RAM**: 32GB system RAM
- **Storage**: 100GB+ free space

## Success Metrics

### Phase 1 Success
- [ ] Successfully fine-tune TinyLlama on custom text data
- [ ] Automatic hardware detection working
- [ ] Basic .env configuration system
- [ ] Model downloads and caches properly

### Phase 2 Success
- [ ] PDF processing with structure preservation
- [ ] Groq Q&A generation working reliably
- [ ] All 4 model pipelines implemented
- [ ] Smart data chunking for large documents

### Phase 3 Success
- [ ] All 3 presets working optimally
- [ ] Training monitoring and logging
- [ ] Checkpoint recovery system
- [ ] Advanced parameter tuning

### Phase 4 Success
- [ ] FastAPI integration examples
- [ ] Easy deployment patterns
- [ ] Comprehensive documentation
- [ ] PyPI package published

## Technical Decisions

### Why These Models?
- **TinyLlama**: Fastest training, lowest VRAM, good for prototyping
- **Phi-3.5**: Microsoft's efficient model, great performance/size ratio
- **Mistral**: Industry standard, excellent quality
- **Qwen**: Strong multilingual and coding capabilities

### Why LoRA?
- Memory efficient (fits in consumer GPUs)
- Fast training compared to full fine-tuning
- Easy to merge and deploy
- Proven effectiveness across model types

### Why Groq for Q&A Generation?
- Fast inference for data processing
- Cost-effective for large datasets
- Good quality Q&A generation
- Easy API integration

## Future Considerations

### Potential Phase 5+ Features
- Web UI for non-technical users
- Multi-GPU training support
- More model formats (GGUF export)
- Cloud training integration
- Model quantization options
- Custom evaluation metrics
- Integration with more deployment platforms

### Scalability Considerations
- Plugin architecture for new models
- Distributed training support
- Enterprise features (team management, model versioning)
- Integration with MLOps platforms

---

This project aims to make fine-tuning as easy as `pip install llmcustoms` and running a few lines of Python code, while maintaining the flexibility for advanced users to customize everything.