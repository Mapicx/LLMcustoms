# Phase 2: Data Intelligence

## Overview
Expand data processing capabilities with PDF support, intelligent Q&A generation using Groq API, and implement model-specific data pipelines for all supported models.

## Duration: 3-4 weeks

## Goals
- Implement PDF text extraction with structure preservation
- Build robust Q&A generation system using Groq API
- Create model-specific data formatters for all 4 models
- Implement smart data chunking for large documents
- Add automatic content type detection and processing

## Deliverables

### 1. Enhanced Project Structure
```
llmcustoms/
├── data/
│   ├── __init__.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── base_processor.py
│   │   ├── pdf_processor.py
│   │   ├── text_processor.py
│   │   └── qa_generator.py
│   └── formatters/
│       ├── __init__.py
│       ├── base_formatter.py
│       ├── tinyllama_formatter.py
│       ├── phi_formatter.py
│       ├── mistral_formatter.py
│       └── qwen_formatter.py
├── models/
│   ├── __init__.py
│   ├── configs/
│   │   ├── tinyllama.json
│   │   ├── phi.json
│   │   ├── mistral.json
│   │   └── qwen.json
│   └── templates/
│       ├── tinyllama_template.py
│       ├── phi_template.py
│       ├── mistral_template.py
│       └── qwen_template.py
```

### 2. Data Processors

#### 2.1 Base Processor (`base_processor.py`)
```python
class BaseProcessor:
    def __init__(self, preserve_structure: bool = True)
    def process_file(self, file_path: str) -> dict
    def validate_content(self, content: str) -> bool
    def chunk_content(self, content: str, max_tokens: int) -> list[str]
    def extract_metadata(self, file_path: str) -> dict
```

#### 2.2 PDF Processor (`pdf_processor.py`)
```python
class PDFProcessor(BaseProcessor):
    def extract_text_with_structure(self, pdf_path: str) -> dict
    def preserve_headings(self, content: str) -> str
    def extract_tables(self, pdf_path: str) -> list[dict]
    def handle_multi_column_layout(self, content: str) -> str
    def clean_extracted_text(self, text: str) -> str
```

**Features:**
- Extract text while preserving document structure
- Maintain headings, paragraphs, and sections
- Handle multi-column layouts intelligently
- Extract and format tables as structured text
- Clean up common PDF extraction artifacts
- Support for encrypted PDFs (with password)

**Dependencies:**
```python
# New requirements
PyPDF2>=3.0.0
pdfplumber>=0.9.0
pymupdf>=1.23.0  # For complex layouts
```

#### 2.3 Enhanced Text Processor (`text_processor.py`)
```python
class TextProcessor(BaseProcessor):
    def detect_content_type(self, content: str) -> str
    def extract_existing_qa_pairs(self, content: str) -> list[dict]
    def split_by_sections(self, content: str) -> list[str]
    def preserve_formatting(self, content: str) -> str
    def handle_code_blocks(self, content: str) -> str
```

**Features:**
- Auto-detect content types (documentation, FAQ, tutorial, etc.)
- Extract existing Q&A pairs if present
- Intelligent section splitting
- Preserve important formatting (code blocks, lists)
- Handle technical documentation

#### 2.4 Q&A Generator (`qa_generator.py`)
```python
class QAGenerator:
    def __init__(self, groq_api_key: str, model: str = "llama-3.3-70b-versatile")
    def generate_qa_pairs(self, content: str, num_pairs: int = 5) -> list[dict]
    def generate_contextual_questions(self, content: str) -> list[str]
    def improve_answer_quality(self, question: str, answer: str) -> str
    def batch_generate(self, content_chunks: list[str]) -> list[dict]
    def validate_qa_quality(self, qa_pair: dict) -> bool
```

**Features:**
- Generate high-quality Q&A pairs from any text content
- Context-aware question generation
- Answer quality improvement and validation
- Batch processing for large documents
- Configurable question types (factual, explanatory, how-to)
- Rate limiting and error handling for Groq API

**Q&A Generation Strategies:**
```python
GENERATION_PROMPTS = {
    "factual": "Generate factual questions about key information in this text:",
    "explanatory": "Create questions that require explanation of concepts:",
    "how_to": "Generate how-to questions based on procedures described:",
    "troubleshooting": "Create troubleshooting questions for problems mentioned:",
    "comparative": "Generate questions comparing different aspects:"
}
```

### 3. Model-Specific Formatters

#### 3.1 Base Formatter (`base_formatter.py`)
```python
class BaseFormatter:
    def __init__(self, model_config: dict)
    def format_qa_pair(self, question: str, answer: str) -> str
    def format_conversation(self, messages: list[dict]) -> str
    def validate_format(self, formatted_text: str) -> bool
    def get_max_context_length(self) -> int
    def tokenize_and_validate(self, text: str) -> dict
```

#### 3.2 TinyLlama Formatter (`tinyllama_formatter.py`)
```python
class TinyLlamaFormatter(BaseFormatter):
    CHAT_TEMPLATE = "<|user|>\n{question}\n<|assistant|>\n{answer}<|end|>"
    MAX_CONTEXT = 512
    
    def format_qa_pair(self, question: str, answer: str) -> str
    def optimize_for_context_length(self, text: str) -> str
    def handle_long_answers(self, answer: str) -> str
```

#### 3.3 Phi-3.5 Formatter (`phi_formatter.py`)
```python
class PhiFormatter(BaseFormatter):
    CHAT_TEMPLATE = "<|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>"
    MAX_CONTEXT = 1024
    
    def format_qa_pair(self, question: str, answer: str) -> str
    def handle_system_messages(self, system_prompt: str) -> str
    def optimize_attention_patterns(self, text: str) -> str
```

#### 3.4 Mistral Formatter (`mistral_formatter.py`)
```python
class MistralFormatter(BaseFormatter):
    CHAT_TEMPLATE = "[INST] {question} [/INST] {answer}"
    MAX_CONTEXT = 2048
    
    def format_qa_pair(self, question: str, answer: str) -> str
    def handle_multi_turn_conversations(self, conversation: list) -> str
    def optimize_for_instruction_following(self, text: str) -> str
```

#### 3.5 Qwen Formatter (`qwen_formatter.py`)
```python
class QwenFormatter(BaseFormatter):
    CHAT_TEMPLATE = "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
    MAX_CONTEXT = 2048
    
    def format_qa_pair(self, question: str, answer: str) -> str
    def handle_multilingual_content(self, text: str) -> str
    def optimize_for_coding_tasks(self, text: str) -> str
```

### 4. Smart Data Chunking

#### 4.1 Intelligent Chunking (`chunking.py`)
```python
class SmartChunker:
    def __init__(self, model_type: str, max_tokens: int)
    def chunk_by_semantic_similarity(self, content: str) -> list[str]
    def chunk_by_document_structure(self, content: str) -> list[str]
    def chunk_with_overlap(self, content: str, overlap_ratio: float = 0.1) -> list[str]
    def preserve_context_boundaries(self, chunks: list[str]) -> list[str]
    def optimize_chunk_sizes(self, chunks: list[str]) -> list[str]
```

**Chunking Strategies:**
- **Semantic Chunking**: Group related content together
- **Structure-Based**: Split by headings, sections, paragraphs
- **Token-Aware**: Ensure chunks fit within model context limits
- **Overlap Preservation**: Maintain context between chunks
- **Quality Optimization**: Ensure each chunk can generate good Q&A pairs

### 5. Enhanced Configuration System

#### 5.1 Model Configurations (`models/configs/`)

**TinyLlama Config (`tinyllama.json`):**
```json
{
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "max_context_length": 512,
    "recommended_vram": 4,
    "chat_template": "<|user|>\n{question}\n<|assistant|>\n{answer}<|end|>",
    "lora_config": {
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout": 0.1
    },
    "training_config": {
        "learning_rate": 5e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "warmup_steps": 50
    }
}
```

**Similar configs for Phi, Mistral, and Qwen with their specific parameters**

### 6. Content Type Detection

#### 6.1 Content Analyzer (`content_analyzer.py`)
```python
class ContentAnalyzer:
    def detect_content_type(self, text: str) -> str
    def analyze_document_structure(self, text: str) -> dict
    def identify_qa_patterns(self, text: str) -> bool
    def detect_code_content(self, text: str) -> float
    def assess_technical_level(self, text: str) -> str
```

**Content Types:**
- **Documentation**: Technical docs, manuals, guides
- **FAQ**: Existing question-answer content
- **Tutorial**: Step-by-step instructions
- **Reference**: API docs, specifications
- **Conversational**: Chat logs, interviews
- **Mixed**: Multiple content types

### 7. Enhanced Data Pipeline

#### 7.1 Updated FineTuner Integration
```python
class FineTuner:
    def __init__(self, data_path: str, model: str = "auto", preset: str = "quality"):
        self.data_processor = self._get_processor()
        self.qa_generator = QAGenerator(groq_api_key)
        self.formatter = self._get_formatter(model)
    
    def process_documents(self) -> list[dict]:
        # Auto-detect file types and process accordingly
        # Generate Q&A pairs using Groq
        # Format for specific model
        # Return processed training data
    
    def _get_processor(self) -> BaseProcessor:
        # Return appropriate processor based on file types
    
    def _get_formatter(self, model: str) -> BaseFormatter:
        # Return model-specific formatter
```

### 8. Quality Assurance

#### 8.1 Data Quality Validation
```python
class DataQualityValidator:
    def validate_qa_pairs(self, qa_pairs: list[dict]) -> dict
    def check_answer_relevance(self, question: str, answer: str) -> float
    def detect_duplicate_content(self, qa_pairs: list[dict]) -> list[int]
    def assess_difficulty_distribution(self, qa_pairs: list[dict]) -> dict
    def validate_formatting(self, formatted_data: list[str]) -> bool
```

**Quality Metrics:**
- Answer relevance score
- Question diversity
- Difficulty distribution
- Format compliance
- Content coverage

### 9. Enhanced Error Handling

#### 9.1 Robust Error Management
- PDF processing failures (corrupted files, unsupported formats)
- Groq API rate limiting and failures
- Large document processing timeouts
- Memory management for big files
- Network connectivity issues

#### 9.2 User Feedback System
- Progress bars for long operations
- Quality warnings for generated content
- Suggestions for improving data quality
- Clear error messages with solutions

## Technical Requirements

### New Dependencies
```txt
# PDF Processing
PyPDF2>=3.0.0
pdfplumber>=0.9.0
pymupdf>=1.23.0

# Text Processing
nltk>=3.8.0
spacy>=3.7.0
sentence-transformers>=2.2.0

# API Integration
groq>=0.4.1
tenacity>=8.2.0  # For retry logic

# Content Analysis
textstat>=0.7.0
langdetect>=1.0.9
```

### Enhanced Environment Configuration
```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MAX_RETRIES=3
GROQ_TIMEOUT=30

# Data Processing
MAX_FILE_SIZE_MB=100
PRESERVE_STRUCTURE=true
GENERATE_QA_PAIRS=true
QA_PAIRS_PER_CHUNK=3

# Quality Settings
MIN_ANSWER_LENGTH=20
MAX_ANSWER_LENGTH=500
QUALITY_THRESHOLD=0.7
```

## Testing Strategy

### Unit Tests
- PDF text extraction accuracy
- Q&A generation quality
- Model-specific formatting
- Content type detection
- Chunking algorithms

### Integration Tests
- End-to-end document processing
- Multi-model pipeline testing
- Large document handling
- API integration reliability
- Error recovery scenarios

### Quality Assurance Tests
- Generated Q&A pair relevance
- Format compliance across models
- Performance with large documents
- Memory usage optimization
- API rate limiting handling

## Success Criteria

### Functional Requirements
- [ ] Successfully process PDF documents with structure preservation
- [ ] Generate high-quality Q&A pairs using Groq API
- [ ] Support all 4 models with proper formatting
- [ ] Handle documents up to 100MB efficiently
- [ ] Automatic content type detection working

### Performance Requirements
- [ ] Process 10MB PDF in < 2 minutes
- [ ] Generate Q&A pairs at 10+ pairs/minute
- [ ] Memory usage stays under 4GB for large documents
- [ ] API calls optimized with proper rate limiting

### Quality Requirements
- [ ] Generated Q&A pairs score >0.7 relevance
- [ ] Model-specific formatting 100% compliant
- [ ] Document structure preservation >90% accurate
- [ ] Error recovery success rate >95%

## Risk Mitigation

### Technical Risks
- **PDF complexity**: Test with various PDF types and layouts
- **Groq API limits**: Implement robust retry and fallback logic
- **Memory usage**: Implement streaming processing for large files
- **Model compatibility**: Extensive testing with all supported models

### Quality Risks
- **Q&A generation quality**: Implement quality validation and filtering
- **Format compliance**: Automated testing for all model formats
- **Content preservation**: Validation against original documents

## Next Phase Preparation

### Phase 3 Prerequisites
- Stable multi-model data processing pipeline
- Reliable Q&A generation system
- Comprehensive error handling
- Quality validation framework

### Technical Foundation for Phase 3
- Training monitoring hooks
- Checkpoint management system
- Advanced parameter tuning framework
- Performance optimization baseline

---

**Phase 2 Completion Target**: Robust data processing system that can handle PDFs and text files, generate high-quality training data for all 4 supported models, with intelligent content analysis and formatting.