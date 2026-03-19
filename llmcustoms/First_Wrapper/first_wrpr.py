import os
from ..core.hardware_detector_new import HardwareDetector
from ..models.BaseModel import BaseModel
from ..models.gemma_9b import Gemma
from ..models.qwen_7b import Qwen
from ..models.Tinyllama import TinyLlama
from ..models.phi_35 import Phi
from ..core.model_selector import ModelSelector
from ..DataHandler.instruction_based import InstructionDataHandler
from ..DataHandler.chat_based import ChatDataHandler

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from ..utils.config import settings


def _patch_rope_scaling():
    """
    Patch Phi3Config._rope_scaling_validation to accept 'longrope' by
    remapping it to 'yarn' (fallback: 'su'). Included here so users who
    clone the repo without the venv sitecustomize.py still get the fix.
    """
    try:
        from transformers.models.phi3.configuration_phi3 import Phi3Config

        if getattr(Phi3Config._rope_scaling_validation, "_patched", False):
            return  # already patched

        _original = Phi3Config._rope_scaling_validation

        def _patched(self):
            rs = getattr(self, "rope_scaling", None)
            if isinstance(rs, dict) and rs.get("type") not in ("su", "yarn"):
                for rope_type in ("yarn", "su"):
                    self.rope_scaling = {**rs, "type": rope_type}
                    try:
                        return _original(self)
                    except ValueError:
                        continue
                self.rope_scaling = rs  # restore if both fail
            return _original(self)

        _patched._patched = True
        Phi3Config._rope_scaling_validation = _patched

    except Exception:
        pass


_patch_rope_scaling()


class FineTuner:

    SUPPORTED_PRESETS = ["highspeed", "balanced", "highquality"]

    MODEL_MAP = {
        'gemma': Gemma,
        'qwen': Qwen,
        'tinyllama': TinyLlama,
        'phi': Phi
    }

    def __init__(
        self,
        data_path: str = None,
        model: str = 'auto',
        preset: str = 'auto',
        training_mode: str = 'instruction',  # 'instruction', 'completion', 'chat'
        prompt_template: str = 'alpaca',      # template for instruction formatting
        mask_instruction: bool = True,        # whether to mask instruction during training
        dataset_name: str = None,             # HuggingFace dataset name (alternative to data_path)
        max_steps: int = None                 # override preset max_steps (useful for testing)
    ):
        # Validate that either data_path or dataset_name is provided
        if not data_path and not dataset_name:
            raise ValueError("Either data_path or dataset_name must be provided")

        # Validate data_path if provided
        if data_path and not dataset_name:
            if not os.path.exists(data_path):
                raise ValueError(f"Data path does not exist: {data_path}")

            if not os.path.isfile(data_path):
                raise ValueError(f"Expected a file but got something else: {data_path}")

        self.data_path = data_path
        self.dataset_name = dataset_name
        self.training_mode = training_mode
        self.prompt_template = prompt_template
        self.mask_instruction = mask_instruction
        self.max_steps_override = max_steps

        # Validate training mode
        if training_mode not in ['instruction', 'completion', 'chat']:
            raise ValueError(
                f"Unsupported training_mode '{training_mode}'. "
                f"Choose from ['instruction', 'completion', 'chat']"
            )

        # Validate inputs
        if model != 'auto' and model.lower() not in self.MODEL_MAP:
            raise ValueError(
                f"Unsupported model '{model}'. "
                f"Choose from {list(self.MODEL_MAP.keys())} or 'auto'"
            )

        if preset != 'auto' and preset not in self.SUPPORTED_PRESETS:
            raise ValueError(
                f"Unsupported preset '{preset}'. "
                f"Choose from {self.SUPPORTED_PRESETS} or 'auto'"
            )

        # Initialize model selector
        modelselector = ModelSelector()

        # Select model and preset
        if model == 'auto' and preset != 'auto':
            supported_models = [cls() for cls in self.MODEL_MAP.values()]
            self.model = modelselector.select_best_model(supported_models, preset)
            self.preset = preset

        elif model != 'auto' and preset == 'auto':
            model_instance = self.MODEL_MAP[model.lower()]()
            self.model = model_instance
            self.preset = modelselector.select_best_preset(model_instance)

        elif model == 'auto' and preset == 'auto':
            model_preset = modelselector.select_best_model_and_preset()
            self.model = list(model_preset.keys())[0]
            self.preset = list(model_preset.values())[0]

        else:
            self.model = self.MODEL_MAP[model.lower()]()
            self.preset = preset

        # Initialize data handler based on training mode
        if self.training_mode == 'instruction':
            self.data_handler = InstructionDataHandler(
                dataset_path=self.data_path,
                dataset_name=self.dataset_name
            )
        elif self.training_mode == 'completion':
            # TODO: Implement completion handler
            raise NotImplementedError("Completion mode not yet implemented")
        elif self.training_mode == 'chat':
            self.data_handler = ChatDataHandler(
                dataset_path=self.data_path,
                dataset_name=self.dataset_name
            )

    def train(self):
        # Login to HuggingFace if token is available
        if settings.hf_token:
            login(token=settings.hf_token)

        def get_torch_dtype(dtype_str: str):
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            return dtype_map.get(dtype_str.lower(), torch.float16)

        from transformers import BitsAndBytesConfig

        loading_config = self.model.get_loading_config(self.preset)
        training_config = self.model.get_training_config(self.preset)
        lora_config = self.model.get_lora_config(self.preset)

        if loading_config['LOAD_IN_8BIT']:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=loading_config.get('LLM_INT8_THRESHOLD', 6.0)
            )
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=get_torch_dtype(loading_config['BNB_4BIT_COMPUTE_DTYPE']),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=loading_config['BNB_4BIT_QUANT_TYPE']
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model.model_key,
            quantization_config=bnb_config,
            device_map={"": 0},  # Force everything onto GPU
            torch_dtype=get_torch_dtype(loading_config['BNB_4BIT_COMPUTE_DTYPE']),
            attn_implementation=loading_config['attn_implementation']
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model.model_key)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        """## Freezing original weights"""

        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        class CastOutputToFloat(nn.Sequential):
            def forward(self, x):
                return super().forward(x).to(torch.float32)

        model.lm_head = CastOutputToFloat(model.lm_head)

        """## Setting up the LoRa Adapters"""

        def print_trainable_parameters(model):
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(
                f"trainable params: {trainable_params} || "
                f"all params: {all_param} || "
                f"trainable%: {100 * trainable_params / all_param}"
            )

        for name, module in model.named_modules():
            if 'attn' in name or 'attention' in name:
                print(name)
                for sub_name, sub_module in module.named_modules():
                    print(f"  - {sub_name}")

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=lora_config['LORA_R'],
            lora_alpha=lora_config['LORA_ALPHA'],
            target_modules=lora_config['LORA_TARGET_MODULES'],
            lora_dropout=lora_config['LORA_DROPOUT'],
            bias="none",
            task_type="CAUSAL_LM"
        )

        lora_model = get_peft_model(model, config)
        print_trainable_parameters(model)
        print_trainable_parameters(lora_model)

        """## Training data import"""

        import transformers

        # Load and prepare dataset using data handler
        print("=" * 60)
        print("Loading and preparing dataset...")
        print("=" * 60)

        # Set special tokens for the data handler
        self.data_handler.bos_token = tokenizer.bos_token if hasattr(tokenizer, 'bos_token') else None
        self.data_handler.eos_token = tokenizer.eos_token

        # Load dataset
        raw_dataset = self.data_handler.load_dataset()
        print(f"Loaded {len(raw_dataset)} records")

        # Validate dataset
        self.data_handler.validate_dataset(raw_dataset)
        print("Dataset validation passed")

        # Get and print statistics
        stats = self.data_handler.get_statistics(raw_dataset)
        print("\nDataset Statistics:")
        print(f"  Total records: {stats['total_records']}")
        if self.training_mode == 'chat':
            print(f"  Avg turns: {stats['avg_turns']:.1f}")
            print(f"  Avg message length: {stats['avg_message_length']:.1f} chars")
        else:
            print(f"  Avg instruction length: {stats['avg_instruction_length']:.1f} chars")
            print(f"  Avg response length: {stats['avg_response_length']:.1f} chars")
            if stats['avg_context_length'] > 0:
                print(f"  Avg context length: {stats['avg_context_length']:.1f} chars")
            if 'category_distribution' in stats:
                print(f"  Categories: {stats['category_distribution']}")
        print()

        # Prepare training data
        if self.training_mode == 'chat':
            formatted_dataset = self.data_handler.prepare_training_data(
                dataset=raw_dataset,
                tokenizer=tokenizer,
                mask_user_messages=self.mask_instruction
            )
        else:
            formatted_dataset = self.data_handler.prepare_training_data(
                dataset=raw_dataset,
                prompt_template=self.prompt_template,
                mask_instruction=self.mask_instruction
            )
        print(f"Formatted {len(formatted_dataset)} training examples")

        # Tokenize the formatted data
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=training_config['MAX_LENGTH']
            )

        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names
        )

        print(f"Tokenized {len(tokenized_dataset)} examples")
        print("=" * 60)

        """## Training"""

        trainer = transformers.Trainer(
            model=lora_model,
            train_dataset=tokenized_dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=training_config['BATCH_SIZE'],
                gradient_accumulation_steps=training_config['GRAD_ACCUM_STEPS'],
                warmup_steps=10,
                max_steps=self.max_steps_override or training_config['MAX_STEPS'],
                learning_rate=training_config['LEARNING_RATE'],
                fp16=True,
                logging_steps=1,
                output_dir=settings.output_dir
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )

        model.config.use_cache = False
        trainer.train()

        """SAVE FINE-TUNED LORA ADAPTER LOCALLY"""

        # Use model's display name for the save path
        model_name = self.model.display_name.replace('/', '-').replace('\\', '-')
        save_path = os.path.join(settings.output_dir, f"{model_name}-{self.preset}-lora")

        lora_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print("=" * 60)
        print(f"MODEL SAVED AT: {save_path}")
        print("=" * 60)

        return save_path
