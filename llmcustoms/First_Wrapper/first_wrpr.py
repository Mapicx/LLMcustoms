import os
from ..core.hardware_detector_new import HardwareDetector
from ..models.BaseModel import BaseModel
from ..models.gemma_9b import Gemma
from ..models.qwen_7b import Qwen
from ..models.Tinyllama import TinyLlama
from ..models.phi_35 import Phi
from ..core.model_selector import ModelSelector

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from huggingface_hub import login
from ..utils.config import settings

class FineTuner:

    SUPPORTED_PRESETS = ["highspeed", "balanced", "highquality"]
    
    MODEL_MAP = {
        'gemma': Gemma,
        'qwen': Qwen,
        'tinyllama': TinyLlama,
        'phi': Phi
    }

    def __init__(self, data_path: str, model: str = 'auto', preset: str = 'auto'):
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")
        
        if not os.path.isfile(data_path):
            raise ValueError(f"Expected a file but got something else: {data_path}")
        
        self.data_path = data_path
        
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
        
        modelselector = ModelSelector()
        
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
            device_map={"": 0},  #Force everything onto GPU
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
            print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

        for name, module in model.named_modules():
            if 'attn' in name or 'attention' in name:
                print(name)
                for sub_name, sub_module in module.named_modules():
                    print(f"  - {sub_name}")

        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r = lora_config['LORA_R'],
            lora_alpha = lora_config['LORA_ALPHA'],
            target_modules = lora_config['LORA_TARGET_MODULES'],
            lora_dropout = lora_config['LORA_DROPOUT'],
            bias = "none",
            task_type = "CAUSAL_LM"
        )

        lora_model = get_peft_model(model, config)
        print_trainable_parameters(model)
        print_trainable_parameters(lora_model)

        """## Training data import"""

        import transformers
        from datasets import load_dataset, load_from_disk

        # Use the data_path from constructor instead of settings
        if self.data_path.endswith(".csv"):
            data = load_dataset("csv", data_files=self.data_path)
        elif self.data_path.endswith(".json"):
            data = load_dataset("json", data_files=self.data_path)
        elif os.path.isdir(self.data_path):
            # If it's a directory, assume it's a saved HF dataset
            data = load_from_disk(self.data_path)
        else:
            # Fallback: try to load as HuggingFace dataset name
            data = load_dataset(self.data_path)


        def merge_columns(example):
            text = example[settings.dataset_text_field]

            # If a label/target column exists, append it
            if settings.dataset_label_field and settings.dataset_label_field in example:
                label = example[settings.dataset_label_field]
                example["text"] = f"{text} ->: {label}"
            else:
                example["text"] = text

            return example

        data[settings.dataset_split] = data[settings.dataset_split].map(merge_columns)


        data = data.map(
            lambda samples: tokenizer(
                samples["text"],
                truncation=True,
                padding="max_length",
                max_length=settings.max_length
            ),
            batched=True
        )

        """## Training"""

        trainer = transformers.Trainer(
            model = lora_model,
            train_dataset = data[settings.dataset_split],
            args = transformers.TrainingArguments(
                per_device_train_batch_size = training_config['BATCH_SIZE'],
                gradient_accumulation_steps = training_config['GRAD_ACCUM_STEPS'],
                warmup_steps = 10,
                max_steps = training_config['MAX_STEPS'], # defines number of - (Forward pass + backward pass + weights update) means 30 times all this is done
                learning_rate = training_config['LEARNING_RATE'],
                fp16 = True,
                logging_steps = 1,
                output_dir = settings.output_dir
            ),
            data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False)
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

        

        