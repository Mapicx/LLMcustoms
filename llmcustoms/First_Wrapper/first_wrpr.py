import os
from ..core.hardware_detector import HardwareDetector

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from huggingface_hub import login
from ..utils.config import settings

class FineTuner:
    SUPPORTED_MODELS = {
        "phi35b": "phi 3.5b",
        "gemma9b": "gemma 9b",
        "tinyllama11b": "tinyllama 1.1b"
    }

    SUPPORTED_PRESETS = ["highspeed", "balanced", "highquality"]

    def __init__(self, data_path: str, model: str, preset: str):
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")

        if not os.path.isfile(data_path):
            raise ValueError(f"Expected a file but got something else: {data_path}")

        detector = HardwareDetector()
        model_preset_dict = detector.get_runnable_models()

        
        if model == "auto":
            normalized_model = model_preset_dict["model"]
        else:
            normalized_model = (
                model.replace("-", "")
                     .replace(" ", "")
                     .replace(".", "")
                     .lower()
                     .strip()
            )

            if normalized_model not in self.SUPPORTED_MODELS:
                raise ValueError(
                    f"Unsupported model '{model}'. Supported models are: "
                    f"{', '.join(self.SUPPORTED_MODELS.values())}"
                )

       
        if model == "auto":
            self.model = model_preset_dict["model"]
        else:
            self.model = normalized_model

        
        if preset == "auto" or preset not in self.SUPPORTED_PRESETS:
            self.preset = model_preset_dict["preset"]
        else:
            self.preset = preset

        self.data_path = data_path

        self.recommended_model = model_preset_dict["model"]
        self.recommended_preset = model_preset_dict["preset"]

    def train(self):

        def get_torch_dtype(dtype_str: str):
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            return dtype_map.get(dtype_str.lower(), torch.float16)


        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=settings.load_in_4bit,
            bnb_4bit_compute_dtype=get_torch_dtype(settings.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=settings.bnb_4bit_quant_type
        )

        model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            quantization_config=bnb_config,
            device_map={"": 0},  #Force everything onto GPU
            torch_dtype=get_torch_dtype(settings.bnb_4bit_compute_dtype),
            attn_implementation=settings.attn_implementation
        )


        tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

        # Gemma requires a pad token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # doesnt work with 4 bit quantized model since it is very unstable while training and model is splitted on CPU + GPU
        # input_text = "Write me a poem about Machine Learning."
        # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

        # outputs = model.generate(**input_ids, max_new_tokens=320, use_cache=False)
        # print(tokenizer.decode(outputs[0]))

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
            r = settings.lora_r,
            lora_alpha = settings.lora_alpha,
            target_modules = settings.lora_target_modules,
            lora_dropout = settings.lora_dropout,
            bias = "none",
            task_type = "CAUSAL_LM"
        )

        lora_model = get_peft_model(model, config)
        print_trainable_parameters(model)
        print_trainable_parameters(lora_model)

        """## Training data import"""

        import transformers
        from datasets import load_dataset, load_from_disk

        if settings.dataset_path:
            # Local dataset (csv/json/text or HF saved dataset)
            if settings.dataset_path.endswith(".csv"):
                data = load_dataset("csv", data_files=settings.dataset_path)
            elif settings.dataset_path.endswith(".json"):
                data = load_dataset("json", data_files=settings.dataset_path)
            else:
                data = load_from_disk(settings.dataset_path)
        else:
            # Hugging Face dataset
            data = load_dataset(settings.dataset_name)


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
                per_device_train_batch_size = settings.batch_size,
                gradient_accumulation_steps = settings.grad_accum_steps,
                warmup_steps = 10,
                max_steps = settings.max_steps, # defines number of - (Forward pass + backward pass + weights update) means 30 times all this is done
                learning_rate = settings.learning_rate,
                fp16 = True,
                logging_steps = 1,
                output_dir = settings.output_dir
            ),
            data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False)
        )

        model.config.use_cache = False
        trainer.train()

        """SAVE FINE-TUNED LORA ADAPTER LOCALLY"""

        save_path = os.path.join(settings.output_dir, f"{self.model}-{self.preset}-lora")

        lora_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print("=" * 60)
        print(f"MODEL SAVED AT: {save_path}")
        print("=" * 60)

        return save_path

        

        