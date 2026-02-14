def get_preset(name: str) -> dict:
    name = name.lower()

    presets = {
        "highspeed": {
            "MAX_LENGTH": 64,

            "LOAD_IN_4BIT": True,
            "LOAD_IN_8BIT": False,
            "BNB_4BIT_COMPUTE_DTYPE": "float16",
            "BNB_4BIT_QUANT_TYPE": "nf4",

            "LORA_R": 8,
            "LORA_ALPHA": 16,
            "LORA_DROPOUT": 0.1,
            "LORA_TARGET_MODULES": ["q_proj", "v_proj"],

            "BATCH_SIZE": 1,
            "GRAD_ACCUM_STEPS": 4,
            "LEARNING_RATE": 2e-4,
            "MAX_STEPS": 80,

            "ATTN_IMPLEMENTATION": "auto",
        },

        "balanced": {
            "MAX_LENGTH": 96,

            "LOAD_IN_4BIT": True,
            "LOAD_IN_8BIT": False,
            "BNB_4BIT_COMPUTE_DTYPE": "bfloat16",
            "BNB_4BIT_QUANT_TYPE": "nf4",

            "LORA_R": 16,
            "LORA_ALPHA": 32,
            "LORA_DROPOUT": 0.05,
            "LORA_TARGET_MODULES": ["q_proj", "k_proj", "v_proj", "o_proj"],

            "BATCH_SIZE": 2,
            "GRAD_ACCUM_STEPS": 4,
            "LEARNING_RATE": 1e-4,
            "MAX_STEPS": 120,

            "ATTN_IMPLEMENTATION": "auto",
        },

        "highquality": {
            "MAX_LENGTH": 128,

            "LOAD_IN_4BIT": False,
            "LOAD_IN_8BIT": True,

            "LORA_R": 32,
            "LORA_ALPHA": 64,
            "LORA_DROPOUT": 0.03,
            "LORA_TARGET_MODULES": ["q_proj", "k_proj", "v_proj", "o_proj"],

            "BATCH_SIZE": 2,
            "GRAD_ACCUM_STEPS": 8,
            "LEARNING_RATE": 8e-5,
            "MAX_STEPS": 200,

            "ATTN_IMPLEMENTATION": "auto",
        }
    }

    if name not in presets:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {list(presets.keys())}")

    return presets[name]
