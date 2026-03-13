"""
Example script showing how to use instruction-based training with LLMCustoms.
"""

from llmcustoms.First_Wrapper.first_wrpr import FineTuner

# Example 1: Train with Dolly-15k dataset from HuggingFace
print("=" * 60)
print("Example 1: Training with Dolly-15k from HuggingFace")
print("=" * 60)

tuner = FineTuner(
    data_path=None,  # Not using local file
    dataset_name="databricks/databricks-dolly-15k",  # HuggingFace dataset
    model="auto",  # Auto-select best model for your hardware
    preset="auto",  # Auto-select best preset
    training_mode="instruction",  # Instruction-based training
    prompt_template="alpaca",  # Use Alpaca prompt format
    mask_instruction=True  # Only train on responses
)

# Start training
model_path = tuner.train()
print(f"\nModel saved at: {model_path}")

# Example 2: Train with local CSV file
print("\n" + "=" * 60)
print("Example 2: Training with local CSV file")
print("=" * 60)

# Your CSV should have columns: instruction, response, context (optional)
tuner2 = FineTuner(
    data_path="./my_instructions.csv",
    model="phi",  # Specific model
    preset="balanced",  # Specific preset
    training_mode="instruction",
    prompt_template="simple"  # Use simple prompt format
)

model_path2 = tuner2.train()
print(f"\nModel saved at: {model_path2}")

# Example 3: Custom field mapping
print("\n" + "=" * 60)
print("Example 3: Custom field mapping")
print("=" * 60)

tuner3 = FineTuner(
    data_path="./custom_data.json",
    model="gemma",
    preset="highspeed",
    training_mode="instruction"
)

# If your dataset has different field names
tuner3.data_handler.set_field_mapping(
    instruction_field="question",  # Your dataset uses "question" instead of "instruction"
    response_field="answer",  # Your dataset uses "answer" instead of "response"
    context_field="background"  # Your dataset uses "background" instead of "context"
)

model_path3 = tuner3.train()
print(f"\nModel saved at: {model_path3}")

# Example 4: Different prompt templates
print("\n" + "=" * 60)
print("Example 4: Using different prompt templates")
print("=" * 60)

# Alpaca template (default)
tuner_alpaca = FineTuner(
    dataset_name="databricks/databricks-dolly-15k",
    model="auto",
    preset="auto",
    prompt_template="alpaca"
)

# Simple template
tuner_simple = FineTuner(
    dataset_name="databricks/databricks-dolly-15k",
    model="auto",
    preset="auto",
    prompt_template="simple"
)

# Vicuna template
tuner_vicuna = FineTuner(
    dataset_name="databricks/databricks-dolly-15k",
    model="auto",
    preset="auto",
    prompt_template="vicuna"
)

# Custom template
custom_template = """Question: {instruction}

Context: {context}

Answer: """

tuner_custom = FineTuner(
    dataset_name="databricks/databricks-dolly-15k",
    model="auto",
    preset="auto",
    prompt_template=custom_template
)

print("\nAll examples configured successfully!")
print("Uncomment the train() calls to actually train the models.")
