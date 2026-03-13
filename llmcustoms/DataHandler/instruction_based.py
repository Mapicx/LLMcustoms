from datasets import Dataset, load_dataset
import os
import logging

logger = logging.getLogger(__name__)


class InstructionDataHandler:
    def __init__(self, dataset_path: str = None, dataset_name: str = None):
        """
        Initialize the instruction data handler.
        
        Args:
            dataset_path: Path to local dataset file (CSV, JSON, JSONL)
            dataset_name: HuggingFace dataset name (e.g., "databricks/databricks-dolly-15k")
        """
        if not dataset_path and not dataset_name:
            raise ValueError("Either dataset_path or dataset_name must be provided")
        
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        
        # Default field mapping (can be overridden with set_field_mapping)
        self.field_mapping = {
            "instruction": "instruction",
            "response": "response",
            "context": "context"
        }
        
        # Special tokens (can be set later)
        self.bos_token = None
        self.eos_token = None
    
    def load_dataset(self) -> Dataset:
        """
        Load dataset from local file or HuggingFace.
        
        Returns:
            Dataset: HuggingFace Dataset object
        """
        try:
            # Load from HuggingFace
            if self.dataset_name:
                logger.info(f"Loading dataset from HuggingFace: {self.dataset_name}")
                dataset = load_dataset(self.dataset_name)
                # Return the train split, or first available split
                if "train" in dataset:
                    return dataset["train"]
                else:
                    # Get first available split
                    first_split = list(dataset.keys())[0]
                    logger.warning(f"No 'train' split found, using '{first_split}' split")
                    return dataset[first_split]
            
            # Load from local file
            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
            
            ext = os.path.splitext(self.dataset_path)[1].lower()
            logger.info(f"Loading dataset from local file: {self.dataset_path}")
            
            if ext == ".csv":
                dataset = load_dataset("csv", data_files=self.dataset_path)
            elif ext == ".json":
                dataset = load_dataset("json", data_files=self.dataset_path)
            elif ext == ".jsonl":
                dataset = load_dataset("json", data_files=self.dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}. Supported formats: .csv, .json, .jsonl")
            
            return dataset["train"]
        
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def set_field_mapping(self, instruction_field: str, response_field: str, context_field: str = None):
        """
        Set custom field names for the dataset.
        
        Args:
            instruction_field: Name of the instruction column
            response_field: Name of the response column
            context_field: Name of the context column (optional)
        """
        self.field_mapping = {
            "instruction": instruction_field,
            "response": response_field,
            "context": context_field
        }
        logger.info(f"Field mapping updated: {self.field_mapping}")
    
    def validate_dataset(self, dataset: Dataset) -> bool:
        """
        Validate that the dataset has required fields and quality.
        
        Args:
            dataset: HuggingFace Dataset to validate
            
        Returns:
            bool: True if valid
            
        Raises:
            ValueError: If validation fails
        """
        instruction_field = self.field_mapping["instruction"]
        response_field = self.field_mapping["response"]
        context_field = self.field_mapping.get("context")

        # Check required fields exist
        if instruction_field not in dataset.column_names:
            raise ValueError(f"Missing instruction field: '{instruction_field}'. Available fields: {dataset.column_names}")

        if response_field not in dataset.column_names:
            raise ValueError(f"Missing response field: '{response_field}'. Available fields: {dataset.column_names}")

        # Context is optional, so just warn if specified but missing
        if context_field and context_field not in dataset.column_names:
            logger.warning(f"Context field '{context_field}' not found in dataset. Will proceed without context.")
            self.field_mapping["context"] = None

        # Quality checks
        empty_instructions = 0
        empty_responses = 0
        too_short = 0
        too_long = 0
        
        total_records = len(dataset)
        sample_size = min(1000, total_records)  # Check first 1000 records for speed

        for i, row in enumerate(dataset.select(range(sample_size))):
            instruction = row.get(instruction_field, "")
            response = row.get(response_field, "")

            if not instruction or str(instruction).strip() == "":
                empty_instructions += 1

            if not response or str(response).strip() == "":
                empty_responses += 1

            if len(str(instruction)) < 3 or len(str(response)) < 5:
                too_short += 1

            if len(str(instruction)) > 10000 or len(str(response)) > 10000:
                too_long += 1

        # Report issues
        if empty_instructions > 0:
            logger.warning(f"Found {empty_instructions} empty instructions in sample of {sample_size}")
        
        if empty_responses > 0:
            raise ValueError(f"Found {empty_responses} empty responses in sample of {sample_size}. Dataset is invalid.")

        if too_short > sample_size * 0.1:  # More than 10% too short
            logger.warning(f"Found {too_short} entries that are very short (may be low quality)")

        if too_long > 0:
            logger.warning(f"Found {too_long} entries that are very long (>10k chars)")

        logger.info(f"Dataset validation passed. Total records: {total_records}")
        return True


    def format_prompt(self, instruction: str, context: str = None, prompt_template: str = None) -> str:
        """
        Format instruction and context into a prompt using a template.
        
        Args:
            instruction: The instruction text
            context: Optional context text
            prompt_template: Template name or custom template string
            
        Returns:
            str: Formatted prompt
        """
        # Handle None or empty context
        context = context if context and str(context).strip() else ""
        
        # Alpaca template (default)
        if prompt_template is None or prompt_template == "alpaca":
            if context:
                prompt = (
                    "Below is an instruction that describes a task, paired with an input that "
                    "provides further context. Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n"
                    f"{instruction}\n\n"
                    "### Input:\n"
                    f"{context}\n\n"
                    "### Response:\n"
                )
            else:
                prompt = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n"
                    f"{instruction}\n\n"
                    "### Response:\n"
                )

        # Simple template
        elif prompt_template == "simple":
            if context:
                prompt = f"{instruction}\n\n{context}\n\n"
            else:
                prompt = f"{instruction}\n\n"

        # Vicuna template
        elif prompt_template == "vicuna":
            if context:
                prompt = f"USER: {instruction}\n\n{context}\n\nASSISTANT: "
            else:
                prompt = f"USER: {instruction}\n\nASSISTANT: "

        # Custom template
        else:
            try:
                prompt = prompt_template.format(
                    instruction=instruction,
                    context=context
                )
            except KeyError as e:
                raise ValueError(f"Invalid template. Missing placeholder: {e}")

        return prompt


    def prepare_training_data(
        self,
        dataset: Dataset,
        prompt_template: str = "alpaca",
        mask_instruction: bool = True
    ) -> Dataset:
        """
        Prepare dataset for training by formatting prompts and responses.
        
        Args:
            dataset: HuggingFace Dataset
            prompt_template: Template to use for formatting
            mask_instruction: Whether to mark instruction for masking (not implemented yet)
            
        Returns:
            Dataset: Formatted dataset with 'text' field
        """
        instruction_field = self.field_mapping["instruction"]
        response_field = self.field_mapping["response"]
        context_field = self.field_mapping.get("context")

        def format_example(example):
            instruction = example[instruction_field]
            response = example[response_field]

            # Get context if available
            context = None
            if context_field and context_field in example:
                context = example.get(context_field)

            # Format the prompt
            prompt = self.format_prompt(
                instruction=instruction,
                context=context,
                prompt_template=prompt_template
            )

            # Combine prompt and response
            text = prompt + str(response)

            # Add EOS token if available
            if self.eos_token:
                text += self.eos_token

            # Add BOS token if available
            if self.bos_token:
                text = self.bos_token + text

            return {"text": text}

        # Map the formatting function to all examples
        formatted_dataset = dataset.map(format_example, remove_columns=dataset.column_names)
        
        logger.info(f"Prepared {len(formatted_dataset)} training examples")
        return formatted_dataset
    
    def get_statistics(self, dataset: Dataset) -> dict:
        """
        Get statistics about the dataset.
        
        Args:
            dataset: HuggingFace Dataset
            
        Returns:
            dict: Statistics about the dataset
        """
        instruction_field = self.field_mapping["instruction"]
        response_field = self.field_mapping["response"]
        context_field = self.field_mapping.get("context")
        
        total_records = len(dataset)
        
        # Calculate lengths
        instruction_lengths = []
        response_lengths = []
        context_lengths = []
        
        # Sample for speed (check first 1000 records)
        sample_size = min(1000, total_records)
        
        for row in dataset.select(range(sample_size)):
            instruction = str(row.get(instruction_field, ""))
            response = str(row.get(response_field, ""))
            
            instruction_lengths.append(len(instruction))
            response_lengths.append(len(response))
            
            if context_field and context_field in row:
                context = str(row.get(context_field, ""))
                context_lengths.append(len(context))
        
        stats = {
            "total_records": total_records,
            "avg_instruction_length": sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
            "avg_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            "avg_context_length": sum(context_lengths) / len(context_lengths) if context_lengths else 0,
            "min_instruction_length": min(instruction_lengths) if instruction_lengths else 0,
            "max_instruction_length": max(instruction_lengths) if instruction_lengths else 0,
            "min_response_length": min(response_lengths) if response_lengths else 0,
            "max_response_length": max(response_lengths) if response_lengths else 0,
        }
        
        # Add category distribution if available
        if "category" in dataset.column_names:
            categories = {}
            for row in dataset.select(range(sample_size)):
                cat = row.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            stats["category_distribution"] = categories
        
        return stats