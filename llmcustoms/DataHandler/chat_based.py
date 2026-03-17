from datasets import Dataset, load_dataset
import os
import logging

logger = logging.getLogger(__name__)


class ChatDataHandler:
    def __init__(self, dataset_path: str = None, dataset_name: str = None, format: str = "auto"):
        """
        Initialize the chat data handler.

        Args:
            dataset_path: Path to local dataset file (JSON, JSONL)
            dataset_name: HuggingFace dataset name
            format: Conversation format (sharegpt, openai, simple, auto)
        """
        if not dataset_path and not dataset_name:
            raise ValueError("Either dataset_path or dataset_name must be provided")

        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

        supported_formats = ["sharegpt", "openai", "simple", "auto"]
        format = format.lower().strip()

        if format not in supported_formats:
            raise ValueError(f"Unsupported format '{format}'. Supported formats: {supported_formats}")

        self.format = format

        if format == "sharegpt":
            self.field_mapping = {"conversation_field": "conversations", "role_field": "from", "content_field": "value"}
        elif format == "openai":
            self.field_mapping = {"conversation_field": "messages", "role_field": "role", "content_field": "content"}
        elif format == "simple":
            self.field_mapping = {"conversation_field": "conversation", "role_field": "speaker", "content_field": "text"}
        else:
            self.field_mapping = {"conversation_field": None, "role_field": None, "content_field": None}

        self.bos_token = None
        self.eos_token = None

    def load_dataset(self) -> Dataset:
        """Load dataset from local file or HuggingFace."""
        try:
            if self.dataset_name:
                logger.info(f"Loading dataset from HuggingFace: {self.dataset_name}")
                dataset = load_dataset(self.dataset_name)
                if "train" in dataset:
                    return dataset["train"]
                first_split = list(dataset.keys())[0]
                logger.warning(f"No 'train' split found, using '{first_split}' split")
                return dataset[first_split]

            if not os.path.exists(self.dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

            ext = os.path.splitext(self.dataset_path)[1].lower()
            logger.info(f"Loading dataset from local file: {self.dataset_path}")

            if ext in (".json", ".jsonl"):
                dataset = load_dataset("json", data_files=self.dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}. Supported: .json, .jsonl")

            return dataset["train"]

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def detect_format(self, sample_record: dict) -> str:
        """Auto-detect conversation format from a sample record."""
        if not isinstance(sample_record, dict):
            raise ValueError("Sample record must be a dictionary")

        if "conversations" in sample_record:
            conv = sample_record["conversations"]
            if isinstance(conv, list) and len(conv) > 0:
                first_msg = conv[0]
                if isinstance(first_msg, dict) and "from" in first_msg and "value" in first_msg:
                    self.field_mapping = {"conversation_field": "conversations", "role_field": "from", "content_field": "value"}
                    self.format = "sharegpt"
                    return "sharegpt"

        if "messages" in sample_record:
            conv = sample_record["messages"]
            if isinstance(conv, list) and len(conv) > 0:
                first_msg = conv[0]
                if isinstance(first_msg, dict) and "role" in first_msg and "content" in first_msg:
                    self.field_mapping = {"conversation_field": "messages", "role_field": "role", "content_field": "content"}
                    self.format = "openai"
                    return "openai"

        if "conversation" in sample_record:
            conv = sample_record["conversation"]
            if isinstance(conv, list) and len(conv) > 0:
                first_msg = conv[0]
                if isinstance(first_msg, dict) and "speaker" in first_msg and "text" in first_msg:
                    self.field_mapping = {"conversation_field": "conversation", "role_field": "speaker", "content_field": "text"}
                    self.format = "simple"
                    return "simple"

        raise ValueError(f"Could not detect dataset format. Available keys: {list(sample_record.keys())}")

    def set_field_mapping(self, conversation_field: str, role_field: str, content_field: str):
        """Override field mapping manually."""
        self.field_mapping = {
            "conversation_field": conversation_field,
            "role_field": role_field,
            "content_field": content_field
        }
        logger.info(f"Field mapping updated: {self.field_mapping}")

    def parse_conversation(self, record: dict) -> list:
        """
        Normalize a raw record into a list of {"role": ..., "content": ...} dicts.
        Maps human/gpt roles to user/assistant.
        """
        conv_field = self.field_mapping["conversation_field"]
        role_field = self.field_mapping["role_field"]
        content_field = self.field_mapping["content_field"]

        raw_conv = record.get(conv_field, [])
        if not isinstance(raw_conv, list):
            return []

        role_map = {
            "human": "user",
            "gpt": "assistant",
            "bot": "assistant",
            "system": "system",
            "user": "user",
            "assistant": "assistant"
        }

        parsed = []
        for turn in raw_conv:
            role = str(turn.get(role_field, "")).lower()
            content = str(turn.get(content_field, "")).strip()
            normalized_role = role_map.get(role, role)
            parsed.append({"role": normalized_role, "content": content})

        return parsed

    def validate_conversation_quality(self, conversation: list) -> bool:
        """
        Check that a conversation has alternating speakers and no empty messages.
        Returns True if valid, False otherwise.
        """
        if not conversation or len(conversation) < 2:
            return False

        for turn in conversation:
            if not turn.get("content", "").strip():
                return False

        # Check for at least one user and one assistant turn
        roles = [t["role"] for t in conversation]
        if "user" not in roles and "human" not in roles:
            return False
        if "assistant" not in roles and "gpt" not in roles:
            return False

        return True

    def validate_dataset(self, dataset: Dataset) -> bool:
        """
        Validate that the dataset has required fields and conversation quality.
        Auto-detects format if needed.
        """
        # Auto-detect format if not set
        if self.format == "auto" or self.field_mapping["conversation_field"] is None:
            sample = dataset[0]
            self.detect_format(sample)

        conv_field = self.field_mapping["conversation_field"]

        if conv_field not in dataset.column_names:
            raise ValueError(f"Missing conversation field: '{conv_field}'. Available: {dataset.column_names}")

        sample_size = min(500, len(dataset))
        invalid_count = 0

        for row in dataset.select(range(sample_size)):
            conv = self.parse_conversation(row)
            if not self.validate_conversation_quality(conv):
                invalid_count += 1

        if invalid_count > sample_size * 0.2:
            raise ValueError(f"{invalid_count}/{sample_size} conversations failed quality check (>20% invalid)")

        if invalid_count > 0:
            logger.warning(f"{invalid_count}/{sample_size} conversations failed quality check")

        logger.info(f"Dataset validation passed. Total records: {len(dataset)}")
        return True

    def apply_chat_template(self, conversation: list, tokenizer) -> str:
        """
        Apply chat template using tokenizer's built-in template, or fall back to manual formatting.
        """
        # Try tokenizer's built-in chat template
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            try:
                return tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            except Exception as e:
                logger.warning(f"Tokenizer chat template failed: {e}. Falling back to manual formatting.")

        # Manual fallback
        text = ""
        if self.bos_token:
            text += self.bos_token

        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            if role == "system":
                text += f"System: {content}\n\n"
            elif role == "user":
                text += f"User: {content}\n"
            elif role == "assistant":
                text += f"Assistant: {content}\n"
                if self.eos_token:
                    text += self.eos_token
            text += "\n"

        return text.strip()

    def prepare_training_data(
        self,
        dataset: Dataset,
        tokenizer,
        max_turns: int = None,
        mask_user_messages: bool = True
    ) -> Dataset:
        """
        Format all conversations into training-ready text.

        Args:
            dataset: HuggingFace Dataset
            tokenizer: Tokenizer for applying chat template
            max_turns: Max number of turns to keep per conversation (None = keep all)
            mask_user_messages: Whether to note user turns for loss masking
        """
        def format_example(example):
            conv = self.parse_conversation(example)

            if not conv:
                return {"text": ""}

            # Trim to max_turns if specified
            if max_turns is not None:
                conv = conv[:max_turns]

            text = self.apply_chat_template(conv, tokenizer)
            return {"text": text}

        formatted = dataset.map(format_example, remove_columns=dataset.column_names)

        # Filter out empty examples
        formatted = formatted.filter(lambda x: len(x["text"].strip()) > 0)

        logger.info(f"Prepared {len(formatted)} training examples")
        return formatted

    def get_statistics(self, dataset: Dataset) -> dict:
        """Get statistics about the chat dataset."""
        sample_size = min(1000, len(dataset))
        turn_counts = []
        msg_lengths = []

        for row in dataset.select(range(sample_size)):
            conv = self.parse_conversation(row)
            turn_counts.append(len(conv))
            for turn in conv:
                msg_lengths.append(len(turn.get("content", "")))

        return {
            "total_records": len(dataset),
            "avg_turns": sum(turn_counts) / len(turn_counts) if turn_counts else 0,
            "min_turns": min(turn_counts) if turn_counts else 0,
            "max_turns": max(turn_counts) if turn_counts else 0,
            "avg_message_length": sum(msg_lengths) / len(msg_lengths) if msg_lengths else 0,
        }
