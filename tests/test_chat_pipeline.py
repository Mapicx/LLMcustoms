"""
End-to-end test: Chat-based training pipeline.
Uses the 'philschmid/sharegpt-raw' dataset from HuggingFace (ShareGPT format).
Trains for 3 steps only — just enough to verify the full pipeline runs.
"""

import sys
import os
import shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

TEST_OUTPUT_DIR = os.path.join(project_root, "test_outputs", "chat")

# OpenAI messages format, well-maintained, always works
CHAT_DATASET = "HuggingFaceH4/ultrachat_200k"


def patch_training_config():
    from llmcustoms.utils import config as cfg_module
    cfg_module.settings.output_dir = TEST_OUTPUT_DIR


def test_chat_pipeline():
    print("=" * 60)
    print("E2E TEST: Chat Pipeline (ShareGPT)")
    print("=" * 60)

    patch_training_config()

    from llmcustoms.First_Wrapper.first_wrpr import FineTuner

    # --- Step 1: Init ---
    print("\n[1/4] Initializing FineTuner...")
    tuner = FineTuner(
        data_path=None,
        dataset_name=CHAT_DATASET,
        model="auto",
        preset="auto",
        training_mode="chat",
        max_steps=3,
    )
    print(f"      Model   : {tuner.model.display_name}")
    print(f"      Preset  : {tuner.preset}")
    print("      ✅ FineTuner initialized")

    # --- Step 2: Load & validate dataset ---
    print("\n[2/4] Loading and validating dataset...")
    raw = tuner.data_handler.load_dataset()
    raw = raw.select(range(50))
    print(f"      Records : {len(raw)}")

    # Auto-detect format from first record
    detected = tuner.data_handler.detect_format(raw[0])
    print(f"      Detected format: {detected}")

    tuner.data_handler.validate_dataset(raw)
    print("      ✅ Dataset loaded and validated")

    # --- Step 3: Statistics ---
    print("\n[3/4] Dataset statistics...")
    stats = tuner.data_handler.get_statistics(raw)
    print(f"      Total records      : {stats['total_records']}")
    print(f"      Avg turns          : {stats['avg_turns']:.1f}")
    print(f"      Avg message length : {stats['avg_message_length']:.0f} chars")
    print("      ✅ Statistics OK")

    # --- Step 4: Full train() ---
    print("\n[4/4] Running train() (3 steps)...")
    original_load = tuner.data_handler.load_dataset
    tuner.data_handler.load_dataset = lambda: raw

    save_path = tuner.train()

    tuner.data_handler.load_dataset = original_load
    print(f"\n      Adapter saved at: {save_path}")
    assert os.path.isdir(save_path), "Save path does not exist!"
    assert any(f.endswith(".safetensors") for f in os.listdir(save_path)), \
        "No adapter weights found in save path!"
    print("      ✅ Adapter saved and verified")

    print("\n" + "=" * 60)
    print("✅ CHAT PIPELINE TEST PASSED")
    print("=" * 60)

    return save_path


if __name__ == "__main__":
    try:
        save_path = test_chat_pipeline()
        print(f"\nOutput: {save_path}")
    except Exception as e:
        import traceback
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        test_out = os.path.join(project_root, "test_outputs")
        if os.path.exists(test_out):
            shutil.rmtree(test_out)
            print("Cleaned up test_outputs/")
