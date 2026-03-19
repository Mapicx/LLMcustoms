"""
End-to-end test: Instruction-based training pipeline.
Uses databricks/databricks-dolly-15k from HuggingFace.
Trains for 3 steps only — just enough to verify the full pipeline runs.
"""

import sys
import os
import shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

TEST_OUTPUT_DIR = os.path.join(project_root, "test_outputs", "instruction")

def patch_training_config():
    """Override output dir for testing."""
    from llmcustoms.utils import config as cfg_module
    cfg_module.settings.output_dir = TEST_OUTPUT_DIR

def test_instruction_pipeline():
    print("=" * 60)
    print("E2E TEST: Instruction Pipeline (Dolly-15k)")
    print("=" * 60)

    patch_training_config()

    from llmcustoms.First_Wrapper.first_wrpr import FineTuner

    # --- Step 1: Init ---
    print("\n[1/4] Initializing FineTuner...")
    tuner = FineTuner(
        data_path=None,
        dataset_name="databricks/databricks-dolly-15k",
        model="auto",
        preset="auto",
        training_mode="instruction",
        prompt_template="alpaca",
        mask_instruction=True,
        max_steps=3
    )
    print(f"      Model   : {tuner.model.display_name}")
    print(f"      Preset  : {tuner.preset}")
    print("      ✅ FineTuner initialized")

    # --- Step 2: Load & validate dataset manually ---
    print("\n[2/4] Loading and validating dataset...")
    raw = tuner.data_handler.load_dataset()
    # Use only first 50 records to keep it fast
    raw = raw.select(range(50))
    print(f"      Records : {len(raw)}")
    tuner.data_handler.validate_dataset(raw)
    print("      ✅ Dataset loaded and validated")

    # --- Step 3: Check statistics ---
    print("\n[3/4] Dataset statistics...")
    stats = tuner.data_handler.get_statistics(raw)
    print(f"      Total records          : {stats['total_records']}")
    print(f"      Avg instruction length : {stats['avg_instruction_length']:.0f} chars")
    print(f"      Avg response length    : {stats['avg_response_length']:.0f} chars")
    print("      ✅ Statistics OK")

    # --- Step 4: Full train() ---
    print("\n[4/4] Running train() (3 steps)...")
    # Monkey-patch the dataset load inside train() to use our 50-record slice
    original_load = tuner.data_handler.load_dataset
    tuner.data_handler.load_dataset = lambda: raw

    save_path = tuner.train()

    tuner.data_handler.load_dataset = original_load  # restore
    print(f"\n      Adapter saved at: {save_path}")
    assert os.path.isdir(save_path), "Save path does not exist!"
    assert any(f.endswith(".safetensors") for f in os.listdir(save_path)), \
        "No adapter weights found in save path!"
    print("      ✅ Adapter saved and verified")

    print("\n" + "=" * 60)
    print("✅ INSTRUCTION PIPELINE TEST PASSED")
    print("=" * 60)

    return save_path


if __name__ == "__main__":
    try:
        save_path = test_instruction_pipeline()
        print(f"\nOutput: {save_path}")
    except Exception as e:
        import traceback
        print(f"\n❌ TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up test outputs
        test_out = os.path.join(project_root, "test_outputs")
        if os.path.exists(test_out):
            shutil.rmtree(test_out)
            print("Cleaned up test_outputs/")
