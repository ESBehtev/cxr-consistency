from pathlib import Path
import run_text_encoder_sweep as base

ROOT = Path(__file__).resolve().parents[1]
base.DEFAULT_CONFIG_ROOT = ROOT / "configs" / "text_sweeps_compact"
base.DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "text_sweeps_compact"

if __name__ == "__main__":
    base.main()
