from pathlib import Path
import audit_text_encoder_sweep as base

ROOT = Path(__file__).resolve().parents[1]
base.DEFAULT_RESULTS_ROOT = ROOT / "experiments" / "text_sweeps_compact"
base.DEFAULT_OUT_DIR = ROOT / "results" / "text_sweeps_compact"

if __name__ == "__main__":
    base.main()
