from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from multimodal_web_agent.training.grpo.rewards.text_exploration_smoke_runner import (
    run_text_exploration_smoke,
)


def _classify_failure(message: str) -> str:
    lowered = str(message).casefold()
    # Alignment failures may also carry the strict EXACT_* prefix. Classify
    # the concrete failing contract before the generic exact-policy guard.
    if "alignment" in lowered or "action boundary" in lowered:
        return "ACTION_SPAN_ALIGNMENT_FAILED"
    if "exact_text_search_exploration_not_supported" in lowered or "logprob" in lowered:
        return "BEHAVIOR_LOGPROB_CONTRACT_FAILED"
    if "non-finite" in lowered or "nonfinite" in lowered:
        return "NONFINITE_TRAINING_SIGNAL"
    if "model_update_failed" in lowered or "checkpoint" in lowered:
        return "MODEL_UPDATE_FAILED"
    return "OTHER"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt-label", default="initial")
    parser.add_argument("--supersedes-failed-dir")
    args = parser.parse_args()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )
    config_path = (
        args.config if args.config.is_absolute()
        else ROOT / args.config
    )
    try:
        run_text_exploration_smoke(
            project_root=ROOT,
            output_dir=output_dir,
            config_path=config_path,
            attempt_label=args.attempt_label,
            supersedes_failed_dir=args.supersedes_failed_dir,
        )
    except Exception as exc:
        # Expected hard-gate failures already write the complete audit suite.
        # For earlier/unexpected failures, preserve a minimal, machine-readable
        # classification in the one permitted attempt directory as well.
        output_dir.mkdir(parents=True, exist_ok=True)
        if not (output_dir / "run_manifest.json").is_file():
            message = str(exc)
            classification = _classify_failure(message)
            (output_dir / "failure_report.json").write_text(
                json.dumps({
                    "status": "failed",
                    "failure_classification": classification,
                    "error_type": type(exc).__name__,
                    "error": message,
                    "training_performed": False,
                    "full_training_performed": False,
                    "unified_frozen_test_accessed": False,
                    "automatic_retry_performed": False,
                    "smoke_attempt_label": args.attempt_label,
                    "supersedes_failed_dir": args.supersedes_failed_dir,
                }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
