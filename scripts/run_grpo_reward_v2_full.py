#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_web_agent.training.grpo.rewards.full_runner import (  # noqa: E402
    run_reward_v2_full,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-config", type=Path,
        default=Path("configs/grpo/reward_v2_full_server.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = args.training_config
    if not config.is_absolute():
        config = ROOT / config
    output = args.output_dir
    if not output.is_absolute():
        output = ROOT / output
    run_reward_v2_full(
        project_root=ROOT,
        output_dir=output,
        training_config_path=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

