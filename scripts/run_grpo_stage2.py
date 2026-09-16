#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodal_web_agent.training.grpo.stage2_runner import run_stage2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    run_stage2(project_root=ROOT, config_path=config, output_dir=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

