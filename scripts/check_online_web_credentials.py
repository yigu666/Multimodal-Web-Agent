#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def status() -> dict[str, object]:
    return {
        "schema_version": "online-web-credentials-check-v2",
        "SERPER_AVAILABLE": bool(os.environ.get("SERPER_API_KEY")),
        "SERPAPI_LENS_AVAILABLE": bool(os.environ.get("SERPAPI_API_KEY")),
        "JINA_AVAILABLE": bool(os.environ.get("JINA_API_KEY")),
        "JINA_OPTIONAL": True,
        "GOOGLE_CLOUD_REQUIRED": False,
        "secrets_redacted": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = status()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(value, sort_keys=True))
    required = value["SERPER_AVAILABLE"] and value["SERPAPI_LENS_AVAILABLE"]
    print("ONLINE_CREDENTIAL_READY" if required else "ONLINE_CREDENTIAL_MISSING")
    return 0 if required else 2


if __name__ == "__main__":
    raise SystemExit(main())
