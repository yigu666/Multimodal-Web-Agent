#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import time
from typing import Any


def gpu_status() -> dict[str, Any]:
    memory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpus = []
    for row in csv.reader(io.StringIO(memory.stdout)):
        if not row:
            continue
        gpus.append({
            "index": int(row[0].strip()),
            "name": row[1].strip(),
            "memory_total_mib": int(row[2].strip()),
            "memory_used_mib": int(row[3].strip()),
            "memory_free_mib": int(row[4].strip()),
        })
    compute = []
    for row in csv.reader(io.StringIO(processes.stdout)):
        if len(row) < 3:
            continue
        pid = int(row[0].strip())
        if pid == os.getpid():
            continue
        compute.append({
            "pid": pid,
            "process_name": row[1].strip(),
            "used_memory_mib": int(row[2].strip()),
        })
    return {"gpus": gpus, "compute_processes": compute}


def wait_for_gpu_idle(*, minimum_free_mib: int, timeout_seconds: int, poll_seconds: int = 30) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        status = gpu_status()
        free_ok = bool(status["gpus"]) and all(
            int(gpu["memory_free_mib"]) >= minimum_free_mib
            for gpu in status["gpus"]
        )
        idle = not status["compute_processes"] and free_ok
        status.update({
            "minimum_free_mib": minimum_free_mib,
            "idle": idle,
            "waited_seconds": round(time.monotonic() - started, 3),
        })
        if idle:
            return status
        if timeout_seconds <= 0 or time.monotonic() - started >= timeout_seconds:
            raise RuntimeError("GPU_BUSY: %s" % json.dumps(status, sort_keys=True))
        time.sleep(min(poll_seconds, max(1, timeout_seconds - int(time.monotonic() - started))))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-free-mib", type=int, default=18000)
    parser.add_argument("--wait-timeout-seconds", type=int, default=0)
    args = parser.parse_args()
    try:
        value = wait_for_gpu_idle(
            minimum_free_mib=args.minimum_free_mib,
            timeout_seconds=args.wait_timeout_seconds,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 2
    print(json.dumps(value, sort_keys=True))
    print("GPU_IDLE_CONTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
