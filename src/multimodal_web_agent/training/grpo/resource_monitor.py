from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResourceSnapshot:
    peak_vram: int = 0
    peak_cpu_ram: int = 0


def snapshot_resources() -> ResourceSnapshot:
    peak_vram = 0
    try:
        import torch
        if torch.cuda.is_available():
            peak_vram = int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        pass
    return ResourceSnapshot(peak_vram=peak_vram)
