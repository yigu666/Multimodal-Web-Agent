"""Compatibility-facing record module; canonical definitions live in schema."""
from .schema import RolloutRecord, VisualInputRecord, validate_rollout_group

__all__ = ["RolloutRecord", "VisualInputRecord", "validate_rollout_group"]
