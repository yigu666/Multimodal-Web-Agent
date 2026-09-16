from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _equal(left: Any, right: Any) -> bool:
    if hasattr(left, "detach"):
        left = left.detach().cpu()
    if hasattr(right, "detach"):
        right = right.detach().cpu()
    try:
        import torch
        if torch.is_tensor(left) or torch.is_tensor(right):
            return bool(torch.equal(torch.as_tensor(left), torch.as_tensor(right)))
    except ImportError:
        pass
    return left == right


@dataclass(frozen=True)
class InputIdentity:
    input_ids: Any
    response_ids: Any
    policy_action_mask: Any
    information_mask: Any
    image_sha256: str


def assert_input_identity(stored: Mapping[str, Any] | InputIdentity,
                          update: Mapping[str, Any] | InputIdentity) -> None:
    def get(obj: Any, key: str) -> Any:
        return getattr(obj, key) if isinstance(obj, InputIdentity) else obj.get(key)
    for key in ("input_ids", "response_ids", "policy_action_mask", "information_mask"):
        if not _equal(get(stored, key), get(update, key)):
            raise AssertionError(f"input identity mismatch: {key}")
    if str(get(stored, "image_sha256")) != str(get(update, "image_sha256")):
        raise AssertionError("input identity mismatch: image_sha256")
    info_mask = get(stored, "information_mask")
    action_for_check = get(stored, "policy_action_mask")
    try:
        info_sum = int((info_mask * action_for_check).sum().item())
    except AttributeError:
        info_sum = sum(int(a and b) for a, b in zip(info_mask or [], action_for_check or []))
    if info_sum != 0:
        raise AssertionError("information_token_train_mask_sum must be zero")
    action_mask = get(stored, "policy_action_mask")
    try:
        action_count = int(action_mask.sum().item())
    except AttributeError:
        action_count = sum(action_mask or [])
    if action_count <= 0:
        raise AssertionError("policy_action_token_count must be positive")


class InputIdentityContract:
    """Checks the immutable rollout/update boundary without rebuilding inputs."""

    def __init__(self, stored: Mapping[str, Any] | InputIdentity):
        self.stored = stored

    def validate_update(self, update: Mapping[str, Any] | InputIdentity) -> None:
        assert_input_identity(self.stored, update)
