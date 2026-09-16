#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from multimodal_web_agent.training.sft.checkpoint_selection import CheckpointCandidate
from multimodal_web_agent.training.sft.config import load_config
from multimodal_web_agent.training.sft.dataset import (
    FORMAT_TRANSITIONS,
    select_format_smoke_subset,
)
from multimodal_web_agent.training.sft.format_checkpoint_selection import (
    select_format_checkpoint,
)
from multimodal_web_agent.training.sft.format_metrics import format_gate_report
from multimodal_web_agent.training.sft.sampler import (
    DeterministicFormatExposureSampler,
    InitialRouterSamplingConfig,
)


def _examples():
    return [
        SimpleNamespace(
            example_id="%s-%03d" % (transition, index),
            transition=transition,
            state_type=transition.split("_to_", 1)[0],
        )
        for transition in FORMAT_TRANSITIONS
        for index in range(3)
    ]


def _metrics(*, valid: float = 1.0, accuracy: float = 0.0):
    return {
        "protocol_valid_rate": valid,
        "exactly_one_action_rate": valid,
        "malformed_rate": 1.0 - valid,
        "extra_text_rate": 0.0,
        "forged_information_count": 0,
        "nonempty_reason_rate": 1.0,
        "nonempty_action_payload_rate": 1.0,
        "tag_closure_rate": 1.0,
        "target_truncation_count": 0,
        "target_action_distribution": {
            "answer": 10, "image_search": 10, "text_search": 1,
        },
        "action_type_accuracy": accuracy,
        "routing_collapsed": True,
    }


def main() -> int:
    smoke = load_config(
        REPOSITORY_ROOT / "configs/protocol_sft/train_smoke_format_v1.yaml",
        REPOSITORY_ROOT,
    )
    full = load_config(
        REPOSITORY_ROOT / "configs/protocol_sft/train_full_format_v1.yaml",
        REPOSITORY_ROOT,
    )
    assert smoke.is_format and smoke.is_smoke
    assert full.is_format and full.is_full
    assert smoke.training.max_steps == 10
    assert smoke.data.test_file is None and not smoke.data.allow_test_access
    assert full.data.test_file is None and not full.data.allow_test_access
    assert not smoke.loss.weighted_target_loss
    assert not full.loss.weighted_target_loss
    assert not smoke.loss_weights.enabled and not full.loss_weights.enabled

    values = _examples()
    subset = select_format_smoke_subset(values)
    assert len(subset) == 10
    assert {
        transition: sum(row.transition == transition for row in subset)
        for transition in FORMAT_TRANSITIONS
    } == {transition: 2 for transition in FORMAT_TRANSITIONS}

    sampler = DeterministicFormatExposureSampler(
        values,
        InitialRouterSamplingConfig(
            epoch_size=1000,
            transition_quotas={
                transition: 200 for transition in FORMAT_TRANSITIONS
            },
            seed=20260728,
        ),
    )
    epoch = sampler.sample_epoch(0)
    assert len(epoch.indices) == 1000
    assert epoch.audit["sampled_transition_counts"] == {
        transition: 200 for transition in FORMAT_TRANSITIONS
    }
    assert epoch.audit["sampler_is_policy_distribution"] is False
    assert epoch.indices == sampler.sample_epoch(0).indices

    bad_policy_good_format = _metrics(accuracy=0.0)
    assert format_gate_report(bad_policy_good_format)["passed"] is True
    candidates = [
        CheckpointCandidate(
            "checkpoint-epoch-1", Path("one"),
            _metrics(valid=1.0, accuracy=0.0), 0.2,
        ),
        CheckpointCandidate(
            "checkpoint-epoch-2", Path("two"),
            _metrics(valid=0.98, accuracy=1.0), 0.1,
        ),
    ]
    selected, report = select_format_checkpoint(candidates)
    assert selected.name == "checkpoint-epoch-1"
    assert report["policy_metrics_used"] is False
    assert report["routing_collapse_used"] is False
    print("PROTOCOL_FORMAT_SFT_V1_CONTRACTS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
