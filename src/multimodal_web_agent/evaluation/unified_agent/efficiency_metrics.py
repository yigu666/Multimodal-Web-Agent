from __future__ import annotations

from statistics import mean, median
from typing import Any, Mapping, Sequence


def episode_efficiency(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not episodes:
        raise ValueError("efficiency metrics require episodes")
    total_seconds = sum(float(row["wall_clock_seconds"]) for row in episodes)
    return {
        "avg_output_tokens": mean(
            sum(int(turn["output_token_count"]) for turn in row["turns"])
            for row in episodes
        ),
        "avg_agent_turns": mean(
            int(row["agent_turn_count"]) for row in episodes
        ),
        "avg_tool_calls": mean(
            int(row["tool_call_count"]) for row in episodes
        ),
        "throughput_episodes_per_minute": (
            len(episodes) / total_seconds * 60.0 if total_seconds else 0.0
        ),
    }


def median_efficiency_runs(
    runs: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    if len(runs) != 3:
        raise ValueError("Unified efficiency requires exactly three runs")
    return {
        key: median(float(run[key]) for run in runs)
        for key in (
            "avg_output_tokens",
            "avg_agent_turns",
            "avg_tool_calls",
            "throughput_episodes_per_minute",
        )
    }
