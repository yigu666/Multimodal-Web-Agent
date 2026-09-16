from __future__ import annotations

from typing import Any, Callable


_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_reward(name: str):
    def decorator(factory):
        if name in _REGISTRY and _REGISTRY[name] is not factory:
            raise ValueError(f"reward already registered: {name}")
        _REGISTRY[name] = factory
        return factory
    return decorator


def get_reward(name: str) -> Callable[..., Any]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown reward {name!r}; available={sorted(_REGISTRY)}") from exc


def available_rewards() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_reward(name: str, **kwargs: Any) -> Any:
    return get_reward(name)(**kwargs)
