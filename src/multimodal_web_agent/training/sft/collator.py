from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset import DatasetItem
from .masking import TokenizedExample


@dataclass
class ProtocolSFTCollator:
    pad_token_id: int
    label_pad_token_id: int = -100
    tokenize_fn: Callable[[Any], TokenizedExample] | None = None

    def _tokenized(self, item: Any) -> TokenizedExample:
        if isinstance(item, TokenizedExample):
            return item
        if isinstance(item, DatasetItem) and self.tokenize_fn is not None:
            return self.tokenize_fn(item)
        if isinstance(item, dict) and isinstance(item.get("tokenized"), TokenizedExample):
            return item["tokenized"]
        raise TypeError("ProtocolSFTCollator expects TokenizedExample values or a tokenize_fn")

    @staticmethod
    def _merge_visual(values: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(values) == 1:
            return values[0]
        # Qwen-VL pixel_values are patch-major, so concatenate patches.  A
        # conventional batch dimension is also safely concatenated at dim 0.
        return torch.cat(list(values), dim=0)

    def __call__(self, items: Sequence[Any]) -> Dict[str, Any]:
        tokenized = [self._tokenized(item) for item in items]
        if not tokenized:
            raise ValueError("cannot collate an empty batch")
        input_ids = pad_sequence(
            [item.input_ids for item in tokenized],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [item.attention_mask for item in tokenized],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [item.labels for item in tokenized],
            batch_first=True,
            padding_value=self.label_pad_token_id,
        )
        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "metadata": [item.metadata.to_dict() for item in tokenized],
        }
        weights = [
            item.token_weights
            for item in tokenized
            if item.token_weights is not None
        ]
        segments = [
            item.segment_ids
            for item in tokenized
            if item.segment_ids is not None
        ]
        if weights or segments:
            if len(weights) != len(tokenized) or len(segments) != len(tokenized):
                raise ValueError(
                    "Target weights missing for part of a weighted batch"
                )
            batch["token_weights"] = pad_sequence(
                weights,
                batch_first=True,
                padding_value=0.0,
            )
            batch["segment_ids"] = pad_sequence(
                segments,
                batch_first=True,
                padding_value=0,
            )
        visual = [item.pixel_values for item in tokenized if item.pixel_values is not None]
        if visual:
            if len(visual) != len(tokenized):
                raise ValueError("pixel_values missing for part of a multimodal batch")
            batch["pixel_values"] = self._merge_visual(visual)
        grids = [item.image_grid_thw for item in tokenized if item.image_grid_thw is not None]
        if grids:
            if len(grids) != len(tokenized):
                raise ValueError("image_grid_thw missing for part of a multimodal batch")
            batch["image_grid_thw"] = torch.cat(grids, dim=0)
        return batch
