from typing import Any

from .local_archive import LocalArchiveAcquisitionPlugin


class MMSearchAcquisitionPlugin(LocalArchiveAcquisitionPlugin):
    id_fields = ("sample_id", "data_id", "id", "source_data_id")
    question_fields = ("query", "question")

    def prepare_row(self, row: dict[str, Any]) -> dict[str, Any]:
        value = dict(row)
        websites: list[Any] = []
        for key, item in row.items():
            lowered = key.casefold()
            if (
                lowered.startswith("website")
                or lowered in {
                    "search_results",
                    "retrieved_websites",
                    "rerank_results",
                }
            ) and item not in (None, "", [], {}):
                websites.append(item)
        if websites and not value.get("text_search_results"):
            value["text_search_results"] = websites
        return value
