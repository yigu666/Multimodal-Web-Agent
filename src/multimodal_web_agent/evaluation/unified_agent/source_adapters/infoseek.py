from pathlib import Path
from typing import Any, Mapping

from .generic import GenericHeldOutAdapter
from .generic_heldout import GenericHeldoutSourceAdapter


class InfoSeekAdapter(GenericHeldOutAdapter):
    source_name = "infoseek"

    def __init__(
        self,
        path_or_config: Path | Mapping[str, Any] | None,
        *,
        project_root: Path | None = None,
    ):
        self._heldout_config = (
            dict(path_or_config)
            if isinstance(path_or_config, Mapping) else None
        )
        self._project_root = Path(project_root or ".")
        super().__init__(
            None if self._heldout_config is not None else path_or_config
        )

    def scan(self):
        if self._heldout_config is None:
            return super().scan()
        return GenericHeldoutSourceAdapter(
            self._heldout_config,
            project_root=self._project_root,
            source_name="infoseek_heldout",
        ).scan()
