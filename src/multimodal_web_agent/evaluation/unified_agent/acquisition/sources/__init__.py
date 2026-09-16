from .infoseek import InfoSeekAcquisitionPlugin
from .local_archive import LocalArchiveAcquisitionPlugin
from .mmsearch import MMSearchAcquisitionPlugin
from .visual_infoseek import VisualInfoSeekAcquisitionPlugin


PLUGINS = {
    "infoseek": InfoSeekAcquisitionPlugin,
    "mmsearch": MMSearchAcquisitionPlugin,
    "local_archive": LocalArchiveAcquisitionPlugin,
    "official_huggingface": LocalArchiveAcquisitionPlugin,
    "official_git_release": LocalArchiveAcquisitionPlugin,
    "visual_infoseek": VisualInfoSeekAcquisitionPlugin,
}


def get_plugin(name: str):
    try:
        return PLUGINS[name]
    except KeyError as exc:
        raise ValueError("unsupported acquisition plugin: %s" % name) from exc


__all__ = ["get_plugin"]
