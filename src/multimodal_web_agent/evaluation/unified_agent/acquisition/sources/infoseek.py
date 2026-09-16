from .local_archive import LocalArchiveAcquisitionPlugin


class InfoSeekAcquisitionPlugin(LocalArchiveAcquisitionPlugin):
    id_fields = ("data_id", "sample_id", "id", "source_data_id")
    question_fields = ("question", "query")
