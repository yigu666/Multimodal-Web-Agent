class AcquisitionError(RuntimeError):
    exit_code = 1


class SourceInvalidError(AcquisitionError):
    exit_code = 22


class ArchiveRequiredError(AcquisitionError):
    exit_code = 23


class OvenAccessRequiredError(AcquisitionError):
    exit_code = 23


class InsufficientDiskError(AcquisitionError):
    exit_code = 24


class UpstreamUnavailableError(AcquisitionError):
    exit_code = 25
