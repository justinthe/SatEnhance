"""Exit codes shared by both systems (PRD section 12.1)."""

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    UNEXPECTED = 1
    INVALID_INPUT = 2
    AOI_TOO_LARGE = 3
    AUTH_FAILURE = 4
    NETWORK_FAILURE = 5
    NO_DATA = 10
    GEOCODE_NEEDS_CONFIRMATION = 11
    ENHANCE_INPUT_INVALID = 20
    MODEL_UNSUPPORTED = 21
    INFERENCE_FAILURE = 22


class SatEnhanceError(Exception):
    """Raised anywhere; CLIs translate it to a process exit code."""

    def __init__(self, code: ExitCode, message: str):
        super().__init__(message)
        self.code = ExitCode(code)
        self.message = message
        # Extra facts a caller can attach on the way up (e.g. {"run_dir": ...}) for error reports.
        self.context: dict = {}
