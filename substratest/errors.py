class TError(Exception):
    """Substra Test Error."""


class FutureTimeoutError(TError):
    """Future execution timed out."""


class FutureFailureError(TError):
    """Future execution failed."""


class SynchronizationTimeoutError(TError):
    """Asset could not be synchronized inn time."""
