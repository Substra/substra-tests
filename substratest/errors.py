class TError(Exception):
    """Substra Test Error."""

    pass


class FutureTimeoutError(TError):
    """Future execution timed out."""

    pass


class FutureFailureError(TError):
    """Future execution failed."""

    pass


class SynchronizationTimeoutError(TError):
    """Asset could not be synchronized inn time."""

    pass
