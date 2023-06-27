from substra.exception import FutureError


class SynchronizationTimeoutError(FutureError):
    """Asset could not be synchronized inn time."""
