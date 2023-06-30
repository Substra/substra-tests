from substra.sdk.exceptions import FutureError


class SynchronizationTimeoutError(FutureError):
    """Asset could not be synchronized inn time."""
