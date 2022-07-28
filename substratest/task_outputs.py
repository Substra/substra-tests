from dataclasses import dataclass

from substra.sdk.schemas import ComputeTaskOutput
from substra.sdk.schemas import Permissions


@dataclass
class OutputIdentifiers:
    MODEL = "model"
    LOCAL = "local"
    SHARED = "shared"
    PREDICTIONS = "predictions"
    PERFORMANCE = "performance"


DEFAULT_TRAINTUPLE_OUTPUTS = {
    OutputIdentifiers.MODEL: ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[]))
}
DEFAULT_AGGREGATETUPLE_OUTPUTS = {
    OutputIdentifiers.MODEL: ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[]))
}
DEFAULT_PREDICTTUPLE_OUTPUTS = {
    OutputIdentifiers.PREDICTIONS: ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[]))
}
DEFAULT_TESTTUPLE_OUTPUTS = {
    OutputIdentifiers.PERFORMANCE: ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[]))
}
DEFAULT_COMPOSITE_TRAINTUPLE_OUTPUTS = {
    OutputIdentifiers.SHARED: ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[])),
    OutputIdentifiers.LOCAL: ComputeTaskOutput(permissions=Permissions(public=True, authorized_ids=[])),
}
