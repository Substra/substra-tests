from dataclasses import dataclass

from substra.sdk.schemas import InputRef

from substratest.task_outputs import OutputIdentifiers


@dataclass
class InputIdentifiers:
    MODEL = "model"
    LOCAL = "local"
    SHARED = "shared"
    PREDICTIONS = "predictions"
    DATA_SAMPLE = "opener"
    DATA_SAMPLE = "datasamples"
    OPENER = "opener"


def opener(opener_key):
    return [InputRef(identifier=InputIdentifiers.OPENER, asset_key=opener_key)]


def data_samples(data_samples_keys):
    return [
        InputRef(identifier=InputIdentifiers.DATA_SAMPLE, asset_key=data_sample) for data_sample in data_samples_keys
    ]


def data(opener_key, data_samples_keys):
    return opener(opener_key=opener_key) + data_samples(data_samples_keys=data_samples_keys)


def trains_to_train(model_keys):
    return [
        InputRef(
            identifier=InputIdentifiers.MODEL,
            parent_task_key=model_key,
            parent_task_output_identifier=OutputIdentifiers.MODEL,
        )
        for model_key in model_keys
    ]


def trains_to_aggregate(model_keys):
    return [
        InputRef(
            identifier=InputIdentifiers.MODEL,
            parent_task_key=model_key,
            parent_task_output_identifier=OutputIdentifiers.MODEL,
        )
        for model_key in model_keys
    ]


def train_to_predict(model_key):
    return [
        InputRef(
            identifier=InputIdentifiers.MODEL,
            parent_task_key=model_key,
            parent_task_output_identifier=OutputIdentifiers.MODEL,
        )
    ]


def predict_to_test(model_key):
    return [
        InputRef(
            identifier=InputIdentifiers.PREDICTIONS,
            parent_task_key=model_key,
            parent_task_output_identifier=OutputIdentifiers.PREDICTIONS,
        )
    ]


def composite_to_predict(model_key):
    return [
        InputRef(
            identifier=InputIdentifiers.MODEL,
            parent_task_key=model_key,
            parent_task_output_identifier=OutputIdentifiers.LOCAL,
        ),
        InputRef(
            identifier=InputIdentifiers.SHARED,
            parent_task_key=model_key,
            parent_task_output_identifier=OutputIdentifiers.SHARED,
        ),
    ]


def composite_to_local(model_key):
    return [
        InputRef(
            identifier=InputIdentifiers.LOCAL,
            parent_task_key=model_key,
            parent_task_output_identifier=OutputIdentifiers.LOCAL,
        )
    ]


def composite_to_composite(model1_key, model2_key=None):
    return [
        InputRef(
            identifier=InputIdentifiers.LOCAL,
            parent_task_key=model1_key,
            parent_task_output_identifier=OutputIdentifiers.LOCAL,
        ),
        InputRef(
            identifier=InputIdentifiers.SHARED,
            parent_task_key=model2_key or model1_key,
            parent_task_output_identifier=OutputIdentifiers.SHARED,
        ),
    ]


def aggregate_to_shared(model_key):
    return [
        InputRef(
            identifier=InputIdentifiers.SHARED,
            parent_task_key=model_key,
            parent_task_output_identifier=OutputIdentifiers.MODEL,
        )
    ]


def composites_to_aggregate(model_keys):
    return [
        InputRef(
            identifier=InputIdentifiers.MODEL,
            parent_task_key=model_key,
            parent_task_output_identifier=OutputIdentifiers.SHARED,
        )
        for model_key in model_keys
    ]


def aggregate_to_predict(model_key):
    return [
        InputRef(
            identifier=InputIdentifiers.MODEL,
            parent_task_key=model_key,
            parent_task_output_identifier=OutputIdentifiers.MODEL,
        )
    ]
