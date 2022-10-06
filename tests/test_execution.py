import pytest
import substra
from substra.sdk.models import Status
from substra.sdk.schemas import TaskSpec

import substratest as sbt
from substratest.factory import AlgoCategory
from substratest.factory import AugmentedDataset
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import FLTaskOutputGenerator
from substratest.fl_interface import InputIdentifiers
from substratest.fl_interface import OutputIdentifiers


@pytest.mark.slow
def test_tuples_execution_on_same_organization(factory, network, client, default_dataset, default_metric, worker):
    """Execution of a traintuple, a following testtuple and a following traintuple."""

    spec = factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    predict_algo_spec = factory.create_algo(AlgoCategory.predict)
    predict_algo = client.add_algo(predict_algo_spec)

    # create traintuple
    def get_traintuple_spec() -> TaskSpec:
        return factory.create_traintuple(
            algo=algo,
            inputs=default_dataset.train_data_inputs,
            metadata={"foo": "bar"},
            worker=worker,
        )

    spec = get_traintuple_spec()
    traintuple = client.add_task(spec)
    traintuple = client.wait(traintuple)
    assert traintuple.status == Status.done
    assert traintuple.error_type is None
    assert traintuple.metadata == {"foo": "bar"}
    assert len(traintuple.outputs) == 1
    assert traintuple.outputs[OutputIdentifiers.model].value is not None

    if network.options.enable_model_download:
        model = traintuple.outputs[OutputIdentifiers.model].value
        assert client.download_model(model.key) == b'{"value": 2.2}'

    # check we can add twice the same traintuple
    spec = get_traintuple_spec()
    client.add_task(spec)

    # create testtuple
    spec = factory.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintuple.key),
        worker=worker,
    )

    predicttuple = client.add_task(spec)
    predicttuple = client.wait(predicttuple)
    assert predicttuple.status == Status.done
    assert predicttuple.error_type is None

    spec = factory.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttuple.key),
        worker=worker,
    )
    testtuple = client.add_task(spec)
    testtuple = client.wait(testtuple)
    assert testtuple.status == Status.done
    assert testtuple.error_type is None
    assert testtuple.outputs[OutputIdentifiers.performance].value == pytest.approx(2)

    # add a traintuple depending on first traintuple
    first_traintuple_key = traintuple.key
    spec = factory.create_traintuple(
        algo=algo,
        inputs=default_dataset.train_data_inputs + FLTaskInputGenerator.trains_to_train([first_traintuple_key]),
        metadata=None,
        worker=worker,
    )
    traintuple = client.add_task(spec)
    traintuple = client.wait(traintuple)
    assert traintuple.status == Status.done
    assert testtuple.error_type is None
    assert traintuple.metadata == {}

    expected_inputs = default_dataset.train_data_inputs + FLTaskInputGenerator.trains_to_train([first_traintuple_key])
    assert traintuple.inputs == expected_inputs


@pytest.mark.slow
def test_federated_learning_workflow(factory, client, default_datasets, workers):
    """Test federated learning workflow on each organization."""

    # create test environment
    spec = factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    # create 1 traintuple per dataset and chain them
    traintuple = None
    rank = 0
    compute_plan_key = None

    # default_datasets contains datasets on each organization and
    # that has a result we can use for federated learning
    for index, dataset in enumerate(default_datasets):

        traintuples = [traintuple.key] if traintuple else []

        spec = factory.create_traintuple(
            algo=algo,
            inputs=dataset.train_data_inputs + FLTaskInputGenerator.trains_to_train(traintuples),
            tag="foo",
            rank=rank,
            compute_plan_key=compute_plan_key,
            worker=workers[index],
        )
        traintuple = client.add_task(spec)
        traintuple = client.wait(traintuple)
        assert traintuple.status == Status.done
        assert traintuple.error_type is None
        assert len(traintuple.outputs) == 1
        assert traintuple.outputs[OutputIdentifiers.model].value is not None
        assert traintuple.tag == "foo"
        assert traintuple.compute_plan_key  # check it is not None or ''

        rank += 1
        compute_plan_key = traintuple.compute_plan_key

    # check a compute plan has been created and its status is at done
    cp = client.get_compute_plan(compute_plan_key)
    assert cp.status == "PLAN_STATUS_DONE"


@pytest.mark.slow
@pytest.mark.remote_only
def test_tuples_execution_on_different_organizations(
    factory,
    client_1,
    client_2,
    default_metric_1,
    default_dataset_1,
    default_dataset_2,
    channel,
    workers,
):
    """Execution of a traintuple on organization 1 and the following testtuple on organization 2."""
    # add test data samples / dataset / metric on organization 1
    spec = factory.create_algo(AlgoCategory.simple)
    algo_2 = client_2.add_algo(spec)

    predict_algo_spec = factory.create_algo(AlgoCategory.predict)
    predict_algo_2 = client_2.add_algo(predict_algo_spec)

    channel.wait_for_asset_synchronized(algo_2)
    channel.wait_for_asset_synchronized(predict_algo_2)

    # add traintuple on organization 2; should execute on organization 2 (dataset located on organization 2)
    spec = factory.create_traintuple(
        algo=algo_2,
        inputs=default_dataset_2.train_data_inputs,
        worker=workers[1],
    )
    traintuple = client_1.add_task(spec)
    traintuple = client_1.wait(traintuple)
    assert traintuple.status == Status.done
    assert traintuple.error_type is None
    assert len(traintuple.outputs) == 1
    assert traintuple.outputs[OutputIdentifiers.model].value is not None
    assert traintuple.worker == client_2.organization_id

    # add testtuple; should execute on organization 1 (default_dataset_1 is located on organization 1)
    spec = factory.create_predicttuple(
        algo=predict_algo_2,
        inputs=default_dataset_1.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintuple.key),
        worker=workers[0],
    )
    predicttuple = client_1.add_task(spec)
    predicttuple = client_1.wait(predicttuple)
    assert predicttuple.status == Status.done
    assert predicttuple.error_type is None
    assert predicttuple.worker == client_1.organization_id

    spec = factory.create_testtuple(
        algo=default_metric_1,
        inputs=default_dataset_1.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttuple.key),
        worker=workers[0],
    )
    testtuple = client_1.add_task(spec)
    testtuple = client_1.wait(testtuple)
    assert testtuple.status == Status.done
    assert testtuple.error_type is None
    assert testtuple.worker == client_1.organization_id
    assert testtuple.outputs[OutputIdentifiers.performance].value == pytest.approx(2)


@pytest.mark.slow
@pytest.mark.subprocess_skip
def test_algo_build_failure(factory, network, default_dataset_1, worker):
    """Invalid Dockerfile is causing compute task failure."""

    dockerfile = factory.default_algo_dockerfile(method_name=sbt.factory.DEFAULT_ALGO_METHOD_NAME[AlgoCategory.simple])
    dockerfile += "\nRUN invalid_command"
    spec = factory.create_algo(category=AlgoCategory.simple, dockerfile=dockerfile)
    algo = network.clients[0].add_algo(spec)

    spec = factory.create_traintuple(algo=algo, inputs=default_dataset_1.train_data_inputs, worker=worker)

    if network.clients[0].backend_mode != substra.BackendType.REMOTE:
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.BuildError):
            network.clients[0].add_task(spec)
    else:
        traintuple = network.clients[0].add_task(spec)
        traintuple = network.clients[0].wait(traintuple, raises=False)

        assert traintuple.status == Status.failed
        assert traintuple.error_type == substra.sdk.models.TaskErrorType.build
        assert traintuple.outputs[OutputIdentifiers.model].value is None

        for client in (network.clients[0], network.clients[1]):
            logs = client.download_logs(traintuple.key)
            assert "invalid_command: not found" in logs
            assert client.get_logs(traintuple.key) == logs


@pytest.mark.slow
def test_task_execution_failure(factory, network, default_dataset_1, worker):
    """Invalid algo script is causing compute task failure."""

    spec = factory.create_algo(category=AlgoCategory.simple, py_script=sbt.factory.INVALID_ALGO_SCRIPT)
    algo = network.clients[0].add_algo(spec)

    spec = factory.create_traintuple(algo=algo, inputs=default_dataset_1.train_data_inputs, worker=worker)

    if network.clients[0].backend_mode != substra.BackendType.REMOTE:
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.ExecutionError):
            network.clients[0].add_task(spec)
    else:
        traintuple = network.clients[0].add_task(spec)
        traintuple = network.clients[0].wait(traintuple, raises=False)

        assert traintuple.status == Status.failed
        assert traintuple.error_type == substra.sdk.models.TaskErrorType.execution
        assert traintuple.outputs[OutputIdentifiers.model].value is None

        for client in (network.clients[0], network.clients[1]):
            logs = client.download_logs(traintuple.key)
            assert "Traceback (most recent call last):" in logs
            assert client.get_logs(traintuple.key) == logs


@pytest.mark.slow
def test_composite_traintuple_execution_failure(factory, client, default_dataset, worker):
    """Invalid composite algo script is causing traintuple failure."""

    spec = factory.create_algo(AlgoCategory.composite, py_script=sbt.factory.INVALID_COMPOSITE_ALGO_SCRIPT)
    algo = client.add_algo(spec)

    spec = factory.create_composite_traintuple(algo=algo, inputs=default_dataset.train_data_inputs, worker=worker)
    if client.backend_mode == substra.BackendType.REMOTE:
        composite_traintuple = client.add_task(spec)
        composite_traintuple = client.wait(composite_traintuple, raises=False)

        assert composite_traintuple.status == Status.failed
        assert composite_traintuple.error_type == substra.sdk.models.TaskErrorType.execution
        assert composite_traintuple.outputs[OutputIdentifiers.local].value is None
        assert composite_traintuple.outputs[OutputIdentifiers.shared].value is None
        assert "Traceback (most recent call last):" in client.download_logs(composite_traintuple.key)

    elif client.backend_mode in (substra.BackendType.LOCAL_SUBPROCESS, substra.BackendType.LOCAL_DOCKER):
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.ExecutionError):
            composite_traintuple = client.add_task(spec)

    else:
        raise NotImplementedError(f"Backend mode '{client.backend_mode}' is not supported.")


@pytest.mark.slow
def test_aggregatetuple_execution_failure(factory, client, default_dataset, worker):
    """Invalid algo script is causing traintuple failure."""

    spec = factory.create_algo(AlgoCategory.composite)
    composite_algo = client.add_algo(spec)

    spec = factory.create_algo(AlgoCategory.aggregate, py_script=sbt.factory.INVALID_AGGREGATE_ALGO_SCRIPT)
    aggregate_algo = client.add_algo(spec)

    composite_traintuple_keys = []
    for i in [0, 1]:
        spec = factory.create_composite_traintuple(
            algo=composite_algo,
            inputs=default_dataset.opener_input + [default_dataset.train_data_sample_inputs[i]],
            worker=worker,
        )
        composite_traintuple_keys.append(client.add_task(spec).key)

    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        inputs=FLTaskInputGenerator.composites_to_aggregate(composite_traintuple_keys),
        worker=client.organization_id,
    )

    if client.backend_mode == substra.BackendType.REMOTE:
        aggregatetuple = client.add_task(spec)
        aggregatetuple = client.wait(aggregatetuple, raises=False)

        for composite_traintuple_key in composite_traintuple_keys:
            composite_traintuple = client.get_task(composite_traintuple_key)
            assert composite_traintuple.status == Status.done
            assert composite_traintuple.error_type is None

        assert aggregatetuple.status == Status.failed
        assert aggregatetuple.error_type == substra.sdk.models.TaskErrorType.execution
        assert aggregatetuple.outputs[OutputIdentifiers.model].value is None
        assert "Traceback (most recent call last):" in client.download_logs(aggregatetuple.key)

    elif client.backend_mode in (substra.BackendType.LOCAL_SUBPROCESS, substra.BackendType.LOCAL_DOCKER):
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.ExecutionError):
            aggregatetuple = client.add_task(spec)

    else:
        raise NotImplementedError(f"Backend mode '{client.backend_mode}' is not supported.")


@pytest.mark.slow
def test_composite_traintuples_execution(factory, client, default_dataset, default_metric, worker):
    """Execution of composite traintuples."""

    spec = factory.create_algo(AlgoCategory.composite)
    algo = client.add_algo(spec)

    spec = factory.create_algo(AlgoCategory.predict_composite)
    predict_algo = client.add_algo(spec)

    # first composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        inputs=default_dataset.train_data_inputs,
        worker=worker,
    )
    composite_traintuple_1 = client.add_task(spec)
    composite_traintuple_1 = client.wait(composite_traintuple_1)
    assert composite_traintuple_1.status == Status.done
    assert composite_traintuple_1.error_type is None
    assert len(composite_traintuple_1.outputs) == 2

    # second composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        inputs=default_dataset.train_data_inputs
        + FLTaskInputGenerator.composite_to_composite(composite_traintuple_1.key),
        worker=worker,
    )
    composite_traintuple_2 = client.add_task(spec)
    composite_traintuple_2 = client.wait(composite_traintuple_2)
    assert composite_traintuple_2.status == Status.done
    assert composite_traintuple_2.error_type is None
    assert len(composite_traintuple_2.outputs) == 2

    # add a 'composite' testtuple
    spec = factory.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.composite_to_predict(composite_traintuple_2.key),
        worker=worker,
    )
    predicttuple = client.add_task(spec)
    predicttuple = client.wait(predicttuple)
    assert predicttuple.status == Status.done
    assert predicttuple.error_type is None

    spec = factory.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttuple.key),
        worker=worker,
    )
    testtuple = client.add_task(spec)
    testtuple = client.wait(testtuple)
    assert testtuple.status == Status.done
    assert testtuple.error_type is None
    assert testtuple.outputs[OutputIdentifiers.performance].value == pytest.approx(32)

    # list composite traintuple
    composite_traintuples = client.list_task()
    composite_traintuple_keys = set([t.key for t in composite_traintuples])
    assert set([composite_traintuple_1.key, composite_traintuple_2.key]).issubset(composite_traintuple_keys)


@pytest.mark.slow
def test_aggregatetuple(factory, client, default_metric, default_dataset, worker):
    """Execution of aggregatetuple aggregating traintuples. (traintuples -> aggregatetuple)"""

    number_of_traintuples_to_aggregate = 3

    train_data_sample_inputs = default_dataset.train_data_sample_inputs[:number_of_traintuples_to_aggregate]

    spec = factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    spec = factory.create_algo(AlgoCategory.predict)
    predict_algo = client.add_algo(spec)

    # add traintuples
    traintuple_keys = []
    for data_sample_input in train_data_sample_inputs:
        spec = factory.create_traintuple(
            algo=algo, inputs=default_dataset.opener_input + [data_sample_input], worker=worker
        )
        traintuple = client.add_task(spec)
        traintuple_keys.append(traintuple.key)

    spec = factory.create_algo(AlgoCategory.aggregate)
    aggregate_algo = client.add_algo(spec)

    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        worker=client.organization_id,
        inputs=FLTaskInputGenerator.trains_to_aggregate(traintuple_keys),
    )
    aggregatetuple = client.add_task(spec)
    assert (
        len([i for i in aggregatetuple.inputs if i.identifier == InputIdentifiers.models])
        == number_of_traintuples_to_aggregate
    )

    spec = factory.create_predicttuple(
        algo=predict_algo,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.aggregate_to_predict(aggregatetuple.key),
        worker=worker,
    )
    predicttuple = client.add_task(spec)
    predicttuple = client.wait(predicttuple)

    spec = factory.create_testtuple(
        algo=default_metric,
        inputs=default_dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttuple.key),
        worker=worker,
    )
    testtuple = client.add_task(spec)
    testtuple = client.wait(testtuple)


@pytest.mark.slow
def test_aggregatetuple_chained(factory, client, default_dataset, worker):
    """Execution of 2 chained aggregatetuple (traintuple -> aggregatetuple -> aggregatetuple)."""

    number_of_traintuples_to_aggregate = 1

    train_data_sample_input = default_dataset.train_data_sample_inputs[:1]

    spec = factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    # add traintuples
    spec = factory.create_traintuple(
        algo=algo,
        inputs=default_dataset.opener_input + train_data_sample_input,
        worker=worker,
    )
    traintuple = client.add_task(spec)

    spec = factory.create_algo(AlgoCategory.aggregate)
    aggregate_algo = client.add_algo(spec)

    # add first layer of aggregatetuples
    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        worker=client.organization_id,
        inputs=FLTaskInputGenerator.trains_to_aggregate([traintuple.key]),
    )

    aggregatetuple_1 = client.add_task(spec)
    assert (
        len([i for i in aggregatetuple_1.inputs if i.identifier == InputIdentifiers.models])
        == number_of_traintuples_to_aggregate
    )

    # add second layer of aggregatetuple
    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        worker=client.organization_id,
        inputs=FLTaskInputGenerator.trains_to_aggregate([aggregatetuple_1.key]),
    )

    aggregatetuple_2 = client.add_task(spec)
    aggregatetuple_2 = client.wait(aggregatetuple_2)
    assert aggregatetuple_2.status == Status.done
    assert aggregatetuple_2.error_type is None
    assert len([i for i in aggregatetuple_2.inputs if i.identifier == InputIdentifiers.models]) == 1


@pytest.mark.slow
def test_aggregatetuple_traintuple(factory, client, default_dataset, worker):
    """Execution of traintuple after an aggregatetuple (traintuples -> aggregatetuple -> traintuples)"""

    number_of_traintuples = 2

    train_data_sample_inputs = default_dataset.train_data_sample_inputs[:number_of_traintuples]
    train_data_sample_input_1 = train_data_sample_inputs[0]
    train_data_sample_input_2 = train_data_sample_inputs[1]

    spec = factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    # add first part of the traintuples
    spec = factory.create_traintuple(
        algo=algo,
        inputs=default_dataset.opener_input + [train_data_sample_input_1],
        worker=worker,
    )
    traintuple_1 = client.add_task(spec)

    spec = factory.create_algo(AlgoCategory.aggregate)
    aggregate_algo = client.add_algo(spec)

    # add aggregatetuple
    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        worker=client.organization_id,
        inputs=FLTaskInputGenerator.trains_to_aggregate([traintuple_1.key]),
    )
    aggregatetuple = client.add_task(spec)
    assert len([i for i in aggregatetuple.inputs if i.identifier == InputIdentifiers.models]) == 1

    # add second part of the traintuples
    spec = factory.create_traintuple(
        algo=algo,
        inputs=default_dataset.opener_input
        + [train_data_sample_input_2]
        + FLTaskInputGenerator.trains_to_train([aggregatetuple.key]),
        worker=worker,
    )

    traintuple_2 = client.add_task(spec)
    traintuple_2 = client.wait(traintuple_2)

    assert traintuple_2.status == Status.done
    assert traintuple_2.error_type is None


@pytest.mark.slow
@pytest.mark.remote_only
def test_composite_traintuple_2_organizations_to_composite_traintuple(factory, clients, default_datasets, workers):
    """A composite traintuple which take as input a composite traintuple (input_head_model) from
    organization 1 and another composite traintuple (inpute_trunk_model) from organization 2
    """

    spec = factory.create_algo(AlgoCategory.composite)
    composite_algo = clients[0].add_algo(spec)

    # composite traintuples on organization 1 and organization 2
    composite_traintuple_keys = []
    for index, dataset in enumerate(default_datasets):
        spec = factory.create_composite_traintuple(
            algo=composite_algo,
            inputs=dataset.opener_input + dataset.train_data_sample_inputs[:1],
            outputs=FLTaskOutputGenerator.composite_traintuple(
                shared_authorized_ids=[c.organization_id for c in clients],
                local_authorized_ids=[dataset.owner],
            ),
            worker=workers[index],
        )
        composite_traintuple_key = clients[0].add_task(spec).key
        composite_traintuple_keys.append(composite_traintuple_key)

    spec = factory.create_composite_traintuple(
        algo=composite_algo,
        inputs=default_datasets[0].train_data_inputs
        + FLTaskInputGenerator.composite_to_composite(composite_traintuple_keys[0], composite_traintuple_keys[1]),
        rank=1,
        outputs=FLTaskOutputGenerator.composite_traintuple(
            shared_authorized_ids=[c.organization_id for c in clients],
            local_authorized_ids=[dataset.owner],
        ),
        worker=workers[0],
    )
    composite_traintuple = clients[0].add_task(spec)
    composite_traintuple = clients[0].wait(composite_traintuple)

    assert composite_traintuple.status == Status.done


@pytest.mark.slow
def test_aggregate_composite_traintuples(factory, network, clients, default_datasets, default_metrics, workers):
    """Do 2 rounds of composite traintuples aggregations on multiple organizations.

    Compute plan details:

    Round 1:
    - Create 2 composite traintuples executed on two datasets located on organization 1 and
      organization 2.
    - Create an aggregatetuple on organization 1, aggregating the two previous composite
      traintuples (trunk models aggregation).

    Round 2:
    - Create 2 composite traintuples executed on each organizations that depend on: the
      aggregated tuple and the previous composite traintuple executed on this organization. That
      is to say, the previous round aggregated trunk models from all organizations and the
      previous round head model from this organization.
    - Create an aggregatetuple on organization 1, aggregating the two previous composite
      traintuples (similar to round 1 aggregatetuple).
    - Create a testtuple for each previous composite traintuples and aggregate tuple
      created during this round.

    (optional) if the option "enable_intermediate_model_removal" is True:
    - Since option "enable_intermediate_model_removal" is True, the aggregate model created on round 1 should
      have been deleted from the backend after round 2 has completed.
    - Create a traintuple that depends on the aggregate tuple created on round 1. Ensure that it fails to start.

    This test refers to the model composition use case.
    """

    aggregate_worker = clients[0].organization_id
    number_of_rounds = 2

    # register algos on first organization
    spec = factory.create_algo(AlgoCategory.composite)
    composite_algo = clients[0].add_algo(spec)
    spec = factory.create_algo(AlgoCategory.aggregate)
    aggregate_algo = clients[0].add_algo(spec)
    spec = factory.create_algo(AlgoCategory.predict)
    predict_algo = clients[0].add_algo(spec)
    spec = factory.create_algo(AlgoCategory.predict_composite)
    predict_algo_composite = clients[0].add_algo(spec)

    # launch execution
    previous_aggregatetuple_key = None
    previous_composite_traintuple_keys = []

    for round_ in range(number_of_rounds):
        # create composite traintuple on each organization
        composite_traintuple_keys = []
        for index, dataset in enumerate(default_datasets):

            if previous_aggregatetuple_key:
                input_models = FLTaskInputGenerator.composite_to_local(
                    previous_composite_traintuple_keys[index]
                ) + FLTaskInputGenerator.aggregate_to_shared(previous_aggregatetuple_key)

            else:
                input_models = []

            spec = factory.create_composite_traintuple(
                algo=composite_algo,
                inputs=[dataset.train_data_sample_inputs[0 + round_]] + dataset.opener_input + input_models,
                outputs=FLTaskOutputGenerator.composite_traintuple(
                    shared_authorized_ids=[c.organization_id for c in clients],
                    local_authorized_ids=[dataset.owner],
                ),
                worker=workers[index],
            )

            t = clients[0].add_task(spec)
            t = clients[0].wait(t)
            composite_traintuple_keys.append(t.key)

        # create aggregate on its organization
        spec = factory.create_aggregatetuple(
            algo=aggregate_algo,
            worker=aggregate_worker,
            inputs=FLTaskInputGenerator.composites_to_aggregate(composite_traintuple_keys),
        )
        aggregatetuple = clients[0].add_task(spec)
        aggregatetuple = clients[0].wait(aggregatetuple)

        # save state of round
        previous_aggregatetuple_key = aggregatetuple.key
        previous_composite_traintuple_keys = composite_traintuple_keys

    # last round: create associated testtuple for composite and aggregate
    for index, (traintuple_key, metric, dataset) in enumerate(
        zip(previous_composite_traintuple_keys, default_metrics, default_datasets)
    ):
        spec = factory.create_predicttuple(
            algo=predict_algo_composite,
            inputs=dataset.test_data_inputs + FLTaskInputGenerator.composite_to_predict(traintuple_key),
            worker=workers[index],
        )
        predicttuple = clients[0].add_task(spec)
        predicttuple = clients[0].wait(predicttuple)

        spec = factory.create_testtuple(
            algo=metric,
            inputs=dataset.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttuple.key),
            worker=workers[index],
        )
        testtuple = clients[0].add_task(spec)
        testtuple = clients[0].wait(testtuple)
        # y_true: [20], y_pred: [52.0], result: 32.0
        assert testtuple.outputs[OutputIdentifiers.performance].value == pytest.approx(32 + index)

    spec = factory.create_predicttuple(
        algo=predict_algo,
        inputs=default_datasets[0].test_data_inputs
        + FLTaskInputGenerator.aggregate_to_predict(previous_aggregatetuple_key),
        worker=workers[0],
    )
    predicttuple = clients[0].add_task(spec)
    predicttuple = clients[0].wait(predicttuple)

    spec = factory.create_testtuple(
        algo=default_metrics[0],
        inputs=default_datasets[0].test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttuple.key),
        worker=workers[0],
    )
    testtuple = clients[0].add_task(spec)
    testtuple = clients[0].wait(testtuple)
    # y_true: [20], y_pred: [28.0], result: 8.0
    assert testtuple.outputs[OutputIdentifiers.performance].value == pytest.approx(8)

    if network.options.enable_model_download:
        # Optional (if "enable_model_download" is True): ensure we can export out-models.
        #
        # - One out-model download is not proxified (direct download)
        # - One out-model download is proxified (as it belongs to another org)
        for key in previous_composite_traintuple_keys:
            assert clients[0].download_model_from_task(key, identifier=OutputIdentifiers.shared) == b'{"value": 2.8}'

    if network.options.enable_intermediate_model_removal:
        # Optional (if "enable_intermediate_model_removal" is True): ensure the aggregatetuple of round 1 has been
        # deleted.
        #
        # We do this by creating a new traintuple that depends on the deleted aggregatatuple, and ensuring that starting
        # the traintuple fails.
        #
        # Ideally it would be better to try to do a request "as a backend" to get the deleted model. This would be
        # closer to what we want to test and would also check that this request is correctly handled when the model
        # has been deleted. Here, we cannot know for sure the failure reason. Unfortunately this cannot be done now
        # as the username/password are not available in the settings files.

        client = clients[0]
        dataset = default_datasets[0]
        algo = client.add_algo(spec)

        spec = factory.create_traintuple(algo=algo, inputs=dataset.train_data_inputs, worker=workers[0])
        traintuple = client.add_task(spec)
        traintuple = client.wait(traintuple)
        assert traintuple.status == Status.failed
        assert traintuple.error_type == substra.sdk.models.TaskErrorType.execution


@pytest.mark.remote_only
def test_use_data_sample_located_in_shared_path(factory, network, client, organization_cfg, default_metric, worker):
    if not organization_cfg.shared_path:
        pytest.skip("requires a shared path")

    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    spec = factory.create_data_sample(datasets=[dataset])
    spec.move_data_to_server(organization_cfg.shared_path, network.options.minikube)
    data_sample_key = client.add_data_sample(spec, local=False)  # should not raise

    dataset = AugmentedDataset(client.get_dataset(dataset.key))

    spec = factory.create_algo(AlgoCategory.simple)
    algo = client.add_algo(spec)

    spec = factory.create_algo(AlgoCategory.predict)
    predict_algo = client.add_algo(spec)

    spec = factory.create_traintuple(algo=algo, inputs=dataset.train_data_inputs, worker=worker)
    traintuple = client.add_task(spec)
    traintuple = client.wait(traintuple)
    assert traintuple.status == Status.done
    assert traintuple.error_type is None
    assert traintuple.outputs[OutputIdentifiers.model].value is not None

    # create testtuple
    spec = factory.create_predicttuple(
        algo=predict_algo, traintuple=traintuple, dataset=dataset, data_samples=[data_sample_key], worker=worker
    )
    predicttuple = client.add_task(spec)
    predicttuple = client.wait(predicttuple)
    assert predicttuple.status == Status.done
    assert predicttuple.error_type is None

    spec = factory.create_testtuple(
        algo=default_metric, predicttuple=predicttuple, dataset=dataset, data_samples=[data_sample_key], worker=worker
    )
    testtuple = client.add_task(spec)
    testtuple = client.wait(testtuple)
    assert testtuple.status == Status.done
    assert testtuple.error_type is None
    assert testtuple.outputs[OutputIdentifiers.performance].value == pytest.approx(2)


@pytest.mark.subprocess_skip
def test_user_creates_model_folder(factory, client, default_dataset, worker):
    """Check that the model folder is not overwritten by substra"""
    dockerfile = (
        f"FROM {factory.default_tools_image}\nCOPY algo.py .\nRUN mkdir model\n"
        + 'RUN echo \'{"name":"Jane"}\' >> model/model\nENTRYPOINT ["python3", "algo.py", "--method-name", "train"]\n'
    )
    algo_script = f"""
import json
import substratools as tools

from pathlib import Path
def train(inputs, outputs, task_properties):
    model_path = Path.cwd() / 'model' / 'model'
    assert model_path.is_file()
    loaded = json.loads(model_path.read_text())
    assert loaded == {{'name':'Jane'}}
    save_model(dict(), outputs['{OutputIdentifiers.model}'])


def predict(inputs, outputs, task_properties):
    save_predictions(None, outputs['{OutputIdentifiers.predictions}'])

def load_model(path):
    with open(path) as f:
        return json.load(f)

def save_model(model, path):
    with open(path, 'w') as f:
        return json.dump(model, f)

def save_predictions(predictions, path):
    with open(path, 'w') as f:
        return json.dump(predictions, f)

if __name__ == '__main__':
    tools.function.execute_cli([train, predict])
"""  # noqa
    spec = factory.create_algo(AlgoCategory.simple, py_script=algo_script, dockerfile=dockerfile)
    algo = client.add_algo(spec)
    spec = factory.create_traintuple(algo=algo, inputs=default_dataset.train_data_inputs, worker=worker)
    traintuple = client.add_task(spec)
    client.wait(traintuple)


WRITE_TO_HOME_DIRECTORY_ALGO = f"""
import json
import substratools as tools


def train(inputs, outputs, task_properties):

    from pathlib import Path
    with open(f"{{str(Path.home())}}/foo", "w") as f:
        f.write("test")

    save_model({{'value': 42 }}, outputs['{OutputIdentifiers.model}'])

def predict(inputs, outputs, task_properties):
    X = inputs['{InputIdentifiers.datasamples}'][0]
    model = load_model(inputs['{InputIdentifiers.model}'])

    res = [x * model['value'] for x in X]
    print(f'Predict, get X: {{X}}, model: {{model}}, return {{res}}')
    save_predictions(res, outputs['{OutputIdentifiers.predictions}'])

def load_model(path):
    with open(path) as f:
        return json.load(f)

def save_model(model, path):
    with open(path, 'w') as f:
        return json.dump(model, f)

def save_predictions(predictions, path):
    with open(path, 'w') as f:
        return json.dump(predictions, f)

if __name__ == '__main__':
    tools.function.execute_cli([train, predict])
"""  # noqa


@pytest.mark.subprocess_skip
def test_write_to_home_directory(factory, client, default_dataset, worker):
    """The algo writes to the home directory (~/foo)"""

    spec = factory.create_algo(AlgoCategory.simple, WRITE_TO_HOME_DIRECTORY_ALGO)
    algo = client.add_algo(spec)
    spec = factory.create_traintuple(algo=algo, inputs=default_dataset.train_data_inputs, worker=worker)
    traintuple = client.add_task(spec)
    traintuple = client.wait(traintuple)

    assert traintuple.status == Status.done
    assert traintuple.error_type is None
