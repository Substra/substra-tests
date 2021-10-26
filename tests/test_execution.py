import pytest

import substra
from substra.sdk.models import Status

import substratest as sbt
from substratest.factory import AlgoCategory, Permissions

from . import settings


@pytest.mark.slow
def test_tuples_execution_on_same_node(factory, network, client, default_dataset, default_metric, default_metric_local):
    """Execution of a traintuple, a following testtuple and a following traintuple."""

    spec = factory.create_algo(AlgoCategory.simple, local=client.debug)
    algo = client.add_algo(spec)

    # create traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
        metadata={"foo": "bar"}
    )
    traintuple = client.add_traintuple(spec)
    traintuple = client.wait(traintuple)
    assert traintuple.status == Status.done
    assert traintuple.metadata == {"foo": "bar"}
    assert len(traintuple.train.models) == 1

    if network.options.enable_model_download:
        model = traintuple.train.models[0]
        assert client.download_model(model.key) == b'{"value": 2.2}'

    # check we can add twice the same traintuple
    client.add_traintuple(spec)

    # create testtuple
    spec = factory.create_testtuple(
        metrics=[default_metric_local] if client.debug else [default_metric],
        traintuple=traintuple,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )
    testtuple = client.add_testtuple(spec)
    testtuple = client.wait(testtuple)
    assert testtuple.status == Status.done
    assert list(testtuple.test.perfs.values())[0] == 2

    # add a traintuple depending on first traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
        traintuples=[traintuple],
        metadata=None
    )
    traintuple = client.add_traintuple(spec)
    traintuple = client.wait(traintuple)
    assert traintuple.status == Status.done
    assert traintuple.metadata == {}
    assert len(traintuple.parent_task_keys) == 1


@pytest.mark.slow
def test_federated_learning_workflow(factory, client, default_datasets):
    """Test federated learning workflow on each node."""

    # create test environment
    spec = factory.create_algo(AlgoCategory.simple, local=client.debug)
    algo = client.add_algo(spec)

    # create 1 traintuple per dataset and chain them
    traintuple = None
    rank = 0
    compute_plan_key = None

    # default_datasets contains datasets on each node and
    # that has a result we can use for federated learning
    for dataset in default_datasets:
        traintuples = [traintuple] if traintuple else []
        spec = factory.create_traintuple(
            algo=algo,
            dataset=dataset,
            data_samples=dataset.train_data_sample_keys,
            traintuples=traintuples,
            tag='foo',
            rank=rank,
            compute_plan_key=compute_plan_key,
        )
        traintuple = client.add_traintuple(spec)
        traintuple = client.wait(traintuple)
        assert traintuple.status == Status.done
        assert len(traintuple.train.models) != 0
        assert traintuple.tag == 'foo'
        assert traintuple.compute_plan_key   # check it is not None or ''

        rank += 1
        compute_plan_key = traintuple.compute_plan_key

    # check a compute plan has been created and its status is at done
    cp = client.get_compute_plan(compute_plan_key)
    assert cp.status == 'PLAN_STATUS_DONE'


@pytest.mark.slow
@pytest.mark.remote_only
def test_tuples_execution_on_different_nodes(factory, client_1, client_2, default_metric_1,
                                             default_dataset_1, default_dataset_2):
    """Execution of a traintuple on node 1 and the following testtuple on node 2."""
    # add test data samples / dataset / metric on node 1

    spec = factory.create_algo(AlgoCategory.simple, local=client_2.debug)
    algo_2 = client_2.add_algo(spec)

    # add traintuple on node 2; should execute on node 2 (dataset located on node 2)
    spec = factory.create_traintuple(
        algo=algo_2,
        dataset=default_dataset_2,
        data_samples=default_dataset_2.train_data_sample_keys,
    )
    traintuple = client_1.add_traintuple(spec)
    traintuple = client_1.wait(traintuple)
    assert traintuple.status == Status.done
    assert len(traintuple.train.models) != 0
    assert traintuple.worker == client_2.node_id

    # add testtuple; should execute on node 1 (default_dataset_1 is located on node 1)
    spec = factory.create_testtuple(
        metrics=[default_metric_1],
        traintuple=traintuple,
        dataset=default_dataset_1,
        data_samples=default_dataset_1.test_data_sample_keys,
    )
    testtuple = client_1.add_testtuple(spec)
    testtuple = client_1.wait(testtuple)
    assert testtuple.status == Status.done
    assert testtuple.worker == client_1.node_id
    assert list(testtuple.test.perfs.values())[0] == 2


@pytest.mark.slow
def test_traintuple_execution_failure(factory, client, default_dataset_1):
    """Invalid algo script is causing traintuple failure."""

    spec = factory.create_algo(category=AlgoCategory.simple, py_script=sbt.factory.INVALID_ALGO_SCRIPT,
                               local=client.debug)
    algo = client.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset_1,
        data_samples=default_dataset_1.train_data_sample_keys,
    )
    if client.debug:
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.ExecutionError):
            traintuple = client.add_traintuple(spec)
    else:
        traintuple = client.add_traintuple(spec)
        traintuple = client.wait(traintuple, raises=False)
        assert traintuple.status == Status.failed
        assert traintuple.train.models is None


@pytest.mark.slow
def test_composite_traintuple_execution_failure(factory, client, default_dataset):
    """Invalid composite algo script is causing traintuple failure."""

    spec = factory.create_algo(AlgoCategory.composite, py_script=sbt.factory.INVALID_COMPOSITE_ALGO_SCRIPT,
                               local=client.debug)
    algo = client.add_algo(spec)

    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
    )
    if client.debug:
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.ExecutionError):
            composite_traintuple = client.add_composite_traintuple(spec)
    else:
        composite_traintuple = client.add_composite_traintuple(spec)
        composite_traintuple = client.wait(composite_traintuple, raises=False)
        assert composite_traintuple.status == Status.failed
        assert composite_traintuple.composite.models is None


@pytest.mark.slow
def test_aggregatetuple_execution_failure(factory, client, default_dataset):
    """Invalid algo script is causing traintuple failure."""

    spec = factory.create_algo(AlgoCategory.composite, local=client.debug)
    composite_algo = client.add_algo(spec)

    spec = factory.create_algo(AlgoCategory.aggregate, py_script=sbt.factory.INVALID_AGGREGATE_ALGO_SCRIPT,
                               local=client.debug)
    aggregate_algo = client.add_algo(spec)

    composite_traintuples = []
    for i in [0, 1]:
        spec = factory.create_composite_traintuple(
            algo=composite_algo,
            dataset=default_dataset,
            data_samples=[default_dataset.train_data_sample_keys[i]],
        )
        composite_traintuples.append(client.add_composite_traintuple(spec))

    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        traintuples=composite_traintuples,
        worker=client.node_id,
    )
    if client.debug:
        with pytest.raises(substra.sdk.backends.local.compute.spawner.base.ExecutionError):
            aggregatetuple = client.add_aggregatetuple(spec)
    else:
        aggregatetuple = client.add_aggregatetuple(spec)
        aggregatetuple = client.wait(aggregatetuple, raises=False)
        for composite_traintuple in composite_traintuples:
            composite_traintuple = client.get_composite_traintuple(composite_traintuple.key)
            assert composite_traintuple.status == Status.done
        assert aggregatetuple.status == Status.failed
        assert aggregatetuple.aggregate.models is None


@pytest.mark.slow
def test_composite_traintuples_execution(factory, client, default_dataset, default_metric, default_metric_local):
    """Execution of composite traintuples."""

    spec = factory.create_algo(AlgoCategory.composite, local=client.debug)
    algo = client.add_algo(spec)

    # first composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
    )
    composite_traintuple_1 = client.add_composite_traintuple(spec)
    composite_traintuple_1 = client.wait(composite_traintuple_1)
    assert composite_traintuple_1.status == Status.done
    assert len(composite_traintuple_1.composite.models) == 2

    # second composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
        head_traintuple=composite_traintuple_1,
        trunk_traintuple=composite_traintuple_1,
    )
    composite_traintuple_2 = client.add_composite_traintuple(spec)
    composite_traintuple_2 = client.wait(composite_traintuple_2)
    assert composite_traintuple_2.status == Status.done
    assert len(composite_traintuple_2.composite.models) == 2

    # add a 'composite' testtuple
    spec = factory.create_testtuple(
        metrics=[default_metric_local] if client.debug else [default_metric],
        traintuple=composite_traintuple_2,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )
    testtuple = client.add_testtuple(spec)
    testtuple = client.wait(testtuple)
    assert testtuple.status == Status.done
    assert list(testtuple.test.perfs.values())[0] == 32

    # list composite traintuple
    composite_traintuples = client.list_composite_traintuple()
    composite_traintuple_keys = set([t.key for t in composite_traintuples])
    assert set([composite_traintuple_1.key, composite_traintuple_2.key]).issubset(
        composite_traintuple_keys
    )


@pytest.mark.slow
def test_aggregatetuple(factory, client, default_metric, default_metric_local, default_dataset):
    """Execution of aggregatetuple aggregating traintuples."""

    number_of_traintuples_to_aggregate = 3

    train_data_sample_keys = default_dataset.train_data_sample_keys[:number_of_traintuples_to_aggregate]

    spec = factory.create_algo(AlgoCategory.simple, local=client.debug)
    algo = client.add_algo(spec)

    # add traintuples
    traintuples = []
    for data_sample_key in train_data_sample_keys:
        spec = factory.create_traintuple(
            algo=algo,
            dataset=default_dataset,
            data_samples=[data_sample_key],
        )
        traintuple = client.add_traintuple(spec)
        traintuple = client.wait(traintuple)
        traintuples.append(traintuple)

    spec = factory.create_algo(AlgoCategory.aggregate, local=client.debug)
    aggregate_algo = client.add_algo(spec)

    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        worker=client.node_id,
        traintuples=traintuples,
    )
    aggregatetuple = client.add_aggregatetuple(spec)
    aggregatetuple = client.wait(aggregatetuple)
    assert aggregatetuple.status == Status.done
    assert len(aggregatetuple.parent_task_keys) == number_of_traintuples_to_aggregate

    spec = factory.create_testtuple(
        metrics=[default_metric_local] if client.debug else [default_metric],
        traintuple=aggregatetuple,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )
    testtuple = client.add_testtuple(spec)
    testtuple = client.wait(testtuple)


@pytest.mark.slow
def test_aggregate_composite_traintuples(factory, network, clients, default_datasets, default_metrics,
                                         default_metrics_local):
    """Do 2 rounds of composite traintuples aggregations on multiple nodes.

    Compute plan details:

    Round 1:
    - Create 2 composite traintuples executed on two datasets located on node 1 and
      node 2.
    - Create an aggregatetuple on node 1, aggregating the two previous composite
      traintuples (trunk models aggregation).

    Round 2:
    - Create 2 composite traintuples executed on each nodes that depend on: the
      aggregated tuple and the previous composite traintuple executed on this node. That
      is to say, the previous round aggregated trunk models from all nodes and the
      previous round head model from this node.
    - Create an aggregatetuple on node 1, aggregating the two previous composite
      traintuples (similar to round 1 aggregatetuple).
    - Create a testtuple for each previous composite traintuples and aggregate tuple
      created during this round.

    (optional) if the option "enable_intermediate_model_removal" is True:
    - Since option "enable_intermediate_model_removal" is True, the aggregate model created on round 1 should
      have been deleted from the backend after round 2 has completed.
    - Create a traintuple that depends on the aggregate tuple created on round 1. Ensure that it fails to start.

    This test refers to the model composition use case.
    """

    aggregate_worker = clients[0].node_id
    number_of_rounds = 2

    # register algos on first node
    spec = factory.create_algo(AlgoCategory.composite, local=clients[0].debug)
    composite_algo = clients[0].add_algo(spec)
    spec = factory.create_algo(AlgoCategory.aggregate, local=clients[0].debug)
    aggregate_algo = clients[0].add_algo(spec)

    # launch execution
    previous_aggregatetuple = None
    previous_composite_traintuples = []

    for round_ in range(number_of_rounds):
        # create composite traintuple on each node
        composite_traintuples = []
        for index, dataset in enumerate(default_datasets):
            kwargs = {}
            if previous_aggregatetuple:
                kwargs = {
                    'head_traintuple': previous_composite_traintuples[index],
                    'trunk_traintuple': previous_aggregatetuple,
                }
            spec = factory.create_composite_traintuple(
                algo=composite_algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0 + round_]],
                permissions=Permissions(public=False, authorized_ids=[c.node_id for c in clients]),
                **kwargs,
            )
            t = clients[0].add_composite_traintuple(spec)
            t = clients[0].wait(t)
            composite_traintuples.append(t)

        # create aggregate on its node
        spec = factory.create_aggregatetuple(
            algo=aggregate_algo,
            worker=aggregate_worker,
            traintuples=composite_traintuples,
        )
        aggregatetuple = clients[0].add_aggregatetuple(spec)
        aggregatetuple = clients[0].wait(aggregatetuple)

        # save state of round
        previous_aggregatetuple = aggregatetuple
        previous_composite_traintuples = composite_traintuples

    # last round: create associated testtuple for composite and aggregate
    for index, (traintuple, metric, metric_local, dataset) in enumerate(zip(
            previous_composite_traintuples, default_metrics, default_metrics_local, default_datasets)):
        spec = factory.create_testtuple(
            metrics=[metric_local] if clients[0].debug else [metric],
            traintuple=traintuple,
            dataset=dataset,
            data_samples=dataset.test_data_sample_keys,
        )
        testtuple = clients[0].add_testtuple(spec)
        testtuple = clients[0].wait(testtuple)
        # y_true: [20], y_pred: [52.0], result: 32.0
        assert list(testtuple.test.perfs.values())[0] == 32 + index

    spec = factory.create_testtuple(
        metrics=[default_metrics_local[0]] if clients[0].debug else [default_metrics[0]],
        traintuple=previous_aggregatetuple,
        dataset=default_datasets[0],
        data_samples=default_datasets[0].test_data_sample_keys,
    )
    testtuple = clients[0].add_testtuple(spec)
    testtuple = clients[0].wait(testtuple)
    # y_true: [20], y_pred: [28.0], result: 8.0
    assert list(testtuple.test.perfs.values())[0] == 8

    if network.options.enable_model_download:
        # Optional (if "enable_model_download" is True): ensure we can export out-models.
        #
        # - One out-model download is not proxified (direct download)
        # - One out-model download is proxified (as it belongs to another org)
        for tuple in previous_composite_traintuples:
            assert clients[0].download_trunk_model_from_composite_traintuple(tuple.key) == b'{"value": 2.8}'

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

        spec = factory.create_traintuple(
            algo=algo,
            dataset=dataset,
            data_samples=dataset.train_data_sample_keys,
        )
        traintuple = client.add_traintuple(spec)
        traintuple = client.wait(traintuple)
        assert traintuple.status == Status.failed


@pytest.mark.remote_only
@pytest.mark.skipif(not settings.HAS_SHARED_PATH, reason='requires a shared path')
def test_use_data_sample_located_in_shared_path(factory, client, node_cfg, default_metric, default_metric_local):
    spec = factory.create_dataset()
    dataset = client.add_dataset(spec)

    spec = factory.create_data_sample(datasets=[dataset])
    spec.move_data_to_server(node_cfg.shared_path, settings.IS_MINIKUBE)
    data_sample_key = client.add_data_sample(spec, local=False)  # should not raise

    spec = factory.create_algo(AlgoCategory.simple, local=client.debug)
    algo = client.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=[data_sample_key],
    )
    traintuple = client.add_traintuple(spec)
    traintuple = client.wait(traintuple)
    assert traintuple.status == Status.done
    assert len(traintuple.train.models) == 1

    # create testtuple
    spec = factory.create_testtuple(
        metrics=[default_metric_local] if client.debug else [default_metric],
        traintuple=traintuple,
        dataset=dataset,
        data_samples=[data_sample_key],
    )
    testtuple = client.add_testtuple(spec)
    testtuple = client.wait(testtuple)
    assert testtuple.status == Status.done
    assert list(testtuple.test.perfs.values())[0] == 2
