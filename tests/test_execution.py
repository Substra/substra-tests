import pytest

import substra
from substra.sdk.models import Status

import substratest as sbt
from substratest.factory import Permissions


@pytest.mark.slow
def test_tuples_execution_on_same_node(factory, network, client, default_dataset, default_objective):
    """Execution of a traintuple, a following testtuple and a following traintuple."""

    spec = factory.create_algo()
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
    assert traintuple.out_model is not None

    if network.options.enable_model_download:
        assert client.download_model(traintuple.out_model.key) == b'{"value": 2.2}'

    # check we can add twice the same traintuple
    client.add_traintuple(spec)

    # create testtuple
    # don't create it before to avoid MVCC errors
    spec = factory.create_testtuple(objective=default_objective, traintuple=traintuple)
    testtuple = client.add_testtuple(spec)
    testtuple = client.wait(testtuple)
    assert testtuple.status == Status.done
    assert testtuple.dataset.perf == 2

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
    assert len(traintuple.in_models) == 1


@pytest.mark.slow
def test_federated_learning_workflow(factory, client, default_datasets):
    """Test federated learning workflow on each node."""

    # create test environment
    spec = factory.create_algo()
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
        assert traintuple.out_model is not None
        assert traintuple.tag == 'foo'
        assert traintuple.compute_plan_key   # check it is not None or ''

        rank += 1
        compute_plan_key = traintuple.compute_plan_key

    # check a compute plan has been created and its status is at done
    cp = client.get_compute_plan(compute_plan_key)
    assert cp.status == Status.done


@pytest.mark.slow
@pytest.mark.remote_only
def test_tuples_execution_on_different_nodes(factory, client_1, client_2, default_objective_1, default_dataset_2):
    """Execution of a traintuple on node 1 and the following testtuple on node 2."""
    # add test data samples / dataset / objective on node 1

    spec = factory.create_algo()
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
    assert traintuple.out_model is not None
    assert traintuple.dataset.worker == client_2.node_id

    # add testtuple; should execute on node 1 (objective dataset is located on node 1)
    spec = factory.create_testtuple(objective=default_objective_1, traintuple=traintuple)
    testtuple = client_1.add_testtuple(spec)
    testtuple = client_1.wait(testtuple)
    assert testtuple.status == Status.done
    assert testtuple.dataset.worker == client_1.node_id
    assert testtuple.dataset.perf == 2


@pytest.mark.slow
def test_traintuple_execution_failure(factory, client, default_dataset_1):
    """Invalid algo script is causing traintuple failure."""

    spec = factory.create_algo(py_script=sbt.factory.INVALID_ALGO_SCRIPT)
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
        assert traintuple.out_model is None


@pytest.mark.slow
def test_composite_traintuple_execution_failure(factory, client, default_dataset):
    """Invalid composite algo script is causing traintuple failure."""

    spec = factory.create_composite_algo(py_script=sbt.factory.INVALID_COMPOSITE_ALGO_SCRIPT)
    algo = client.add_composite_algo(spec)

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
        assert composite_traintuple.out_head_model.out_model is None
        assert composite_traintuple.out_trunk_model.out_model is None


@pytest.mark.slow
def test_aggregatetuple_execution_failure(factory, client, default_dataset):
    """Invalid algo script is causing traintuple failure."""

    spec = factory.create_composite_algo()
    composite_algo = client.add_composite_algo(spec)

    spec = factory.create_aggregate_algo(py_script=sbt.factory.INVALID_AGGREGATE_ALGO_SCRIPT)
    aggregate_algo = client.add_aggregate_algo(spec)

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
        assert aggregatetuple.out_model is None


@pytest.mark.slow
def test_composite_traintuples_execution(factory, client, default_dataset, default_objective):
    """Execution of composite traintuples."""

    spec = factory.create_composite_algo()
    algo = client.add_composite_algo(spec)

    # first composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
    )
    composite_traintuple_1 = client.add_composite_traintuple(spec)
    composite_traintuple_1 = client.wait(composite_traintuple_1)
    assert composite_traintuple_1.status == Status.done
    assert composite_traintuple_1.out_head_model is not None
    assert composite_traintuple_1.out_head_model.out_model is not None
    assert composite_traintuple_1.out_trunk_model is not None
    assert composite_traintuple_1.out_trunk_model.out_model is not None

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
    assert composite_traintuple_2.out_head_model is not None
    assert composite_traintuple_2.out_trunk_model is not None

    # add a 'composite' testtuple
    spec = factory.create_testtuple(objective=default_objective, traintuple=composite_traintuple_2)
    testtuple = client.add_testtuple(spec)
    testtuple = client.wait(testtuple)
    assert testtuple.status == Status.done
    assert testtuple.dataset.perf == 32

    # list composite traintuple
    composite_traintuples = client.list_composite_traintuple()
    composite_traintuple_keys = set([t.key for t in composite_traintuples])
    assert set([composite_traintuple_1.key, composite_traintuple_2.key]).issubset(
        composite_traintuple_keys
    )


@pytest.mark.slow
def test_aggregatetuple(factory, client, default_objective, default_dataset):
    """Execution of aggregatetuple aggregating traintuples."""

    number_of_traintuples_to_aggregate = 3

    train_data_sample_keys = default_dataset.train_data_sample_keys[:number_of_traintuples_to_aggregate]

    spec = factory.create_algo()
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

    spec = factory.create_aggregate_algo()
    aggregate_algo = client.add_aggregate_algo(spec)

    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        worker=client.node_id,
        traintuples=traintuples,
    )
    aggregatetuple = client.add_aggregatetuple(spec)
    aggregatetuple = client.wait(aggregatetuple)
    assert aggregatetuple.status == Status.done
    assert len(aggregatetuple.in_models) == number_of_traintuples_to_aggregate

    spec = factory.create_testtuple(
        objective=default_objective,
        traintuple=aggregatetuple,
        dataset=default_dataset,
        data_samples=default_dataset.test_data_sample_keys,
    )
    testtuple = client.add_testtuple(spec)
    testtuple = client.wait(testtuple)


@pytest.mark.slow
def test_aggregate_composite_traintuples(factory, network, clients, default_datasets, default_objectives):
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
    spec = factory.create_composite_algo()
    composite_algo = clients[0].add_composite_algo(spec)
    spec = factory.create_aggregate_algo()
    aggregate_algo = clients[0].add_aggregate_algo(spec)

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
    for traintuple, objective in zip(previous_composite_traintuples, default_objectives):
        spec = factory.create_testtuple(
            objective=objective,
            traintuple=traintuple,
        )
        testtuple = clients[0].add_testtuple(spec)
        testtuple = clients[0].wait(testtuple)
        # y_true: [20], y_pred: [52.0], result: 32.0
        assert testtuple.dataset.perf == 32
    spec = factory.create_testtuple(
        objective=objective,
        traintuple=previous_aggregatetuple,
        dataset=default_datasets[0],
        data_samples=default_datasets[0].test_data_sample_keys,
    )
    testtuple = clients[0].add_testtuple(spec)
    testtuple = clients[0].wait(testtuple)
    # y_true: [20], y_pred: [28.0], result: 8.0
    assert testtuple.dataset.perf == 8

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
