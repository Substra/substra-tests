import substra

import pytest

import substratest as sbt


def test_tuples_execution_on_same_node(factory, session):
    """Execution of a traintuple, a following testtuple and a following traintuple."""
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    test_data_sample = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    train_data_sample = session.add_data_sample(spec)

    spec = factory.create_objective(
        dataset=dataset,
        data_samples=[test_data_sample],
    )
    objective = session.add_objective(spec)

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    # create traintuple
    spec = factory.create_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
    )
    traintuple = session.add_traintuple(spec).future().wait()
    assert traintuple.status == 'done'
    assert traintuple.out_model is not None

    # create testtuple
    # don't create it before to avoid MVCC errors
    spec = factory.create_testtuple(traintuple=traintuple)
    testtuple = session.add_testtuple(spec).future().wait()
    assert testtuple.status == 'done'

    # add a traintuple depending on first traintuple
    spec = factory.create_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
        traintuples=[traintuple],
    )
    traintuple = session.add_traintuple(spec).future().wait()
    assert traintuple.status == 'done'
    assert len(traintuple.in_models) == 1


def test_federated_learning_workflow(factory, session):
    """Test federated learning workflow."""
    # create test environment
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    test_data_sample = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    train_data_sample = session.add_data_sample(spec)

    spec = factory.create_objective(
        dataset=dataset,
        data_samples=[test_data_sample],
    )
    objective = session.add_objective(spec)

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    # create traintuple with rank 0
    spec = factory.create_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
        tag='foo',
        rank=0,
    )
    traintuple_1 = session.add_traintuple(spec).future().wait()
    assert traintuple_1.status == 'done'
    assert traintuple_1.out_model is not None
    assert traintuple_1.tag == 'foo'
    assert traintuple_1.compute_plan_id is not None

    with pytest.raises(substra.exceptions.AlreadyExists):
        session.add_traintuple(spec)

    # create traintuple with rank 1
    spec = factory.create_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
        traintuples=[traintuple_1],
        tag='foo',
        compute_plan_id=traintuple_1.compute_plan_id,
        rank=1,
    )
    traintuple_2 = session.add_traintuple(spec).future().wait()
    assert traintuple_2.status == 'done'
    assert traintuple_2.out_model is not None
    assert traintuple_2.tag == 'foo'
    assert traintuple_2.compute_plan_id == traintuple_1.compute_plan_id

    with pytest.raises(substra.exceptions.AlreadyExists):
        session.add_traintuple(spec)


def test_tuples_execution_on_different_nodes(factory, session_1, session_2):
    """Execution of a traintuplute on node 1 and the following testtuple on node 2."""
    # add test data samples / dataset / ojective on node 1
    spec = factory.create_dataset()
    dataset_1 = session_1.add_dataset(spec)
    spec = factory.create_data_sample(test_only=True, datasets=[dataset_1])
    test_data_sample_1 = session_1.add_data_sample(spec)
    spec = factory.create_objective(
        dataset=dataset_1,
        data_samples=[test_data_sample_1],
    )
    objective_1 = session_1.add_objective(spec)

    # add train data samples / dataset / algo on node 2
    spec = factory.create_dataset()
    dataset_2 = session_2.add_dataset(spec)
    spec = factory.create_data_sample(test_only=False, datasets=[dataset_2])
    train_data_sample_2 = session_2.add_data_sample(spec)
    spec = factory.create_algo()
    algo_2 = session_2.add_algo(spec)

    # add traintuple on node 2; should execute on node 2 (dataset located on node 2)
    spec = factory.create_traintuple(
        algo=algo_2,
        objective=objective_1,
        dataset=dataset_2,
        data_samples=[train_data_sample_2],
    )
    traintuple = session_1.add_traintuple(spec).future().wait()
    assert traintuple.status == 'done'
    assert traintuple.out_model is not None
    assert traintuple.dataset.worker == session_2.node_id

    # add testtuple; should execute on node 1 (objective dataset is located on node 1)
    spec = factory.create_testtuple(traintuple=traintuple)
    testtuple = session_1.add_testtuple(spec).future().wait()
    assert testtuple.status == 'done'
    assert testtuple.dataset.worker == session_1.node_id


def test_traintuple_execution_failure(factory, session):
    """Invalid algo script is causing traintuple failure."""
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    test_data_sample = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    train_data_sample = session.add_data_sample(spec)

    spec = factory.create_objective(
        dataset=dataset,
        data_samples=[test_data_sample],
    )
    objective = session.add_objective(spec)

    spec = factory.create_algo(py_script=sbt.factory.INVALID_ALGO_SCRIPT)
    algo = session.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
    )
    traintuple = session.add_traintuple(spec).future().wait()
    assert traintuple.status == 'failed'
    assert traintuple.out_model is None


def test_composite_traintuples_execution(factory, session):
    """Execution of composite traintuples."""
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    spec = factory.create_data_sample(test_only=True, datasets=[dataset])
    test_data_sample = session.add_data_sample(spec)

    spec = factory.create_data_sample(test_only=False, datasets=[dataset])
    train_data_sample = session.add_data_sample(spec)

    spec = factory.create_objective(
        dataset=dataset,
        data_samples=[test_data_sample],
    )
    objective = session.add_objective(spec)

    spec = factory.create_composite_algo()
    algo = session.add_composite_algo(spec)

    # first composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
    )
    composite_traintuple_1 = session.add_composite_traintuple(spec).future().wait()
    assert composite_traintuple_1.status == 'done'
    assert composite_traintuple_1.out_head_model is not None
    assert composite_traintuple_1.out_head_model.out_model is not None
    assert composite_traintuple_1.out_trunk_model is not None
    assert composite_traintuple_1.out_trunk_model.out_model is not None

    # second composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        objective=objective,
        dataset=dataset,
        data_samples=[train_data_sample],
        head_traintuple=composite_traintuple_1,
        trunk_traintuple=composite_traintuple_1,
    )
    composite_traintuple_2 = session.add_composite_traintuple(spec).future().wait()
    assert composite_traintuple_2.status == 'done'
    assert composite_traintuple_2.out_head_model is not None
    assert composite_traintuple_2.out_trunk_model is not None

    # add a 'composite' testtuple
    spec = factory.create_testtuple(traintuple=composite_traintuple_2)
    testtuple = session.add_testtuple(spec).future().wait()
    assert testtuple.status == 'done'

    # list composite traintuple
    composite_traintuples = session.list_composite_traintuple()
    composite_traintuple_keys = set([t.key for t in composite_traintuples])
    assert set([composite_traintuple_1.key, composite_traintuple_2.key]).issubset(
        composite_traintuple_keys
    )


def test_aggregatetuples(factory, session):
    """Execution of Aggregatetuples."""
    number_of_traintuples_to_aggregate = 3
    spec = factory.create_dataset()
    dataset = session.add_dataset(spec)

    train_data_samples = []
    for _ in range(number_of_traintuples_to_aggregate):
        spec = factory.create_data_sample(test_only=False, datasets=[dataset])
        data_sample = session.add_data_sample(spec)
        train_data_samples.append(data_sample)

    spec = factory.create_objective(
        dataset=dataset,
        data_samples=[],
    )
    objective = session.add_objective(spec)

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    # add traintuples
    traintuples = []
    for data_sample in train_data_samples:
        spec = factory.create_traintuple(
            algo=algo,
            objective=objective,
            dataset=dataset,
            data_samples=[data_sample],
        )
        traintuple = session.add_traintuple(spec).future().wait()
        traintuples.append(traintuple)

    spec = factory.create_aggregate_algo()
    aggregate_algo = session.add_aggregate_algo(spec)

    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        objective=objective,
        worker=session.node_id,
        traintuples=traintuples,
    )
    aggregatetuple = session.add_aggregatetuple(spec).future().wait()
    assert aggregatetuple.status == 'done'


def test_aggregate_composite_traintuples(factory, session_1, session_2):
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

    This test refers to the model composition use case.
    """
    aggregate_worker = session_1.node_id
    sessions = [session_1, session_2]
    number_of_rounds = 2

    # register objectives, datasets, and data samples
    datasets = []
    for s in sessions:
        # register one dataset per node
        spec = factory.create_dataset()
        dataset = s.add_dataset(spec)
        datasets.append(dataset)

        # register one data sample per dataset per round of aggregation
        for _ in range(number_of_rounds):
            spec = factory.create_data_sample(test_only=False, datasets=[dataset])
            s.add_data_sample(spec)
    # register test data on first node
    spec = factory.create_data_sample(test_only=True, datasets=[datasets[0]])
    test_data_sample = sessions[0].add_data_sample(spec)
    # register objective on first node
    spec = factory.create_objective(
        dataset=datasets[0],
        data_samples=[test_data_sample],
    )
    objective = sessions[0].add_objective(spec)

    # register algos on first node
    spec = factory.create_composite_algo()
    composite_algo = sessions[0].add_composite_algo(spec)
    spec = factory.create_aggregate_algo()
    aggregate_algo = sessions[0].add_aggregate_algo(spec)

    # launch execution
    previous_aggregatetuple = None
    previous_composite_traintuples = []

    for round_ in range(number_of_rounds):
        # create composite traintuple on each node
        composite_traintuples = []
        for index, dataset in enumerate(datasets):
            kwargs = {}
            if previous_aggregatetuple:
                kwargs = {
                    'head_traintuple': previous_composite_traintuples[index],
                    'trunk_traintuple': previous_aggregatetuple,
                }
            spec = factory.create_composite_traintuple(
                algo=composite_algo,
                objective=objective,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0 + round_]],
                **kwargs,
            )
            t = sessions[0].add_composite_traintuple(spec).future().wait()
            composite_traintuples.append(t)

        # create aggregate on its node
        spec = factory.create_aggregatetuple(
            algo=aggregate_algo,
            objective=objective,
            worker=aggregate_worker,
            traintuples=composite_traintuples,
        )
        aggregatetuple = sessions[0].add_aggregatetuple(spec).future().wait()

        # save state of round
        previous_aggregatetuple = aggregatetuple
        previous_composite_traintuples = composite_traintuples

    # last round: create associated testtuple
    for traintuple in previous_composite_traintuples:
        spec = factory.create_testtuple(
            traintuple=traintuple,
        )
        sessions[0].add_testtuple(spec).future().wait()
    spec = factory.create_testtuple(
        traintuple=previous_aggregatetuple,
    )
    sessions[0].add_testtuple(spec).future().wait()
