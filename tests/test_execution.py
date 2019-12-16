import pytest

import substra

import substratest as sbt

from substratest import assets


def test_tuples_execution_on_same_node(global_execution_env):
    """Execution of a traintuple, a following testtuple and a following traintuple."""
    factory, network = global_execution_env
    session = network.sessions[0].copy()

    dataset = session.state.datasets[0]
    objective = session.state.objectives[0]

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    # create traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=dataset.train_data_sample_keys,
    )
    traintuple = session.add_traintuple(spec).future().wait()
    assert traintuple.status == assets.Status.done
    assert traintuple.out_model is not None

    # create testtuple
    # don't create it before to avoid MVCC errors
    spec = factory.create_testtuple(objective=objective, traintuple=traintuple)
    testtuple = session.add_testtuple(spec).future().wait()
    assert testtuple.status == assets.Status.done

    # add a traintuple depending on first traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=dataset.train_data_sample_keys,
        traintuples=[traintuple],
    )
    traintuple = session.add_traintuple(spec).future().wait()
    assert traintuple.status == assets.Status.done
    assert len(traintuple.in_models) == 1


def test_federated_learning_workflow(global_execution_env):
    """Test federated learning workflow."""
    factory, network = global_execution_env
    session = network.sessions[0].copy()

    # get test environment
    dataset = session.state.datasets[0]

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    # create traintuple with rank 0
    spec = factory.create_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=dataset.train_data_sample_keys,
        tag='foo',
        rank=0,
    )
    traintuple_1 = session.add_traintuple(spec).future().wait()
    assert traintuple_1.status == assets.Status.done
    assert traintuple_1.out_model is not None
    assert traintuple_1.tag == 'foo'
    assert traintuple_1.compute_plan_id is not None

    with pytest.raises(substra.exceptions.AlreadyExists):
        session.add_traintuple(spec)

    # create traintuple with rank 1
    spec = factory.create_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=dataset.train_data_sample_keys,
        traintuples=[traintuple_1],
        tag='foo',
        compute_plan_id=traintuple_1.compute_plan_id,
        rank=1,
    )
    traintuple_2 = session.add_traintuple(spec).future().wait()
    assert traintuple_2.status == assets.Status.done
    assert traintuple_2.out_model is not None
    assert traintuple_2.tag == 'foo'
    assert traintuple_2.compute_plan_id == traintuple_1.compute_plan_id

    with pytest.raises(substra.exceptions.AlreadyExists):
        session.add_traintuple(spec)


def test_tuples_execution_on_different_nodes(global_execution_env):
    """Execution of a traintuple on node 1 and the following testtuple on node 2."""
    # add test data samples / dataset / objective on node 1
    factory, network = global_execution_env
    session_1 = network.sessions[0].copy()
    session_2 = network.sessions[1].copy()

    objective_1 = session_1.state.objectives[0]
    dataset_2 = session_2.state.datasets[0]

    spec = factory.create_algo()
    algo_2 = session_2.add_algo(spec)

    # add traintuple on node 2; should execute on node 2 (dataset located on node 2)
    spec = factory.create_traintuple(
        algo=algo_2,
        dataset=dataset_2,
        data_samples=dataset_2.train_data_sample_keys,
    )
    traintuple = session_1.add_traintuple(spec).future().wait()
    assert traintuple.status == assets.Status.done
    assert traintuple.out_model is not None
    assert traintuple.dataset.worker == session_2.node_id

    # add testtuple; should execute on node 1 (objective dataset is located on node 1)
    spec = factory.create_testtuple(objective=objective_1, traintuple=traintuple)
    testtuple = session_1.add_testtuple(spec).future().wait()
    assert testtuple.status == assets.Status.done
    assert testtuple.dataset.worker == session_1.node_id


def test_traintuple_execution_failure(global_execution_env):
    """Invalid algo script is causing traintuple failure."""
    factory, network = global_execution_env
    session = network.sessions[0].copy()

    dataset = session.state.datasets[0]

    spec = factory.create_algo(py_script=sbt.factory.INVALID_ALGO_SCRIPT)
    algo = session.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=dataset.train_data_sample_keys,
    )
    traintuple = session.add_traintuple(spec).future().wait(raises=False)
    assert traintuple.status == assets.Status.failed
    assert traintuple.out_model is None


def test_composite_traintuples_execution(global_execution_env):
    """Execution of composite traintuples."""

    factory, network = global_execution_env
    session = network.sessions[0].copy()

    dataset = session.state.datasets[0]
    objective = session.state.objectives[0]

    spec = factory.create_composite_algo()
    algo = session.add_composite_algo(spec)

    # first composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=dataset.train_data_sample_keys,
    )
    composite_traintuple_1 = session.add_composite_traintuple(spec).future().wait()
    assert composite_traintuple_1.status == assets.Status.done
    assert composite_traintuple_1.out_head_model is not None
    assert composite_traintuple_1.out_head_model.out_model is not None
    assert composite_traintuple_1.out_trunk_model is not None
    assert composite_traintuple_1.out_trunk_model.out_model is not None

    # second composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=dataset.train_data_sample_keys,
        head_traintuple=composite_traintuple_1,
        trunk_traintuple=composite_traintuple_1,
    )
    composite_traintuple_2 = session.add_composite_traintuple(spec).future().wait()
    assert composite_traintuple_2.status == assets.Status.done
    assert composite_traintuple_2.out_head_model is not None
    assert composite_traintuple_2.out_trunk_model is not None

    # add a 'composite' testtuple
    spec = factory.create_testtuple(objective=objective, traintuple=composite_traintuple_2)
    testtuple = session.add_testtuple(spec).future().wait()
    assert testtuple.status == assets.Status.done

    # list composite traintuple
    composite_traintuples = session.list_composite_traintuple()
    composite_traintuple_keys = set([t.key for t in composite_traintuples])
    assert set([composite_traintuple_1.key, composite_traintuple_2.key]).issubset(
        composite_traintuple_keys
    )


def test_aggregatetuple(global_execution_env):
    """Execution of aggregatetuple aggregating traintuples."""

    number_of_traintuples_to_aggregate = 3

    factory, network = global_execution_env
    session = network.sessions[0].copy()

    dataset = session.state.datasets[0]
    train_data_sample_keys = dataset.train_data_sample_keys[:number_of_traintuples_to_aggregate]

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    # add traintuples
    traintuples = []
    for data_sample_key in train_data_sample_keys:
        spec = factory.create_traintuple(
            algo=algo,
            dataset=dataset,
            data_samples=[data_sample_key],
        )
        traintuple = session.add_traintuple(spec).future().wait()
        traintuples.append(traintuple)

    spec = factory.create_aggregate_algo()
    aggregate_algo = session.add_aggregate_algo(spec)

    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        worker=session.node_id,
        traintuples=traintuples,
    )
    aggregatetuple = session.add_aggregatetuple(spec).future().wait()
    assert aggregatetuple.status == assets.Status.done
    assert len(aggregatetuple.in_models) == number_of_traintuples_to_aggregate


def test_aggregate_composite_traintuples(global_execution_env):
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
    factory, network = global_execution_env
    sessions = [s.copy() for s in network.sessions]

    aggregate_worker = sessions[0].node_id
    number_of_rounds = 2

    datasets = sessions[0].state.datasets + sessions[1].state.datasets
    objective = sessions[0].state.objectives[0]

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
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0 + round_]],
                **kwargs,
            )
            t = sessions[0].add_composite_traintuple(spec).future().wait()
            composite_traintuples.append(t)

        # create aggregate on its node
        spec = factory.create_aggregatetuple(
            algo=aggregate_algo,
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
            objective=objective,
            traintuple=traintuple,
        )
        sessions[0].add_testtuple(spec).future().wait()

    if not network.options.enable_intermediate_model_removal:
        return

    # Optional (if "enable_intermediate_model_removal" is True): ensure the aggregatetuple of round 1 has been deleted
    session = sessions[0]
    dataset = session.state.datasets[0]
    algo = session.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=dataset.train_data_sample_keys,
    )
    traintuple = session.add_traintuple(spec).future().wait()
    assert traintuple.status == assets.Status.failed


@pytest.mark.parametrize('fail_count', [1, 2])
def test_execution_retry_on_fail(fail_count, global_execution_env):
    """Execution of a traintuple which fails on the N first tries, and suceeds on the N+1th try"""

    # This test ensures the compute task retry mechanism works correctly.
    #
    # It executes an algorithm that `raise`s on the first N runs, and then
    # succeeds.
    #
    # /!\ This test should ideally be part of the substra-backend project,
    #     not substra-tests. For the sake of expendiency, we're keeping it
    #     as part of substra-tests for now, but we intend to re-implement
    #     it in substra-backend instead eventually.
    # /!\ This test makes use of the "local" folder to keep track of a counter.
    #     This is a hack to make the algo raise or succeed depending on the retry
    #     count. Ideally, we would use a more elegant solution.
    # /!\ This test doesn't validate that an error in the docker build phase (of
    #     the compute task execution) triggers a retry. Ideally, it's that case that
    #     would be tested, since errors in the docker build are the main use-case
    #     the retry feature was build for.

    retry_algo_snippet_toreplace = """
    tools.algo.execute(TestAlgo())"""

    retry_snippet_replacement = f"""
    counter_path = "/sandbox/local/counter"
    counter = 0
    try:
        with open(counter_path) as f:
            counter = int(f.read())
    except IOError:
        pass # file doesn't exist yet

    # Fail if the counter is below the retry count
    if counter < {fail_count}:
        counter = counter + 1
        with open(counter_path, 'w') as f:
            f.write(str(counter))
        raise Exception("Intentionally keep on failing until we have failed {fail_count} time(s). The algo has now \
            failed " + str(counter) + " time(s).")

    # The counter is greater than the retry count
    tools.algo.execute(TestAlgo())"""

    factory, network = global_execution_env
    session = network.sessions[0].copy()

    dataset = session.state.datasets[0]

    py_script = sbt.factory.DEFAULT_ALGO_SCRIPT.replace(retry_algo_snippet_toreplace, retry_snippet_replacement)
    spec = factory.create_algo(py_script)
    algo = session.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=dataset.train_data_sample_keys,
        rank=0,  # make sure it's part of a compute plan, so we have access to the /sandbox/local
                 # folder (that's where we store the counter)
    )
    traintuple = session.add_traintuple(spec).future().wait(raises=False)

    # Assuming that, on the backend, CELERY_TASK_MAX_RETRIES is set to 1, the algo
    # should be retried up to 1 time(s) (i.e. max 2 attempts in total)
    # - if it fails less than 2 times, it should be marked as "done"
    # - if it fails 2 times or more, it should be marked as "failed"
    if fail_count < 2:
        assert traintuple.status == 'done'
    else:
        assert traintuple.status == 'failed'
