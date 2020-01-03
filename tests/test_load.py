import pytest

from substratest import assets

# COMPUTE_PLAN_SIZES = [10, 100, 1000, 10000]
COMPUTE_PLAN_SIZES = [3, 30]


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_linear_traintuples(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A: traintuple -> traintuple -> ... -> traintuple
    Node B: traintuple -> traintuple -> ... -> traintuple
    """

    factory, network = global_execution_env
    session_1 = network.sessions[0].copy()
    session_2 = network.sessions[1].copy()

    dataset_1 = session_1.state.datasets[0]
    dataset_2 = session_2.state.datasets[0]

    spec = factory.create_algo()
    algo = session_1.add_algo(spec)

    cp_spec = factory.create_compute_plan()

    for dataset in [dataset_1, dataset_2]:
        previous_tuple = None
        for _ in range(compute_plan_size):
            previous_tuple = cp_spec.add_traintuple(
                algo=algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0]],
                in_models=[previous_tuple] if previous_tuple else [],
            )

    cp = session_1.add_compute_plan(cp_spec)
    first_tuple_1, first_tuple_2 = cp.list_traintuple()[:2]
    assert first_tuple_1.rank == 0
    assert first_tuple_2.rank == 0
    first_tuple_1 = first_tuple_1.future().wait()
    assert first_tuple_1.status == assets.Status.done
    first_tuple_2 = first_tuple_2.future().wait()
    assert first_tuple_2.status == assets.Status.done

    cp = session_1.cancel_compute_plan(cp.compute_plan_id).future().wait()
    assert cp.status == assets.Status.canceled

    first_tuple_1, first_tuple_2 = cp.list_traintuple()[:2]
    assert first_tuple_1.rank == 0
    assert first_tuple_2.rank == 0
    first_tuple_1 = first_tuple_1.future().wait()
    assert first_tuple_1.status == assets.Status.done
    first_tuple_2 = first_tuple_2.future().wait()
    assert first_tuple_2.status == assets.Status.done


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_linear_traintuples_manual(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A: traintuple -> traintuple -> ... -> traintuple
    Node B: traintuple -> traintuple -> ... -> traintuple
    """

    factory, network = global_execution_env
    session_1 = network.sessions[0].copy()
    session_2 = network.sessions[1].copy()

    dataset_1 = session_1.state.datasets[0]
    dataset_2 = session_2.state.datasets[0]

    spec = factory.create_algo()
    algo = session_1.add_algo(spec)

    previous_tuples = None
    compute_plan_id = None
    for rank in range(compute_plan_size):
        spec = factory.create_traintuple(
            algo=algo,
            dataset=dataset_1,
            data_samples=[dataset_1.train_data_sample_keys[0]],
            traintuples=[previous_tuples[0]] if previous_tuples else [],
            compute_plan_id=compute_plan_id,
            rank=rank,
        )
        tuple_1 = session_1.add_traintuple(spec)
        # necessary at first step so that we fetch the actual compute plan
        compute_plan_id = tuple_1.compute_plan_id

        spec = factory.create_traintuple(
            algo=algo,
            dataset=dataset_2,
            data_samples=[dataset_2.train_data_sample_keys[0]],
            traintuples=[previous_tuples[1]] if previous_tuples else [],
            compute_plan_id=compute_plan_id,
            rank=rank,
        )
        tuple_2 = session_1.add_traintuple(spec)
        previous_tuples = [tuple_1, tuple_2]

    # cp = session_1.get_compute_plan(compute_plan_id)
    # first_tuple_1, first_tuple_2 = cp.list_traintuple()[:2]
    # assert first_tuple_1.rank == 0
    # assert first_tuple_2.rank == 0
    # first_tuple_1 = first_tuple_1.future().wait()
    # assert first_tuple_1.status == assets.Status.done
    # first_tuple_2 = first_tuple_2.future().wait()
    # assert first_tuple_2.status == assets.Status.done

    # cp = session_1.cancel_compute_plan(cp.compute_plan_id).future().wait()
    # assert cp.status == assets.Status.canceled

    # first_tuple_1, first_tuple_2 = cp.list_traintuple()[:2]
    # assert first_tuple_1.rank == 0
    # assert first_tuple_2.rank == 0
    # first_tuple_1 = first_tuple_1.future().wait()
    # assert first_tuple_1.status == assets.Status.done
    # first_tuple_2 = first_tuple_2.future().wait()
    # assert first_tuple_2.status == assets.Status.done


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_linear_composite_traintuples(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A: composite_traintuple -> composite_traintuple -> ... -> composite_traintuple
    Node B: composite_traintuple -> composite_traintuple -> ... -> composite_traintuple
    """

    factory, network = global_execution_env
    session_1 = network.sessions[0].copy()
    session_2 = network.sessions[1].copy()

    dataset_1 = session_1.state.datasets[0]
    dataset_2 = session_2.state.datasets[0]

    spec = factory.create_composite_algo()
    algo = session_1.add_composite_algo(spec)

    cp_spec = factory.create_compute_plan()

    for dataset in [dataset_1, dataset_2]:
        previous_tuple = None
        for _ in range(compute_plan_size):
            previous_tuple = cp_spec.add_composite_traintuple(
                composite_algo=algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0]],
                in_head_model=previous_tuple if previous_tuple else None,
                in_trunk_model=previous_tuple if previous_tuple else None,
            )

    cp = session_1.add_compute_plan(cp_spec)
    first_tuple_1, first_tuple_2 = cp.list_composite_traintuple()[:2]
    assert first_tuple_1.rank == 0
    assert first_tuple_2.rank == 0
    first_tuple_1 = first_tuple_1.future().wait()
    assert first_tuple_1.status == assets.Status.done
    first_tuple_2 = first_tuple_2.future().wait()
    assert first_tuple_2.status == assets.Status.done

    cp = session_1.cancel_compute_plan(cp.compute_plan_id).future().wait()
    assert cp.status == assets.Status.canceled

    first_tuple_1, first_tuple_2 = cp.list_composite_traintuple()[:2]
    assert first_tuple_1.rank == 0
    assert first_tuple_2.rank == 0
    first_tuple_1 = first_tuple_1.future().wait()
    assert first_tuple_1.status == assets.Status.done
    first_tuple_2 = first_tuple_2.future().wait()
    assert first_tuple_2.status == assets.Status.done


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_tangled_traintuples(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A: traintuple \-/-> traintuple \-/-> ... \-/-> traintuple
                        x                x         x
    Node B: traintuple /-\-> traintuple /-\-> ... /-\-> traintuple
    """

    factory, network = global_execution_env
    session_1 = network.sessions[0].copy()
    session_2 = network.sessions[1].copy()

    dataset_1 = session_1.state.datasets[0]
    dataset_2 = session_2.state.datasets[0]

    spec = factory.create_algo()
    algo = session_1.add_algo(spec)

    cp_spec = factory.create_compute_plan()

    previous_tuples = []
    for _ in range(compute_plan_size):
        previous_tuples = [
            cp_spec.add_traintuple(
                algo=algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0]],
                in_models=previous_tuples,
            )
            for dataset in [dataset_1, dataset_2]
        ]

    cp = session_1.add_compute_plan(cp_spec)
    first_tuple_1, first_tuple_2 = cp.list_traintuple()[:2]
    assert first_tuple_1.rank == 0
    assert first_tuple_2.rank == 0
    first_tuple_1 = first_tuple_1.future().wait()
    assert first_tuple_1.status == assets.Status.done
    first_tuple_2 = first_tuple_2.future().wait()
    assert first_tuple_2.status == assets.Status.done

    cp = session_1.cancel_compute_plan(cp.compute_plan_id).future().wait()
    assert cp.status == assets.Status.canceled

    first_tuple_1, first_tuple_2 = cp.list_traintuple()[:2]
    assert first_tuple_1.rank == 0
    assert first_tuple_2.rank == 0
    first_tuple_1 = first_tuple_1.future().wait()
    assert first_tuple_1.status == assets.Status.done
    first_tuple_2 = first_tuple_2.future().wait()
    assert first_tuple_2.status == assets.Status.done


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_tangled_traintuples_single_node(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A data sample 1: traintuple \-/-> traintuple \-/-> ... \-/-> traintuple
                                      x                x         x
    Node A data sample 2: traintuple /-\-> traintuple /-\-> ... /-\-> traintuple
    """

    factory, network = global_execution_env
    session = network.sessions[0].copy()
    dataset = session.state.datasets[0]

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    cp_spec = factory.create_compute_plan()

    previous_tuples = []
    for _ in range(compute_plan_size):
        previous_tuples = [
            cp_spec.add_traintuple(
                algo=algo,
                dataset=dataset,
                data_samples=[data_sample],
                in_models=previous_tuples,
            )
            for data_sample in dataset.train_data_sample_keys[:2]
        ]

    cp = session.add_compute_plan(cp_spec)
    first_tuple_1, first_tuple_2 = cp.list_traintuple()[:2]
    assert first_tuple_1.rank == 0
    assert first_tuple_2.rank == 0
    first_tuple_1 = first_tuple_1.future().wait()
    assert first_tuple_1.status == assets.Status.done
    first_tuple_2 = first_tuple_2.future().wait()
    assert first_tuple_2.status == assets.Status.done

    cp = session.cancel_compute_plan(cp.compute_plan_id).future().wait()
    assert cp.status == assets.Status.canceled

    first_tuple_1, first_tuple_2 = cp.list_traintuple()[:2]
    assert first_tuple_1.rank == 0
    assert first_tuple_2.rank == 0
    first_tuple_1 = first_tuple_1.future().wait()
    assert first_tuple_1.status == assets.Status.done
    first_tuple_2 = first_tuple_2.future().wait()
    assert first_tuple_2.status == assets.Status.done


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_tangled_composite_traintuples(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A: composite traintuple \-/-> composite traintuple \-/-> ... \-/-> composite traintuple
                                  x                          x         x
    Node B: composite traintuple /-\-> composite traintuple /-\-> ... /-\-> composite traintuple
    """

    factory, network = global_execution_env
    session_1 = network.sessions[0].copy()
    session_2 = network.sessions[1].copy()

    dataset_1 = session_1.state.datasets[0]
    dataset_2 = session_2.state.datasets[0]

    spec = factory.create_composite_algo()
    algo = session_1.add_composite_algo(spec)

    cp_spec = factory.create_compute_plan()

    previous_tuples = None
    for _ in range(compute_plan_size):
        previous_tuples = [
            cp_spec.add_composite_traintuple(
                composite_algo=algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0]],
                # use head model from previous composite traintuple on same node
                in_head_model=previous_tuples[i] if previous_tuples else None,
                # use trunk model from previous composite traintuple on other node
                in_trunk_model=previous_tuples[(i+1) % 2] if previous_tuples else None,
            )
            for i, dataset in enumerate([dataset_1, dataset_2])
        ]

    cp = session_1.add_compute_plan(cp_spec)
    first_tuple_1, first_tuple_2 = cp.list_composite_traintuple()[:2]
    assert first_tuple_1.rank == 0
    assert first_tuple_2.rank == 0
    first_tuple_1 = first_tuple_1.future().wait()
    assert first_tuple_1.status == assets.Status.done
    first_tuple_2 = first_tuple_2.future().wait()
    assert first_tuple_2.status == assets.Status.done

    cp = session_1.cancel_compute_plan(cp.compute_plan_id).future().wait()
    assert cp.status == assets.Status.canceled

    first_tuple_1, first_tuple_2 = cp.list_composite_traintuple()[:2]
    assert first_tuple_1.rank == 0
    assert first_tuple_2.rank == 0
    first_tuple_1 = first_tuple_1.future().wait()
    assert first_tuple_1.status == assets.Status.done
    first_tuple_2 = first_tuple_2.future().wait()
    assert first_tuple_2.status == assets.Status.done


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_aggregatetuples(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A: composite traintuple --> aggregate --> composite traintuple --> aggregate ...
    Node B: composite traintuple /             \-> composite traintuple /
    """

    factory, network = global_execution_env
    session_1 = network.sessions[0].copy()
    session_2 = network.sessions[1].copy()

    dataset_1 = session_1.state.datasets[0]
    dataset_2 = session_2.state.datasets[0]

    spec = factory.create_composite_algo()
    composite_algo = session_1.add_composite_algo(spec)

    spec = factory.create_aggregate_algo()
    aggregate_algo = session_1.add_aggregate_algo(spec)

    cp_spec = factory.create_compute_plan()

    previous_aggregatetuple = None
    previous_composite_traintuples = None
    for _ in range(compute_plan_size):
        previous_composite_traintuples = [
            cp_spec.add_composite_traintuple(
                composite_algo=composite_algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0]],
                in_head_model=previous_composite_traintuples[i] if previous_composite_traintuples else [],
                in_trunk_model=previous_aggregatetuple if previous_aggregatetuple else None,
            ) for i, dataset in enumerate([dataset_1, dataset_2])
        ]
        previous_aggregatetuple = cp_spec.add_aggregatetuple(
            aggregate_algo=aggregate_algo,
            worker=session_1.node_id,
            in_models=previous_composite_traintuples,
        )

    cp = session_1.add_compute_plan(cp_spec)
    first_aggregatetuple = cp.list_aggregatetuple()[0].future().wait()
    assert first_aggregatetuple.rank == 1
    assert first_aggregatetuple.status == assets.Status.done

    cp = session_1.cancel_compute_plan(cp.compute_plan_id).future().wait()
    assert cp.status == assets.Status.canceled

    first_aggregatetuple = cp.list_aggregatetuple()[0]
    assert first_aggregatetuple.rank == 1
    assert first_aggregatetuple.status == assets.Status.done


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_aggregatetuples_manual(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A: composite traintuple --> aggregate --> composite traintuple --> aggregate ...
    Node B: composite traintuple /             \-> composite traintuple /
    """

    factory, network = global_execution_env
    session_1 = network.sessions[0].copy()
    session_2 = network.sessions[1].copy()

    dataset_1 = session_1.state.datasets[0]
    dataset_2 = session_2.state.datasets[0]

    spec = factory.create_composite_algo()
    composite_algo = session_1.add_composite_algo(spec)

    spec = factory.create_aggregate_algo()
    aggregate_algo = session_1.add_aggregate_algo(spec)

    previous_aggregatetuple = None
    previous_composite_traintuples = None
    compute_plan_id = None
    first_aggregatetuple = None
    for i in range(compute_plan_size):
        rank = i*2
        # composite traintuple 1
        spec = factory.create_composite_traintuple(
            algo=composite_algo,
            dataset=dataset_1,
            data_samples=[dataset_1.train_data_sample_keys[0]],
            head_traintuple=previous_composite_traintuples[0] if previous_composite_traintuples else [],
            trunk_traintuple=previous_aggregatetuple if previous_aggregatetuple else None,
            rank=rank,
            compute_plan_id=compute_plan_id,
        )
        composite_traintuple_1 = session_1.add_composite_traintuple(spec)
        compute_plan_id = composite_traintuple_1.compute_plan_id

        # composite traintuple 2
        spec = factory.create_composite_traintuple(
            algo=composite_algo,
            dataset=dataset_2,
            data_samples=[dataset_2.train_data_sample_keys[0]],
            head_traintuple=previous_composite_traintuples[1] if previous_composite_traintuples else [],
            trunk_traintuple=previous_aggregatetuple if previous_aggregatetuple else None,
            rank=rank,
            compute_plan_id=compute_plan_id,
        )
        composite_traintuple_2 = session_2.add_composite_traintuple(spec)

        previous_composite_traintuples = [
            composite_traintuple_1,
            composite_traintuple_2,
        ]

        spec = factory.create_aggregatetuple(
            algo=aggregate_algo,
            worker=session_1.node_id,
            traintuples=previous_composite_traintuples,
            rank=rank + 1,
            compute_plan_id=compute_plan_id,
        )
        previous_aggregatetuple = session_1.add_aggregatetuple(spec)

        if not first_aggregatetuple:
            first_aggregatetuple = previous_aggregatetuple

    first_aggregatetuple = first_aggregatetuple.future().wait()
    assert first_aggregatetuple.rank == 1
    assert first_aggregatetuple.status == assets.Status.done

    cp = session_1.cancel_compute_plan(compute_plan_id).future().wait()
    assert cp.status == assets.Status.canceled

    first_aggregatetuple = first_aggregatetuple.future().wait()
    assert first_aggregatetuple.rank == 1
    assert first_aggregatetuple.status == assets.Status.done
