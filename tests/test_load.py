import pytest

from substratest import assets

# COMPUTE_PLAN_SIZES = [10, 100, 1000, 10000]
COMPUTE_PLAN_SIZES = [10]


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_single_node_1_branch(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A: traintuple 0 -> traintuple -> ... -> traintuple N
    """
    factory, network = global_execution_env
    session = network.sessions[0].copy()

    dataset = session.state.datasets[0]

    spec = factory.create_algo()
    algo = session.add_algo(spec)

    cp_spec = factory.create_compute_plan()
    previous_tuple = None
    for _ in range(compute_plan_size):
        previous_tuple = cp_spec.add_traintuple(
            algo=algo,
            dataset=dataset,
            data_samples=[dataset.train_data_sample_keys[0]],
            in_models=[previous_tuple] if previous_tuple else [],
        )

    cp = session.add_compute_plan(cp_spec)
    first_tuple = cp.list_traintuple()[0]
    assert first_tuple.rank == 0
    first_tuple = first_tuple.future().wait()
    assert first_tuple.status == assets.Status.done

    cp = session.cancel_compute_plan(cp.compute_plan_id).future().wait()
    assert cp.status == assets.Status.canceled

    first_tuple = cp.list_traintuple()[0]
    assert first_tuple.rank == 0
    assert first_tuple.status == assets.Status.done


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_multi_node_2_branches(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A: traintuple 0 -> traintuple -> ... -> traintuple
    Node B:              -> traintuple -> ... -> traintuple
    """

    factory, network = global_execution_env
    session_1 = network.sessions[0].copy()
    session_2 = network.sessions[1].copy()

    dataset_1 = session_1.state.datasets[0]
    dataset_2 = session_2.state.datasets[0]

    spec = factory.create_algo()
    algo = session_1.add_algo(spec)

    cp_spec = factory.create_compute_plan()
    first_tuple = cp_spec.add_traintuple(
        algo=algo,
        dataset=dataset_1,
        data_samples=[dataset_1.train_data_sample_keys[0]],
    )

    for dataset in [dataset_1, dataset_2]:
        previous_tuple = first_tuple
        for _ in range(compute_plan_size):
            previous_tuple = cp_spec.add_traintuple(
                algo=algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0]],
                in_models=[previous_tuple],
            )

    cp = session_1.add_compute_plan(cp_spec)
    first_tuple = cp.list_traintuple()[0]
    assert first_tuple.rank == 0
    first_tuple = first_tuple.future().wait()
    assert first_tuple.status == assets.Status.done

    cp = session_1.cancel_compute_plan(cp.compute_plan_id).future().wait()
    assert cp.status == assets.Status.canceled

    first_tuple = cp.list_traintuple()[0]
    assert first_tuple.rank == 0
    assert first_tuple.status == assets.Status.done


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_multi_node_composite_aggregates(compute_plan_size, global_execution_env):
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
