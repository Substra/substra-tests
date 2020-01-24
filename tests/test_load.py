import pytest

from substratest import assets

COMPUTE_PLAN_SIZES = [10, 100, 1000, 10000]


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
