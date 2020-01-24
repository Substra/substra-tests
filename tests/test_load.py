import pytest

COMPUTE_PLAN_SIZES = [10, 100, 1000, 10000]


@pytest.mark.parametrize('compute_plan_size', COMPUTE_PLAN_SIZES)
def test_load_aggregatetuples_manual(compute_plan_size, global_execution_env):
    """
    Shape of the compute plan:

    Node A  : composite traintuple --> aggregate --> composite traintuple --> aggregate ...
    Node B-N: composite traintuple /             \-> composite traintuple /
    """

    factory, network = global_execution_env
    sessions = [s.copy() for s in network.sessions]
    datasets = [s.state.datasets[0] for s in sessions]

    spec = factory.create_composite_algo()
    composite_algo = sessions[0].add_composite_algo(spec)

    spec = factory.create_aggregate_algo()
    aggregate_algo = sessions[0].add_aggregate_algo(spec)

    spec = factory.create_compute_plan()
    compute_plan = sessions[0].add_compute_plan(spec)
    compute_plan_id = compute_plan.compute_plan_id

    previous_aggregatetuple = None
    previous_composite_traintuples = None
    for i in range(compute_plan_size):
        rank = i*2
        composite_traintuples = []

        for j in range(len(sessions)):
            spec = factory.create_composite_traintuple(
                algo=composite_algo,
                dataset=datasets[j],
                data_samples=[datasets[j].train_data_sample_keys[0]],
                head_traintuple=previous_composite_traintuples[0] if previous_composite_traintuples else [],
                trunk_traintuple=previous_aggregatetuple if previous_aggregatetuple else None,
                rank=rank,
                compute_plan_id=compute_plan_id,
            )
            composite_traintuples.append(sessions[j].add_composite_traintuple(spec))

        previous_composite_traintuples = composite_traintuples

        spec = factory.create_aggregatetuple(
            algo=aggregate_algo,
            worker=sessions[0].node_id,
            traintuples=previous_composite_traintuples,
            rank=rank + 1,
            compute_plan_id=compute_plan_id,
        )
        previous_aggregatetuple = sessions[0].add_aggregatetuple(spec)

    sessions[0].get_compute_plan(compute_plan_id).future().wait()
