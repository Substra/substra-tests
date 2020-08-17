import pytest

from substratest import assets
from substratest.factory import Permissions


# @pytest.mark.skip("Need to fix this test")
@pytest.mark.remote_only
@pytest.mark.slow
def test_execution_debug(client, debug_client, factory, default_dataset, default_objective):

    spec = factory.create_algo()
    algo = client.add_algo(spec)

    #  Add the traintuple
    # create traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys[0],
    )
    traintuple = debug_client.add_traintuple(spec)
    assert traintuple.status == assets.Status.done
    assert traintuple.out_model is not None

    # Add the testtuple
    spec = factory.create_testtuple(
        objective=default_objective,
        traintuple=traintuple,
        data_samples=default_dataset.test_data_sample_keys[0],
    )
    testtuple = debug_client.add_testtuple(spec).future().wait()
    assert testtuple.status == assets.Status.done
    assert testtuple.dataset.perf == 3


# @pytest.mark.skip("Need to fix this test")
@pytest.mark.remote_only
@pytest.mark.slow
def test_debug_compute_plan_aggregate_composite(client, debug_client, factory, default_datasets, default_objectives):
    """
    Debug / Compute plan version of the
    `test_aggregate_composite_traintuples` method from `test_execution.py`
    """
    aggregate_worker = debug_client.node_id
    number_of_rounds = 2

    # register algos on first node
    spec = factory.create_composite_algo()
    composite_algo = client.add_composite_algo(spec)
    spec = factory.create_aggregate_algo()
    aggregate_algo = client.add_aggregate_algo(spec)

    # launch execution
    previous_aggregatetuple_spec = None
    previous_composite_traintuple_specs = []

    cp_spec = factory.create_compute_plan()

    for round_ in range(number_of_rounds):
        # create composite traintuple on each node
        composite_traintuple_specs = []
        for index, dataset in enumerate(default_datasets):
            kwargs = {}
            if previous_aggregatetuple_spec:
                kwargs = {
                    'in_head_model': previous_composite_traintuple_specs[index],
                    'in_trunk_model': previous_aggregatetuple_spec,
                }
            spec = cp_spec.add_composite_traintuple(
                composite_algo=composite_algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0 + round_]],
                out_trunk_model_permissions=Permissions(public=False, authorized_ids=[debug_client.node_id]),
                **kwargs,
            )
            composite_traintuple_specs.append(spec)

        # create aggregate on its node
        spec = cp_spec.add_aggregatetuple(
            aggregate_algo=aggregate_algo,
            worker=aggregate_worker,
            in_models=composite_traintuple_specs,
        )

        # save state of round
        previous_aggregatetuple_spec = spec
        previous_composite_traintuple_specs = composite_traintuple_specs

    # last round: create associated testtuple
    for composite_traintuple_spec, objective in zip(previous_composite_traintuple_specs, default_objectives):
        cp_spec.add_testtuple(
            objective=objective,
            traintuple_spec=composite_traintuple_spec,
        )

    cp = debug_client.add_compute_plan(cp_spec).future().wait()
    traintuples = cp.list_traintuple()
    composite_traintuples = cp.list_composite_traintuple()
    aggregatetuples = cp.list_aggregatetuple()
    testtuples = cp.list_testtuple()

    tuples = traintuples + composite_traintuples + aggregatetuples + testtuples
    for t in tuples:
        assert t.status == assets.Status.done


@pytest.mark.remote_only
def test_debug_download_dataset(debug_client, default_dataset):
    debug_client.download_opener(default_dataset.key)
