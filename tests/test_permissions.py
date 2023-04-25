from contextlib import nullcontext as does_not_raise

import pytest
import substra

from substratest.factory import AugmentedDataset
from substratest.factory import FunctionCategory
from substratest.factory import Permissions
from substratest.fl_interface import FLTaskInputGenerator
from substratest.fl_interface import FLTaskOutputGenerator
from substratest.fl_interface import OutputIdentifiers


@pytest.fixture
def public():
    return Permissions(public=True, authorized_ids=[])


@pytest.fixture
def private():
    return Permissions(public=False, authorized_ids=[])


@pytest.fixture
def all_organizations(clients):
    return Permissions(public=False, authorized_ids=[c.organization_id for c in clients])


@pytest.fixture
def organization_1_only(client_1):
    return Permissions(public=False, authorized_ids=[client_1.organization_id])


@pytest.fixture
def organization_2_only(client_2):
    return Permissions(public=False, authorized_ids=[client_2.organization_id])


@pytest.fixture
def organizations_1_and_2_only(client_1, client_2):
    return Permissions(public=False, authorized_ids=[client_1.organization_id, client_2.organization_id])


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.parametrize("is_public", [True, False])
def test_permission_creation(is_public, factory, client):
    """Test asset creation with simple permission."""
    permissions = Permissions(public=is_public, authorized_ids=[])
    spec = factory.create_dataset(permissions=permissions)
    dataset = client.add_dataset(spec)
    assert dataset.permissions.process.public is is_public


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.parametrize(
    "permissions",
    [
        pytest.lazy_fixture("public"),
        pytest.lazy_fixture("private"),
        pytest.lazy_fixture("all_organizations"),
        pytest.lazy_fixture("organization_1_only"),
        pytest.lazy_fixture("organization_2_only"),
    ],
)
def test_get_metadata(permissions, factory, clients, channel):
    """Test get metadata assets with various permissions."""
    clients = clients[:2]

    # add 1 dataset per organization
    datasets = []
    for client in clients:
        spec = factory.create_dataset(permissions=permissions)
        d = client.add_dataset(spec)
        datasets.append(d)

    for d in datasets:
        channel.wait_for_asset_synchronized(d)

    # check that all clients can get access to all metadata
    for client in clients:
        for d in datasets:
            d = client.get_dataset(d.key)
            assert d.permissions.process.public == permissions.public


@pytest.mark.remote_only  # no check on permissions with the local backend
def test_permission_invalid_organization_id(factory, client):
    """Test asset creation with invalid permission."""
    invalid_organization = "unknown-organization"
    invalid_permissions = Permissions(public=False, authorized_ids=[invalid_organization])
    spec = factory.create_dataset(permissions=invalid_permissions)
    with pytest.raises(substra.exceptions.InvalidRequest) as exc:
        client.add_dataset(spec)
    assert "invalid permission input values" in str(exc.value)


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.parametrize(
    "permissions",
    [
        pytest.lazy_fixture("public"),
        pytest.lazy_fixture("organization_2_only"),
    ],
)
def test_download_asset_access_granted(permissions, factory, client_1, client_2, channel):
    """Test asset can be downloaded by all permitted organizations."""
    spec = factory.create_dataset(permissions=permissions)
    dataset = client_1.add_dataset(spec)

    content = client_1.download_opener(dataset.key)
    assert content == spec.read_opener()

    channel.wait_for_asset_synchronized(dataset)
    content = client_2.download_opener(dataset.key)
    assert content == spec.read_opener()


@pytest.mark.remote_only  # no check on permissions with the local backend
def test_download_asset_access_restricted(factory, client_1, client_2, channel):
    """Test public asset can be downloaded by all organizations."""
    permissions = Permissions(public=False, authorized_ids=[])
    spec = factory.create_dataset(permissions=permissions)
    dataset = client_1.add_dataset(spec)

    content = client_1.download_opener(dataset.key)
    assert content == spec.read_opener()
    channel.wait_for_asset_synchronized(dataset)

    with pytest.raises(substra.exceptions.AuthorizationError):
        client_2.download_opener(dataset.key)


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.parametrize(
    "permissions_1,permissions_2,expected_permissions",
    [
        (
            pytest.lazy_fixture("organization_2_only"),
            pytest.lazy_fixture("organization_1_only"),
            pytest.lazy_fixture("organizations_1_and_2_only"),
        ),
        (
            pytest.lazy_fixture("public"),
            pytest.lazy_fixture("organization_1_only"),
            pytest.lazy_fixture("organizations_1_and_2_only"),
        ),
    ],
)
def test_permissions(permissions_1, permissions_2, expected_permissions, factory, client_1, client_2, channel, workers):
    # add train data samples / dataset
    spec = factory.create_dataset(permissions=permissions_1)
    dataset_1 = client_1.add_dataset(spec)
    spec = factory.create_data_sample(datasets=[dataset_1])
    data_sample_key_1 = client_1.add_data_sample(spec)

    dataset_1 = AugmentedDataset(client_1.get_dataset(dataset_1.key))
    dataset_1.set_train_test_dasamples(
        train_data_sample_keys=[data_sample_key_1],
    )
    # add function
    spec = factory.create_function(category=FunctionCategory.simple, permissions=permissions_2)
    function_2 = client_2.add_function(spec)

    # add traintask
    spec = factory.create_traintask(
        function=function_2,
        inputs=dataset_1.train_data_inputs,
        outputs=FLTaskOutputGenerator.traintask(authorized_ids=expected_permissions.authorized_ids),
        worker=workers[0],
    )
    channel.wait_for_asset_synchronized(function_2)
    traintask = client_1.add_task(spec)
    traintask = client_1.wait(traintask)

    # check the compute task executed on the correct worker
    assert traintask.outputs[OutputIdentifiers.model].value is not None
    assert traintask.worker == client_1.organization_id

    # check the permissions
    task_permissions = traintask.outputs[OutputIdentifiers.model].permissions
    assert task_permissions.process.public == expected_permissions.public
    assert set(task_permissions.process.authorized_ids) == set(expected_permissions.authorized_ids)


@pytest.mark.remote_only  # no check on permissions with the local backend
def test_permissions_denied_process(factory, client_1, client_2, channel, workers):
    # setup data

    spec = factory.create_dataset(permissions=Permissions(public=False, authorized_ids=[]))
    dataset_1 = client_1.add_dataset(spec)

    spec = factory.create_data_sample(
        datasets=[dataset_1],
    )
    data_sample_key_1 = client_1.add_data_sample(spec)
    dataset_1 = AugmentedDataset(client_1.get_dataset(dataset_1.key))
    dataset_1.set_train_test_dasamples(
        train_data_sample_keys=[data_sample_key_1],
    )

    # setup function

    spec = factory.create_function(
        category=FunctionCategory.simple, permissions=Permissions(public=False, authorized_ids=[])
    )
    function_2 = client_2.add_function(spec)
    channel.wait_for_asset_synchronized(function_2)

    # traintasks

    spec = factory.create_traintask(function=function_2, inputs=dataset_1.train_data_inputs, worker=workers[0])

    with pytest.raises(substra.exceptions.AuthorizationError):
        client_2.add_task(spec)

    with pytest.raises(substra.exceptions.AuthorizationError):
        client_1.add_task(spec)


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.slow
@pytest.mark.parametrize(
    "client_1_permissions,client_2_permissions,expectation",
    [
        (
            pytest.lazy_fixture("private"),
            pytest.lazy_fixture("private"),
            pytest.raises(substra.exceptions.AuthorizationError),
        ),
        (pytest.lazy_fixture("organization_2_only"), pytest.lazy_fixture("private"), does_not_raise()),
    ],
)
def test_permissions_model_process(
    client_1_permissions, client_2_permissions, expectation, factory, client_1, client_2, channel, workers
):
    """Test that a traintask can/cannot process an in-model depending on permissions."""
    datasets = []
    functions = []
    for client, permissions in zip([client_1, client_2], [client_1_permissions, client_2_permissions]):
        # dataset
        spec = factory.create_dataset(permissions=permissions)
        dataset = client.add_dataset(spec)
        spec = factory.create_data_sample(
            datasets=[dataset],
        )
        data_sample_key = client.add_data_sample(spec)
        augmented_dataset = AugmentedDataset(client.get_dataset(dataset.key))
        augmented_dataset.set_train_test_dasamples(train_data_sample_keys=[data_sample_key])

        datasets.append(augmented_dataset)

        # function
        spec = factory.create_function(category=FunctionCategory.simple, permissions=permissions)
        function = client.add_function(spec)
        channel.wait_for_asset_synchronized(function)
        functions.append(function)

    dataset_1, dataset_2 = datasets
    function_1, function_2 = functions

    # traintasks
    spec = factory.create_traintask(
        function=function_1,
        inputs=dataset_1.train_data_inputs,
        outputs=FLTaskOutputGenerator.traintask(authorized_ids=client_1_permissions.authorized_ids),
        worker=workers[0],
    )
    traintask_1 = client_1.add_task(spec)
    traintask_1 = client_1.wait(traintask_1)

    assert not traintask_1.outputs[OutputIdentifiers.model].permissions.process.public
    assert set(traintask_1.outputs[OutputIdentifiers.model].permissions.process.authorized_ids) == set(
        [client_1.organization_id] + client_1_permissions.authorized_ids
    )

    spec = factory.create_traintask(
        function=function_2,
        inputs=dataset_2.train_data_inputs + FLTaskInputGenerator.trains_to_train([traintask_1.key]),
        worker=workers[1],
    )

    with expectation:
        traintask_2 = client_2.add_task(spec)

        client_2.wait(traintask_2)


@pytest.mark.remote_only  # no check on permissions with the local backend
def test_merge_permissions_denied_process(factory, clients, channel, workers):
    """Test to process asset with merged permissions from 2 other organizations

    - dataset and metrics located on organization 1
    - function located on organization 2
    - traintask created on organization 2
    - failed attempt to create testtask using this traintask from organization 3
    """
    if len(clients) < 3:
        pytest.skip("requires at least 3 organizations")

    # define clients once and for all
    client_1, client_2, client_3, *_ = clients

    permissions_list = [
        (
            Permissions(public=False, authorized_ids=[client_2.organization_id, client_3.organization_id]),
            Permissions(public=False, authorized_ids=[client_1.organization_id]),
        ),
        (
            Permissions(public=True, authorized_ids=[]),
            Permissions(public=False, authorized_ids=[client_1.organization_id]),
        ),
    ]
    for permissions_1, permissions_2 in permissions_list:
        # add train data samples / dataset / metric on organization 1
        spec = factory.create_dataset(permissions=permissions_1)
        dataset_1 = client_1.add_dataset(spec)
        channel.wait_for_asset_synchronized(dataset_1)  # used by client_2 and client_3
        spec = factory.create_data_sample(
            datasets=[dataset_1],
        )
        client_1.add_data_sample(spec)
        spec = factory.create_data_sample(
            datasets=[dataset_1],
        )

        data_sample_key_1 = client_1.add_data_sample(spec)
        spec = factory.create_function(category=FunctionCategory.metric, permissions=permissions_1)
        metric_1 = client_1.add_function(spec)
        channel.wait_for_asset_synchronized(metric_1)  # used by client_3

        # add function on organization 2
        spec = factory.create_function(category=FunctionCategory.simple, permissions=permissions_2)
        function_2 = client_2.add_function(spec)

        # add traintask from organization 2
        spec = factory.create_function(category=FunctionCategory.predict, permissions=permissions_1)
        predict_function_1 = client_1.add_function(spec)

        dataset_1 = AugmentedDataset(client_1.get_dataset(dataset_1.key))
        dataset_1.set_train_test_dasamples(train_data_sample_keys=[data_sample_key_1])

        # add traintask from node 2
        spec = factory.create_traintask(function=function_2, inputs=dataset_1.train_data_inputs, worker=workers[0])
        traintask_2 = client_2.add_task(spec)
        traintask_2 = client_2.wait(traintask_2)
        channel.wait_for_asset_synchronized(traintask_2)  # used by client_3

        # failed to add predicttask from organization 3
        spec = factory.create_predicttask(
            function=predict_function_1,
            inputs=dataset_1.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_2.key),
            worker=workers[0],
        )

        with pytest.raises(substra.exceptions.AuthorizationError):
            client_3.add_task(spec)


@pytest.mark.remote_only  # no check on permissions with the local backend
def test_permissions_denied_head_model_process(factory, client_1, client_2, channel, workers):
    # setup data
    datasets = []
    for client in [client_1, client_2]:
        spec = factory.create_dataset(permissions=Permissions(public=False, authorized_ids=[client.organization_id]))
        dataset = client.add_dataset(spec)

        spec = factory.create_data_sample(
            datasets=[dataset],
        )
        data_sample_key = client.add_data_sample(spec)

        dataset = client.get_dataset(dataset.key)
        channel.wait_for_asset_synchronized(dataset)

        augmented_dataset = AugmentedDataset(client.get_dataset(dataset.key))
        augmented_dataset.set_train_test_dasamples(train_data_sample_keys=[data_sample_key])

        datasets.append(augmented_dataset)

    dataset_1, dataset_2 = datasets

    # create function
    spec = factory.create_function(category=FunctionCategory.composite)
    composite_function = client_1.add_function(spec)
    channel.wait_for_asset_synchronized(composite_function)  # used by client_2

    # create composite task
    spec = factory.create_composite_traintask(
        function=composite_function,
        inputs=dataset_1.train_data_inputs,
        outputs=FLTaskOutputGenerator.composite_traintask(
            shared_authorized_ids=[client_1.organization_id, client_2.organization_id],
            local_authorized_ids=[client_1.organization_id],
        ),
        worker=workers[0],
    )

    composite_traintask_1 = client_1.add_task(spec)
    composite_traintask_1 = client_1.wait(composite_traintask_1)
    channel.wait_for_asset_synchronized(composite_traintask_1)  # used by client_2

    spec = factory.create_composite_traintask(
        function=composite_function,
        inputs=dataset_2.train_data_inputs + FLTaskInputGenerator.composite_to_composite(composite_traintask_1.key),
        worker=workers[1],
    )
    with pytest.raises(substra.exceptions.AuthorizationError):
        client_2.add_task(spec)


@pytest.mark.remote_only  # no check on permissions with the local backend
@pytest.mark.parametrize(
    "permission_train_output, expectation",
    [
        (pytest.lazy_fixture("organization_1_only"), pytest.raises(substra.exceptions.AuthorizationError)),
        (pytest.lazy_fixture("organization_2_only"), does_not_raise()),
    ],
)
def test_permission_to_test_on_org_without_training(
    permission_train_output,
    organization_1_only,
    organization_2_only,
    client_1,
    client_2,
    factory,
    expectation,
    channel,
):
    # training function on client 1
    spec = factory.create_function(category=FunctionCategory.simple, permissions=organization_1_only)
    train_function = client_1.add_function(spec)

    # predict and metric function on client 2
    spec = factory.create_function(category=FunctionCategory.predict, permissions=organization_2_only)
    predict_function = client_2.add_function(spec)

    # predict and metric function on client 2
    spec = factory.create_function(category=FunctionCategory.metric, permissions=organization_2_only)
    metric_function = client_2.add_function(spec)

    # add train data samples on organization 1
    spec = factory.create_dataset(permissions=organization_1_only)
    dataset_1 = client_1.add_dataset(spec)

    spec = factory.create_data_sample(
        datasets=[dataset_1],
    )
    train_datasample = client_1.add_data_sample(spec)

    dataset_1 = AugmentedDataset(client_1.get_dataset(dataset_1.key))
    dataset_1.set_train_test_dasamples(train_data_sample_keys=[train_datasample])

    # add test data samples on organization 2
    spec = factory.create_dataset(permissions=organization_2_only)
    dataset_2 = client_2.add_dataset(spec)

    spec = factory.create_data_sample(
        datasets=[dataset_2],
    )
    test_datasample = client_2.add_data_sample(spec)
    dataset_2 = AugmentedDataset(client_2.get_dataset(dataset_2.key))
    dataset_2.set_train_test_dasamples(test_data_sample_keys=[test_datasample])

    # add traintask on org 1
    spec = factory.create_traintask(
        function=train_function,
        inputs=dataset_1.train_data_inputs,
        outputs=FLTaskOutputGenerator.traintask(authorized_ids=permission_train_output.authorized_ids),
        worker=client_1.organization_id,
    )
    traintask_1 = client_1.add_task(spec)

    # add testtask on org 2
    with expectation:
        spec = factory.create_predicttask(
            function=predict_function,
            inputs=dataset_2.test_data_inputs + FLTaskInputGenerator.train_to_predict(traintask_1.key),
            worker=client_2.organization_id,
        )
        predicttask_2 = client_2.add_task(spec)

        spec = factory.create_testtask(
            function=metric_function,
            inputs=dataset_2.test_data_inputs + FLTaskInputGenerator.predict_to_test(predicttask_2.key),
            worker=client_2.organization_id,
        )
        _ = client_2.add_task(spec)
