import uuid

import pytest


class TestFilterMetadata:
    @pytest.fixture
    def compute_plan_keys(self, factory, client_1):
        cp_key = str(uuid.uuid4())

        # create compute plan
        cp_spec = factory.create_compute_plan(
            key=cp_key,
            tag="foo",
            name="Bar",
            metadata={
                "test_filter_metadata": cp_key,
                "test_2": "cp_1",
                "key_only_1": "1",
                "same_value_both": "same",
            },
        )

        cp_key_2 = str(uuid.uuid4())
        cp_spec_2 = factory.create_compute_plan(
            key=cp_key_2,
            tag="foo",
            name="Bar",
            metadata={
                "test_filter_metadata": cp_key_2,
                "test_2": "cp_2",
                "same_value_both": "same",
            },
        )
        client_1.add_compute_plan(cp_spec)
        client_1.add_compute_plan(cp_spec_2)

        return cp_key, cp_key_2

    def test_filters_metadata_is_one_value(self, client_1, compute_plan_keys):
        cp_key, cp_key_2 = compute_plan_keys

        filtered_cps = client_1.list_compute_plan(
            filters={
                "key": [cp_key, cp_key_2],
                "metadata": [{"key": "test_filter_metadata", "type": "is", "value": cp_key}],
            }
        )
        assert len(filtered_cps) == 1
        assert filtered_cps[0].key == cp_key

    def test_filters_metadata_contains_one_value(self, client_1, compute_plan_keys):
        cp_key, cp_key_2 = compute_plan_keys

        filtered_cps = client_1.list_compute_plan(
            filters={
                "key": [cp_key, cp_key_2],
                "metadata": [{"key": "test_filter_metadata", "type": "contains", "value": cp_key[:8]}],
            }
        )
        assert len(filtered_cps) == 1
        assert filtered_cps[0].key == cp_key

    def test_filters_metadata_exists_one_value(self, client_1, compute_plan_keys):
        cp_key, cp_key_2 = compute_plan_keys

        filtered_cps = client_1.list_compute_plan(
            filters={"key": [cp_key, cp_key_2], "metadata": [{"key": "test_filter_metadata", "type": "exists"}]}
        )
        assert len(filtered_cps) == 2
        assert {cp.key for cp in filtered_cps} == {cp_key, cp_key_2}

    @pytest.mark.parametrize("type_", ["is", "contains"])
    def test_filters_metadata_key_does_not_exist(self, client_1, compute_plan_keys, type_):
        cp_key, cp_key_2 = compute_plan_keys

        filtered_cps = client_1.list_compute_plan(
            filters={
                "key": [cp_key, cp_key_2],
                "metadata": [{"key": "unknown_metadata_key", "type": type_, "value": "blabla"}],
            }
        )
        assert len(filtered_cps) == 0

    def test_filters_metadata_and_condition(self, client_1, compute_plan_keys):
        cp_key, cp_key_2 = compute_plan_keys

        filtered_cps = client_1.list_compute_plan(
            filters={
                "key": [cp_key, cp_key_2],
                "metadata": [
                    {"key": "same_value_both", "type": "is", "value": "same"},
                    {"key": "test_filter_metadata", "type": "is", "value": cp_key},
                ],
            }
        )
        assert len(filtered_cps) == 1
