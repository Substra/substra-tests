# substra-tests

Substra end-to-end (e2e) tests

# Prerequisites

Install tests dependencies:

```
pip3 install -r requirements.txt
```

The tests suite requires a Substra network up and running. The network can be started
either with skaffold (Kubernetes), with docker-compose, or manually.

The substra project is needed for running the tests.
It can be found [here](https://github.com/SubstraFoundation/substra)

You will need to install it thanks to the `pip` binary.

# Run the tests

The network configuration is described in a yaml file.

Two configuration files are currently available:
- `values.yaml` (default): for networks started with Kubernetes
- `local-values.yaml`: for networks started with docker-compose, or started manually

To run the tests using the default `values.yaml` file:

```
make test
```

To run the tests using the provided `local-values.yaml` (or a custom config file):

```
SUBSTRA_TESTS_CONFIG_FILEPATH=local-values.yaml make test
```

# Fast mode

Since tests can take a long time to run, some of them as marked as slow. You can run the "fast" ones with:

```
make fast-test
```

Note that `test_compute_plan` from `test_execution_compute_plan.py` is not marked as slow even though it takes several
seconds to complete. This is because it covers a very basic use case of the platform and is needed to ensure basic
features aren't broken.

# Test design guidelines

When adding or modifying tests, please follow these guidelines:

1. The complete test suite must be independent from the substra network
   - The substra network must be started prior to (and independently from) executing the tests
   - The substra network can be running locally / on the cloud
   - The substra network can be started through docker-compose or through kubernetes
1. It should be possible to run the test suite multiple times without restarting the substra network
1. Each test must have a deterministic behavior (must not fail randomly)
1. Each test must be fast to run:
   - Avoid tests that take a lot of time to complete
   - Group related long running tests (when it makes sense)
1. Each test should not complete before all the tuples it created have been executed. This requirement ensures that the next test will be launched with a substra network ready to execute new tuples
1. Tests must not use hardcoded network configuration settings/values. Use settings files instead (e.g. `values.yaml`)
1. Tests should target a substra network with at least 2 organisations
