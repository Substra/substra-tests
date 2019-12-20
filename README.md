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

## Using a network configuration file

This is the mode used by default. Two files are currently available:
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

## Using a skaffold file

It is possible to use a `substra-backend` skaffold file as the source for the network configuration:
```
SUBSTRA_TESTS_SKAFFOLD_FILEPATH=$SUBSTRA_SOURCE/substra-backend/skaffold.yaml make test
```

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
1. Each test should not complete before all the tuples it created have been executed. This requirement ensures that the next test will be launched with a susbtra network ready to execute new tuples
1. Tests must not use hardcoded network configuration settings/values. Use settings files instead (e.g. `values.yaml`)
1. Tests should target a substra network with at least 2 organisations
