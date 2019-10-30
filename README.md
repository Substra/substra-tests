# substra-tests

Substra end-to-end (e2e) tests

# Prerequisites

Install tests dependencies:

```
pip3 install -r requirements.txt
```

The tests suite requires a Substra network up and running. The network can be started
either with skaffold (Kubernetes), with docker-compose, or manually.

# Run the tests

## Using a network configuration file

This is the mode used by default. Two files are currently available:
- `values.yaml` (default): for networks started with Kubernetes
- `values-docker-compose.yaml`: for networks started with docker-compose, or started manually

To run the tests using the default `values.yaml` file:

```
make test
```

To run the tests using the provided `values-docker-compose.yaml` (or a custom config file):

```
SUBSTRAT_CONFIG_FILEPATH=values-docker-compose.yaml make test
```

## Using a skaffold file

It is possible to use a `substra-backend` skaffold file as the source for the network configuration:

```
SUBSTRAT_SKAFFOLD_FILEPATH=$SUBSTRA_SOURCE/substra-backend/skaffold.yaml make test
```
