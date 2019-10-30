# substra-tests

Substra end-to-end (e2e) tests

# Prerequisites

Install tests dependencies:

```
pip3 install -r requirements.txt
```

The tests suite requires a Substra network up and running. The network can be started
either with skaffold (Kubernetes), manually or through docker-compose.

# How to run tests?

The tests suite needs the network configuration to be executed. There are currently two different modes to define it:
- in a network configuration yaml file (default mode)
- from a skaffold yaml file

## Using a network configuration file

This is the mode used by default and the default file is located at *values.yaml*.
Two files are currently available:
- *values.yaml*: configuration when network is started in Kubernetes environment (default)
- *values-docker-compose.yaml*: configuration when network is started manually or with docker-compose

To execute the tests using the default *values.yaml* file:

```
make test
```

To use the file *values-docker-compose.yaml*, the configuration filepath can be overriden using the `SUBSTRAT_CONFIG_FILEPATH` environment variable. For instance:

```
SUBSTRAT_CONFIG_FILEPATH=values-docker-compose.yaml make test
```

## Using a skaffold file

To get the network configuration from a backend skaffold file, the environment variable `SUBSTRAT_SKAFFOLD_FILEPATH` must be set. For instance:

```
SUBSTRAT_SKAFFOLD_FILEPATH=$SUBSTRA_SOURCE/substra-backend/skaffold.yaml make test
```
