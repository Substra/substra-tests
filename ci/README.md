# CI
This folder contains a script that can run the full connect CI pipeline, a cloud function and some additionnal materials required for the automation of the test pipeline.

## The automation script

This automated script run the full connect-tests suite on a real cluster on the Google Cloud Platform.

### Usage

This script is not made uniquely for the CI, you can run it yourself from your local machine !

First you will need to install the requirements:
```sh
pip install -r requirements.txt
```

Then you can run the script with:
```sh
python run-ci.py
```

While it was made to test the main branches of each repository against each other, you can also target specific branches of each repositories or tags. In order to see all the options, run :
```sh
python run-ci.py --help
```

### What will it do ?

This script performs the following operations:

- Create a new GKE cluster
- Clone all the connect repositories:
    - [hlf-k8s](https://github.com/owkin/connect-hlf-k8s)
    - [connect-backend](https://github.com/owkin/connect-backend)
    - [connect-chaincode](https://github.com/owkin/connect-chaincode)
    - [connect-tests](https://github.com/owkin/connect-tests)
- Build the docker images from these repositories using Google Cloud Builds
- Deploy all the services using the Skaffold files in the repositories
- Wait for the connect application stack to be ready (i.e. wait for the connect-backend servers readiness probe)
- Run the full _connect-tests_ test suite
- Destroy the cluster

### Automatic cluster deletion
   We ensure the GKE cluster always gets destroyed at the end of the execution of this script.

   We use 2 mechanisms:

   - When the script exits, we destroy the cluster as the final step. That's the case even if the script exits with an error or with an interruption (Ctrl+C). See `trap` command.
   - In any other "abrupt shutdown" case (system shutdown, GitHub Action build canceled, etc), we use a daily scheduled task to destroy stale clusters after 24h. (more details [here](#the-deletion-function))

## The deletion function

This Cloud function runs every day and cleanup any remaining artifacts of test runs.

### How does it work ?

It's based on a Cloud Scheduler cron job that publish a message every day to a Pub/Sub queue. The queue then triggers the Cloud Function.

The code of the Cloud Function is automatically published from this repo by a Github Action.

If you want to give it a look the function is located here: [clean-tests-ci-deployment function](https://console.cloud.google.com/functions/details/us-central1/clean-tests-ci-deployment)

## Useful links
- Github Action history: https://github.com/owkin/connect-tests/actions/workflows/end-to-end.yml

