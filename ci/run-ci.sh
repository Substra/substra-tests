#!/bin/bash

#
# ABOUT
#   This script runs the full substra CI pipeline.
#   "Gotta test them all!" ™
#
#   This script performs the following operations:
#     - Create a new GKE cluster
#     - Install a private docker registry on the cluster
#     - Deploy 'substra-tests-stack', which perfoms the following operations
#       from within the cluster:
#       - Clone the repos:
#         - hlf-k8s
#         - substa-backend
#         - substra-tests
#         - substra
#       - Build the images to the private registry (kaniko)
#       - Deploy (helm)
#     - Wait for the substra application stack to be up and ready (i.e. wait
#       for the substra-backend servers readiness probe)
#     - Run the full 'substra-tests' test suite
#     - Destroy the cluster
#
# AUTOMATIC CLUSTER DESTRUCTION
#   We ensure the GKE cluster always gets destroyed at the end of the execution
#   of this script.
#
#   We use 3 mechanisms:
#   - If no error occurs, we destroy the cluster as the final step
#   - If an error occurs, or if the script gets interrupted (e.g. Ctrl+C), we
#     destroy the cluster too (see `trap` command)
#   - In any other "abrupt shutdown" case (system shutdown, Travis build
#     canceled, etc), we use a daily scheduled task
#     to destroy stale clusters after 24h.
#     See https://console.cloud.google.com/functions/details/us-central1/clean-substra-tests-ci-deployment
#
# USEFUL LINKS
#   Travis build logs: https://travis-ci.org/github/SubstraFoundation/substra-tests
#   Stale cluster deletion script: https://console.cloud.google.com/functions/details/us-central1/clean-substra-tests-ci-deployment

CLUSTER_NAME_ALLOWED_PREFIX="substra-tests"
CLUSTER_NAME=""
CLUSTER_MACHINE_TYPE="n1-standard-8"
CLUSTER_VERSION="1.15.11-gke.1"
CLUSTER_PROJECT="substra-208412"
CLUSTER_ZONE="europe-west4-a"
SERVICE_ACCOUNT=substra-tests@substra-208412.iam.gserviceaccount.com
IMAGE_SUBSTRA_TESTS_DEPLOY_REPO="substrafoundation/substra-tests-stack"
IMAGE_SUBSTRA_TESTS_DEPLOY_TAG="latest"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CHARTS_DIR="${DIR}/../charts"
KEYS_DIR="${HOME}/.local/" # overridden with --keys-directory=xyz
KEY_SERVICE_ACCOUNT="substra-208412-3be0df12d87a.json"
KEY_KANIKO_SERVICE_ACCOUNT="kaniko-secret.json"

usage() {
cat <<\EOF
Usage: run-ci.sh <options>
  Required option:
    -N|--cluster-name=<name>        The name if the GKE kubernetes cluster to create
  Options:
    -K|--keys-directory=<path>      The path to a folder containing the GKE service account credentials
    -H|-h|--help                    Display this help"
EOF
}

# Parse command-line arguments
for i in "$@"
do
    case $i in
        -N=*|--cluster-name=*)
        CLUSTER_NAME="${i#*=}"
        shift
        ;;
        -K=*|--keys-directory=*)
        KEYS_DIR="${i#*=}"
        shift
        ;;
        -H|-h|--help)
        usage
        exit 0
        ;;
        *)
            # unknown option
        ;;
    esac
done

if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
fi

if [[ -z "${CLUSTER_NAME}" ]]; then
    usage
    exit 1
fi

if [[ "${CLUSTER_NAME}" != "${CLUSTER_NAME_ALLOWED_PREFIX}"* ]]; then
    # The cluster name must start with 'substra-tests'.
    # This is to ensure the cluster gets picked up by the stale cluster deletion script.
    echo "ERROR: Invalid cluster name \"${CLUSTER_NAME}\". The cluster name must start with \"${CLUSTER_NAME_ALLOWED_PREFIX}\"."
    exit 1
fi

echo "KEYS_DIR     = ${KEYS_DIR}"
echo "CLUSTER_NAME = ${CLUSTER_NAME}"

set -evx

# Always delete the cluster, even if the script fails
delete-cluster() {
    yes | gcloud container clusters delete ${CLUSTER_NAME} --zone ${CLUSTER_ZONE} --project ${CLUSTER_PROJECT}
}
trap 'delete-cluster' EXIT

# Log in
gcloud auth activate-service-account ${SERVICE_ACCOUNT} \
    --key-file="${KEYS_DIR}/${KEY_SERVICE_ACCOUNT}"

# Create cluster
gcloud container clusters create ${CLUSTER_NAME} \
    --cluster-version ${CLUSTER_VERSION} \
    --machine-type ${CLUSTER_MACHINE_TYPE} \
    --zone ${CLUSTER_ZONE} \
    --project ${CLUSTER_PROJECT} \
    --service-account ${SERVICE_ACCOUNT} \
    --num-nodes=1

# Configure kubectl
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${CLUSTER_ZONE} --project ${CLUSTER_PROJECT}
KUBE_CONTEXT="gke_${CLUSTER_PROJECT}_${CLUSTER_ZONE}_${CLUSTER_NAME}"

# Configure kaniko
kubectl --context ${KUBE_CONTEXT} create secret generic kaniko-secret --from-file="${KEYS_DIR}/${KEY_KANIKO_SERVICE_ACCOUNT}"

# Configure Tiller
kubectl --context ${KUBE_CONTEXT} create serviceaccount --namespace kube-system tiller
kubectl --context ${KUBE_CONTEXT} create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller
helm --kube-context ${KUBE_CONTEXT} init --service-account tiller --upgrade --wait

# Install docker registry
helm --kube-context ${KUBE_CONTEXT} install stable/docker-registry --name docker-registry --wait
REGISTRY_POD_NAME=$(kubectl get pods -o name --context ${KUBE_CONTEXT} | grep docker-registry)
REGISTRY=$(kubectl get ${REGISTRY_POD_NAME} --template={{.status.podIP}} --context ${KUBE_CONTEXT}):5000

# Deploy
# TODO: change git.substraTests.branch to master (necessary for now because skaffold.yaml doesn't exist in master)`
helm install ${CHARTS_DIR}/substra-tests-stack \
    --kube-context ${KUBE_CONTEXT} \
    --name substra-tests-stack \
    --set image.repository=${IMAGE_SUBSTRA_TESTS_DEPLOY_REPO} \
    --set image.tag=${IMAGE_SUBSTRA_TESTS_DEPLOY_TAG} \
    --set deploy.defaultRepo=${REGISTRY} \
    --set git.substraTests.branch=nightly-tests \
    --wait

# Wait for the substra stack to be deployed
SUBSTRA_TESTS_DEPLOY_POD=$(kubectl --context ${KUBE_CONTEXT} get pods | grep substra-tests-stack | awk '{print $1}')
kubectl --context ${KUBE_CONTEXT} wait pod/${SUBSTRA_TESTS_DEPLOY_POD} --for=condition=ready --timeout=60s
kubectl --context ${KUBE_CONTEXT} logs -f ${SUBSTRA_TESTS_DEPLOY_POD}
if [ "Succeeded" != "$(kubectl --context ${KUBE_CONTEXT} get pod ${SUBSTRA_TESTS_DEPLOY_POD} -o jsonpath='{.status.phase}')" ]; then
    exit 1
fi
echo "Success! ${SUBSTRA_TESTS_DEPLOY_POD} completed with no error."

# Wait for the `substra-tests` pod to be ready
SUBSTRA_TESTS_POD=$(kubectl --context ${KUBE_CONTEXT} get pods -n substra-tests | grep substra-tests | awk '{print $1}')
if ! kubectl --context ${KUBE_CONTEXT} wait pod/${SUBSTRA_TESTS_POD} -n substra-tests --for=condition=ready --timeout=590s; then
    echo 'ERROR: Timeout while waiting for the substra-tests pod. This means the `substra-backend-server` pods never reached the "ready" state.'
    exit 1
fi
echo "Success! ${SUBSTRA_TESTS_POD} is ready."

# Run the tests
kubectl --context ${KUBE_CONTEXT} exec ${SUBSTRA_TESTS_POD} -n substra-tests -- make test
