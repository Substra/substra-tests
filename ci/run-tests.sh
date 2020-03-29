#/bin/bash

# Please use a cluster name that starts with 'substra-tests' so that it gets
# picked up by the automatic stale cluster deletion script.
# See https://console.cloud.google.com/functions/details/us-central1/clean-substra-tests-ci-deployment
CLUSTER_NAME="substra-tests" # overridden with --cluster-name=xyz
CLUSTER_MACHINE_TYPE="n1-standard-8"
CLUSTER_VERSION="1.15.11-gke.1"
CLUSTER_PROJECT="substra-208412"
CLUSTER_ZONE="europe-west4-a"
SERVICE_ACCOUNT=substra-tests@substra-208412.iam.gserviceaccount.com
IMAGE_SUBSTRA_TESTS_DEPLOY_REPO="substrafoundation/substra-tests-deploy"
IMAGE_SUBSTRA_TESTS_DEPLOY_TAG="latest"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CHARTS_DIR="${DIR}/../charts"
KEYS_DIR="${HOME}/.local/" # overridden with --keys-directory=xyz
KEY_SERVICE_ACCOUNT="substra-208412-3be0df12d87a.json"
KEY_KANIKO_SERVICE_ACCOUNT="kaniko-secret.json"

# Parse command-line arguments
for i in "$@"
do
    case $i in
        -K=*|--keys-directory=*)
        KEYS_DIR="${i#*=}"
        shift
        ;;
        -N=*|--cluster-name=*)
        CLUSTER_NAME="${i#*=}"
        shift
        ;;
        --default)
        DEFAULT=YES
        shift
        ;;
        *)
            # unknown option
        ;;
    esac
done
echo "KEYS_DIR      = ${KEYS_DIR}"
echo "CLUSTER_NAME  = ${CLUSTER_NAME}"
echo "CLUSTER_ZONE  = ${CLUSTER_ZONE}"
if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
fi

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
KUBE_CONTEXT=$(kubectl config get-contexts -o name | grep ${CLUSTER_NAME})

# Configure kaniko
kubectl --context ${KUBE_CONTEXT} create secret generic kaniko-secret --from-file="${KEYS_DIR}/${KEY_KANIKO_SERVICE_ACCOUNT}"

# Configure Tiller
kubectl --context ${KUBE_CONTEXT} create serviceaccount --namespace kube-system tiller
kubectl --context ${KUBE_CONTEXT} create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller
helm --kube-context ${KUBE_CONTEXT} init --service-account tiller --upgrade --wait

# Install docker registry
helm --kube-context ${KUBE_CONTEXT} install stable/docker-registry --name docker-registry --wait
REGISTRY_POD_NAME=$(kubectl get pods -o name --context ${KUBE_CONTEXT}| grep docker-registry)
REGISTRY=$(kubectl get ${REGISTRY_POD_NAME} --template={{.status.podIP}} --context ${KUBE_CONTEXT}):5000

# Deploy substra
helm install ${CHARTS_DIR}/substra-tests-deploy \
    --namespace kube-system \
    --kube-context ${KUBE_CONTEXT} \
    --name substra-tests-deploy \
    --set image.repository=${IMAGE_SUBSTRA_TESTS_DEPLOY_REPO} \
    --set image.tag=${IMAGE_SUBSTRA_TESTS_DEPLOY_TAG} \
    --set deploy.defaultRepo=${REGISTRY} \
    --set serviceAccount=tiller

# Deploy substra-tests
helm install ${CHARTS_DIR}/substra-tests \
    --kube-context ${KUBE_CONTEXT} \
    --name substra-tests \
    --set image.repository=${REGISTRY}/substrafoundation/substra-tests \
    --set image.tag=local \
    --set deploy.defaultRepo=${REGISTRY}

# Wait for the pod
SUBSTRA_TESTS_POD=$(kubectl get pods --context ${KUBE_CONTEXT} | grep substra-tests | grep -v kaniko | awk '{print $1}')

timed-out() {
    echo 'ERROR: Timeout while waiting for the substra-tests pod.'
    echo 'This typically means an error ocurred during one of the following steps:'
    echo '  - `subtra-tests-deploy` deployment on the cluster (check status of deployment `subtra-test-deploy` using k9s) '
    echo '  - `deploy.sh` script execution within the `substra-test-deploy` pod (check logs of `substra-tests-deploy` pod)'
    echo '  - deployment of the substra stack, i.e. hlf-k8s and substra-backend (usual substra stack troubleshooting methods apply)'
    echo 'It would be impractical to dump the statuses and logs of all the kubernetes resources here, '
    echo 'so you might have to instantiate a new kubernetes cluster in order to reproduce and investigate the error. '
    echo 'To do so, you can run this script (or run commands from it) on your local machine, and use k9s to investigate where '
    echo 'the error occurs.'
}

wait-for-pod() {
    # Travis kills the build if we don't send output for more than 10 min.
    # Send some output every 5 min.
    TS_START=$(date +%s)
    until kubectl wait pod/${SUBSTRA_TESTS_POD} --for=condition=ready --context ${KUBE_CONTEXT} --timeout=300s; do
        ELAPSED_SEC=$(($(date +%s) - ${TS_START}))
        if [ ${ELAPSED_SEC} -gt 1200 ]; then
            timed-out
            return
        fi
        echo 'Waiting for the substra-tests pod...'
    done
}

time wait-for-pod

# Run the tests
kubectl --context ${KUBE_CONTEXT} exec ${SUBSTRA_TESTS_POD} -- make test
