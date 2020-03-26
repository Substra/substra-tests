#/bin/bash

CLUSTER_NAME=substra-tests
CLUSTER_MACHINE_TYPE="n1-standard-8"
CLUSTER_VERSION="1.15.8-gke.3"
CLUSTER_ZONE="europe-west4-a"
CLUSTER_PROJECT="substra-208412"
SERVICE_ACCOUNT=substra-tests@substra-208412.iam.gserviceaccount.com
IMAGE_SUBSTRA_TESTS_DEPLOY_REPO="substrafoundation/substra-tests-deploy"
IMAGE_SUBSTRA_TESTS_DEPLOY_TAG="latest"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CHARTS_DIR="${DIR}/../charts"
KEY_SERVICE_ACCOUNT="substra-208412-3be0df12d87a.json"
KEY_KANIKO_SERVICE_ACCOUNT="kaniko-secret.json"

KEYS_DIR="${HOME}/.local/"

# Parse command-line arguments
for i in "$@"
do
    case $i in
        -K=*|--keys-directory=*)
        KEYS_DIR="${i#*=}"
        shift # past argument=value
        ;;
        --default)
        DEFAULT=YES
        shift # past argument with no value
        ;;
        *)
            # unknown option
        ;;
    esac
done
echo "KEYS_DIR  = ${KEYS_DIR}"
if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
fi

set -evx

# Always delete the cluster, even if the bash script fails
delete-cluster() {
    echo "Deleting cluster"
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
gcloud components list --show-versions --filter kubectl
kubectl version
echo $PATH
ls -la $HOME/bin/
# helm version
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
time kubectl wait pod/${SUBSTRA_TESTS_POD} --for=condition=ready --context ${KUBE_CONTEXT} --timeout=1200s

# Run the tests
kubectl --context ${KUBE_CONTEXT} exec ${SUBSTRA_TESTS_POD} -- make test
