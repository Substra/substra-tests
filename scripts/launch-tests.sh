#/bin/bash

SERVICE_ACCOUNT=substra-tests@substra-208412.iam.gserviceaccount.com
SERVICE_ACCOUNT_KEY="${HOME}/.local/substra-208412-3be0df12d87a.json"
KANIKO_SERVICE_ACCOUNT_KEY="${HOME}/.local/kaniko-secret.json"
CLUSTER_NAME=substra-tests
WORKDIR=$(pwd)
IMAGE_HLF_K8S="substrafoundation/hlf-k8s:local"
IMAGE_SUBSTRA_BACKEND="substrafoundation/substra-backend:local"
IMAGE_CELERYWORKER="substrafoundation/celeryworker:local"
IMAGE_CELERYBEAT="substrafoundation/celerybeat:local"
IMAGE_FLOWER="substrafoundation/flower:local"

set -e
set -v

# sed command for linux and macos
SED_COMMAND="sed -i '' "
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_COMMAND="sed -i'' "
fi

# Log in
gcloud auth activate-service-account ${SERVICE_ACCOUNT} \
    --key-file="${SERVICE_ACCOUNT_KEY}"

# Create cluster
gcloud container clusters create --num-nodes=1 ${CLUSTER_NAME} --service-account ${SERVICE_ACCOUNT} --machine-type=n1-standard-8 --zone europe-west4-a --project substra-208412
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone europe-west4-a --project substra-208412
gcloud container clusters describe ${CLUSTER_NAME} --zone europe-west4-a --project substra-208412

KUBE_CONTEXT=$(kubectl config get-contexts -o name | grep ${CLUSTER_NAME})

# Configure kaniko
kubectl --context $KUBE_CONTEXT create secret generic kaniko-secret --from-file=$KANIKO_SERVICE_ACCOUNT_KEY

# Configure Tiller
kubectl --context $KUBE_CONTEXT create serviceaccount --namespace kube-system tiller
kubectl --context $KUBE_CONTEXT create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller
helm --kube-context $KUBE_CONTEXT init --service-account tiller --upgrade --wait

# Install registry
helm --kube-context $KUBE_CONTEXT install stable/docker-registry --name docker-registry --wait

# Build substra-tests docker image
kubectl --context $KUBE_CONTEXT apply -f kaniko.yaml


# Fetch substra ressources

mkdir -p substra-resources
cd substra-resources

# Clone substra
git clone --depth 1 git@github.com:SubstraFoundation/hlf-k8s.git
git clone --depth 1 git@github.com:SubstraFoundation/substra-backend.git
git clone --depth 1 git@github.com:SubstraFoundation/substra-chaincode.git


# Deploy

cd hlf-k8s; skaffold deploy --kube-context=$KUBE_CONTEXT --images=${IMAGE_HLF_K8S} --default-repo docker-registry.default.svc.cluster.local:5000; cd -
cd substra-backend; skaffold deploy --kube-context=$KUBE_CONTEXT --images=${IMAGE_SUBSTRA_BACKEND} --images=${IMAGE_CELERYWORKER} --images=${IMAGE_CELERYBEAT} --images=${IMAGE_FLOWER} --default-repo docker-registry.default.svc.cluster.local:5000; cd -

cd $WORKDIR

# # Delete cluster
# yes | gcloud container clusters delete ${CLUSTER_NAME} --zone europe-west4-a --project substra-208412

# # Remove substra resources
# rm -rf substra-resources
