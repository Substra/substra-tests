#/bin/bash

SERVICE_ACCOUNT=substra-tests@substra-208412.iam.gserviceaccount.com
SERVICE_ACCOUNT_KEY="${HOME}/.local/substra-208412-3be0df12d87a.json"
KANIKO_SERVICE_ACCOUNT_KEY="${HOME}/.local/kaniko-secret.json"
CLUSTER_NAME=substra-tests
HLF_K8S_IMAGE="substrafoundation/hlf-k8s:latest"
SUBSTRA_BACKEND_IMAGES="substrafoundation/substra-backend:latest,substrafoundation/celeryworker:latest,substrafoundation/celerybeat:latest"

set -e

# sed command for linux and macos
SED_COMMAND="sed -i '' "
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_COMMAND="sed -i'' "
fi

# Log in
gcloud auth activate-service-account ${SERVICE_ACCOUNT} \
    --key-file="${SERVICE_ACCOUNT_KEY}"

# Create cluster
gcloud container clusters create --num-nodes=1 ${CLUSTER_NAME}
gcloud container clusters get-credentials ${CLUSTER_NAME}
#gcloud container clusters describe ${CLUSTER_NAME}
KUBE_CONTEXT=$(kubectl config get-contexts -o name | grep ${CLUSTER_NAME})
kubectl config set-context $KUBE_CONTEXT

# Configure kaniko
kubectl create secret generic kaniko-secret --from-file=$KANIKO_SERVICE_ACCOUNT_KEY

# Configure Tiller
kubectl create serviceaccount --namespace kube-system tiller
kubectl create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller
helm --kube-context $KUBE_CONTEXT  init --service-account tiller --upgrade --wait

# Fetch substra ressources 
mkdir substra-resources
cd substra-resources

# Clone substra
git clone --depth 1 git@github.com:SubstraFoundation/hlf-k8s.git
git clone --depth 1 git@github.com:SubstraFoundation/substra-backend.git
git clone --depth 1 git@github.com:SubstraFoundation/substra-chaincode.git

# Deploy
cd hlf-k8s; skaffold deploy --kube-context=$KUBE_CONTEXT --images=${HLF_K8S_IMAGE} ; cd -
cd substra-backend; skaffold deploy --kube-context=$KUBE_CONTEXT --images=${SUBSTRA_BACKEND_IMAGES} ; cd -

# Delete cluster
gcloud container clusters delete ${CLUSTER_NAME}
