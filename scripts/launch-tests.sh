#/bin/bash

CLUSTER_NAME=substra-tests
CLUSTER_MACHINE_TYPE="n1-standard-8"
CLUSTER_VERSION="1.15.8-gke.3"
CLUSTER_ZONE="europe-west4-a"
CLUSTER_PROJECT="substra-208412"
SERVICE_ACCOUNT=substra-tests@substra-208412.iam.gserviceaccount.com
SERVICE_ACCOUNT_KEY="${HOME}/.local/substra-208412-3be0df12d87a.json"
KANIKO_SERVICE_ACCOUNT_KEY="${HOME}/.local/kaniko-secret.json"

set -e
set -v
set -x

# sed command for linux and macos
SED_COMMAND="sed -i '' "
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_COMMAND="sed -i'' "
fi

# Log in
gcloud auth activate-service-account ${SERVICE_ACCOUNT} \
    --key-file="${SERVICE_ACCOUNT_KEY}"

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
kubectl --context ${KUBE_CONTEXT} create secret generic kaniko-secret --from-file=$KANIKO_SERVICE_ACCOUNT_KEY

# Configure Tiller
kubectl --context ${KUBE_CONTEXT} create serviceaccount --namespace kube-system tiller
kubectl --context ${KUBE_CONTEXT} create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller
helm --kube-context ${KUBE_CONTEXT} init --service-account tiller --upgrade --wait

# Install docker registry
helm --kube-context ${KUBE_CONTEXT} install stable/docker-registry --name docker-registry --wait
REGISTRY_POD_NAME=$(kubectl get pods -o name | grep docker-registry)
REGISTRY=$(kubectl get ${REGISTRY_POD_NAME} --template={{.status.podIP}}):5000

# Deploy
sed "s/<<<REGISTRY>>>/${REGISTRY}/g" ./deploy.template.sh > ./tmp/deploy.sh
chmod +x ./tmp/deploy.sh
kubectl apply -f deployment-skaffold.yaml
SKAFFOLD_POD=$(kubectl get pods -n kube-system  | grep skaffold | awk '{print $1}')
kubectl wait pod/${SKAFFOLD_POD} --for=condition=ready --context ${KUBE_CONTEXT} --timeout=-1s -n kube-system
kubectl cp ./tmp/deploy.sh ${SKAFFOLD_POD}:/ -n kube-system
kubectl cp ./kaniko-patch.py ${SKAFFOLD_POD}:/ -n kube-system
# then SSH into the pod and run `./deploy.sh` (todo: automate this step)

# # Delete cluster
# yes | gcloud container clusters delete ${CLUSTER_NAME} --zone ${CLUSTER_ZONE} --project ${CLUSTER_PROJECT}
