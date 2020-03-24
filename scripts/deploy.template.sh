#!/bin/bash

set -e
set -v

REGISTRY='<<<REGISTRY>>>'

# Install kubectl
curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
chmod +x ./kubectl
mv ./kubectl /usr/local/bin/kubectl

# Install helm (v2.14.2)
curl https://get.helm.sh/helm-v2.14.2-linux-amd64.tar.gz -o helm-v2.14.2-linux-amd64.tar.gz
tar xzf helm-v2.14.2-linux-amd64.tar.gz
mv linux-amd64/helm linux-amd64/tiller /usr/local/bin/
helm init --upgrade --force-upgrade --wait

# Install skaffold (v1.0.1)
curl -Lo skaffold https://storage.googleapis.com/skaffold/releases/v1.0.1/skaffold-linux-amd64 && chmod +x skaffold && mv skaffold /usr/local/bin

# Clone repos
mkdir src
cd src
git clone --depth 1 https://github.com/SubstraFoundation/substra-backend.git
git clone --depth 1 https://github.com/SubstraFoundation/hlf-k8s.git
cd ..

# Patch skaffold files for kaniko
python -m venv .venv
source .venv/bin/activate
pip install pyyaml
python kaniko-patch.py

# Deploy hlf-k8s
cd src/hlf-k8s
skaffold run --default-repo ${REGISTRY}

# Deploy backend
cd ../substra-backend
skaffold run --default-repo ${REGISTRY}
