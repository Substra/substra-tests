#!/bin/bash

set -e
set -v

DEFAULT_REPO="$1"

# Init helm
helm init --upgrade --force-upgrade --wait

# Clone repos
rm -rf src
mkdir -p src
cd src
git clone --depth 1 https://github.com/SubstraFoundation/substra-backend.git
git clone --depth 1 https://github.com/SubstraFoundation/hlf-k8s.git
cd ..

# Patch skaffold files for kaniko
python -m venv .venv
source .venv/bin/activate
pip install pyyaml
python ./kaniko-patch.py

# Deploy hlf-k8s
cd src/hlf-k8s
skaffold run --default-repo ${DEFAULT_REPO}

# Deploy backend
cd ../substra-backend

sed "s@defaultDomain: http://substra-backend.node-1.com@defaultDomain: http://backend-org-1-substra-backend-server.org-1.svc.cluster.local:8000@g" ./skaffold.yaml > skaffold-updated-tmp.yaml
sed "s@defaultDomain: http://substra-backend.node-2.com@defaultDomain: http://backend-org-2-substra-backend-server.org-2.svc.cluster.local:8000@g" ./skaffold-updated-tmp.yaml > skaffold-updated.yaml

skaffold run --default-repo ${DEFAULT_REPO} -f skaffold-updated.yaml
