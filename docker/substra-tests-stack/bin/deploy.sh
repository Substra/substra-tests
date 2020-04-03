#!/bin/bash

set -e
set -v

DEFAULT_REPO="$1"
SUBSTRA_TESTS_BRANCH="$2"

# Init helm
helm init --upgrade --force-upgrade --wait

# Clone repos
rm -rf src
mkdir -p src
cd src
git clone --depth 1 https://github.com/SubstraFoundation/substra-backend.git
git clone --depth 1 https://github.com/SubstraFoundation/hlf-k8s.git
git clone --depth 1 https://github.com/SubstraFoundation/substra-tests.git --branch "${SUBSTRA_TESTS_BRANCH}"
cd ..

# Patch skaffold files for kaniko
python ./kaniko-patch.py

# Deploy hlf-k8s
cd src/hlf-k8s
skaffold run --default-repo ${DEFAULT_REPO}

# Deploy backend
cd ../substra-backend
sed "s@defaultDomain: http://substra-backend.node-1.com@defaultDomain: http://backend-org-1-substra-backend-server.org-1.svc.cluster.local:8000@g" ./skaffold.yaml > skaffold-updated-tmp.yaml
sed "s@defaultDomain: http://substra-backend.node-2.com@defaultDomain: http://backend-org-2-substra-backend-server.org-2.svc.cluster.local:8000@g" ./skaffold-updated-tmp.yaml > skaffold-updated.yaml
skaffold run --default-repo ${DEFAULT_REPO} -f skaffold-updated.yaml

# Deploy substra-tests
cd ../substra-tests
skaffold run --default-repo ${DEFAULT_REPO}
