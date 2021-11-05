#!/usr/bin/env python3


from ci.call import call


def setup_helm() -> None:
    print("\n# Setup Helm")
    call("helm repo add stable https://charts.helm.sh/stable")
    call("helm repo add owkin https://owkin.github.io/charts")
    call("helm repo add bitnami https://charts.bitnami.com/bitnami")
    call("helm repo add twuni https://helm.twun.io")
    call("helm repo add couchdb https://apache.github.io/couchdb-helm")
    call("helm repo add jetstack https://charts.jetstack.io")
