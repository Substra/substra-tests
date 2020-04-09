#!/usr/bin/env python3
#
# ABOUT
#   This script runs the full substra CI pipeline.
#   "Gotta test them all!" ™
#
#   This script performs the following operations:
#     - Create a new GKE cluster
#     - Clone the repos:
#       - hlf-k8s
#       - substa-backend
#       - substra-tests
#     - Build the images using Google Cloud Builds
#     - Deploy (skaffold)
#     - Wait for the substra application stack to be up and ready (i.e. wait
#       for the substra-backend servers readiness probe)
#     - Run the full 'substra-tests' test suite
#     - Destroy the cluster
#
# AUTOMATIC CLUSTER DESTRUCTION
#   We ensure the GKE cluster always gets destroyed at the end of the execution
#   of this script.
#
#   We use 2 mechanisms:
#   - When the script exits, we destroy the cluster as the final step. That's
#     the case even if the script exits with an error or with an interruption
#     (Ctrl+C). See `trap` command.
#   - In any other "abrupt shutdown" case (system shutdown, Travis build
#     canceled, etc), we use a daily scheduled task to destroy stale clusters
#     after 24h. See https://console.cloud.google.com/functions/details/us-central1/clean-substra-tests-ci-deployment
#
# USEFUL LINKS
#   Travis build logs: https://travis-ci.org/github/SubstraFoundation/substra-tests
#   Stale cluster deletion script:
#       https://console.cloud.google.com/functions/details/us-central1/clean-substra-tests-ci-deployment
import os
import time
import json
import shutil
import string
import random
import argparse
import subprocess


CLUSTER_NAME_ALLOWED_PREFIX = 'substra-tests'
CLUSTER_NAME = ''
CLUSTER_MACHINE_TYPE = 'n1-standard-8'
CLUSTER_VERSION = '1.15.11-gke.1'
CLUSTER_PROJECT = 'substra-208412'
CLUSTER_ZONE = 'europe-west4-a'

SERVICE_ACCOUNT = 'substra-tests@substra-208412.iam.gserviceaccount.com'
KEY_SERVICE_ACCOUNT = 'substra-208412-3be0df12d87a.json'

SUBSTRA_TESTS_BRANCH = 'nightly-tests'
SUBSTRA_BACKEND_BRANCH = 'master'
HLF_K8S_BRANCH = 'master'

DIR = os.path.dirname(os.path.realpath(__file__))
CHARTS_DIR = os.path.realpath(os.path.join(DIR, '../charts/'))
KEYS_DIR = os.path.realpath(os.path.join(os.getenv('HOME'), '.local/'))
SOURCE_DIR = os.path.realpath(os.path.join(DIR, 'src'))

KUBE_CONTEXT = ''
RUN_TAG = ''.join(random.choice(string.ascii_letters + '0123456789') for _ in range(10))

KANIKO_CACHE_TTL = '168h'  # 1 week


def call(cmd):
    print(cmd)
    return subprocess.check_call([cmd], shell=True)


def cluster_name(value):
    """
    Validate the --cluster-name argument
    The cluster name must start with 'substra-tests'.
    This is to ensure the cluster gets picked up by the stale cluster deletion script.
    """

    if not value.startswith(CLUSTER_NAME_ALLOWED_PREFIX):
        raise argparse.ArgumentTypeError(
            f'Invalid cluster name "{value}". '
            f'The cluster name must start with "{CLUSTER_NAME_ALLOWED_PREFIX}".')

    if len(value) > 35:
        raise argparse.ArgumentTypeError(
            f'Invalid cluster name "{value}". '
            f'The cluster name must not be longer than 35 characters.')

    return value


def arg_parse():

    global KEYS_DIR
    global CLUSTER_NAME
    global SUBSTRA_TESTS_BRANCH
    global SUBSTRA_BACKEND_BRANCH
    global HLF_K8S_BRANCH
    global KANIKO_CACHE_TTL

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--cluster-name', type=cluster_name, required=True,
                        help='The prefix name if the GKE kubernetes cluster to create')
    parser.add_argument('-K', '--keys-directory', type=str, default=KEYS_DIR,
                        help='The path to a folder containing the GKE service account credentials')
    parser.add_argument('--substra-tests', type=str, default=SUBSTRA_TESTS_BRANCH,
                        help='substra-tests branch', metavar='GIT_BRANCH')
    parser.add_argument('--substra-backend', type=str, default=SUBSTRA_BACKEND_BRANCH,
                        help='substra-backend branch', metavar='GIT_BRANCH')
    parser.add_argument('--hlf-k8s', type=str, default=HLF_K8S_BRANCH,
                        help='hlf-k8s branch', metavar='GIT_BRANCH')
    parser.add_argument('--no-cache', action='store_true',
                        help='Use this option to disable kaniko caching')

    args = vars(parser.parse_args())

    CLUSTER_NAME = args['cluster_name']
    # Add RUN_TAG to cluster name to make it non-deterministic in case of retry
    CLUSTER_NAME += f'-{RUN_TAG[:40-len(CLUSTER_NAME)-1]}'
    CLUSTER_NAME = CLUSTER_NAME.lower()   # Make it lower for gcloud compatibility

    KEYS_DIR = args['keys_directory']
    SUBSTRA_TESTS_BRANCH = args['substra_tests']
    SUBSTRA_BACKEND_BRANCH = args['substra_backend']
    HLF_K8S_BRANCH = args['hlf_k8s']
    if args['no_cache']:
        KANIKO_CACHE_TTL = '-1h'

    print(
        f'KEYS_DIR\t\t= {KEYS_DIR}\n'
        f'CLUSTER_NAME\t\t= {CLUSTER_NAME}\n'
        f'SUBSTRA_TESTS_BRANCH\t= {SUBSTRA_TESTS_BRANCH}\n'
        f'SUBSTRA_BACKEND_BRANCH\t= {SUBSTRA_BACKEND_BRANCH}\n'
        f'HLF_K8S_BRANCH\t\t= {HLF_K8S_BRANCH}\n'
        f'KANIKO_CACHE_TTL\t= {KANIKO_CACHE_TTL}\n'
    )

    return


def gcloud_login():
    print('# Log into Google Cloud')
    call(f'gcloud auth activate-service-account {SERVICE_ACCOUNT} --key-file={KEYS_DIR}/{KEY_SERVICE_ACCOUNT}')


def get_kube_context():
    global KUBE_CONTEXT

    print('\n# Fetch kubernetes context')
    call(f'gcloud container clusters get-credentials {CLUSTER_NAME} --zone {CLUSTER_ZONE} --project {CLUSTER_PROJECT}')
    KUBE_CONTEXT = f'gke_{CLUSTER_PROJECT}_{CLUSTER_ZONE}_{CLUSTER_NAME}'


def create_cluster_async():
    print('\n# Create GKE cluster')
    cmd = f'gcloud container clusters create {CLUSTER_NAME} '\
          f'--cluster-version {CLUSTER_VERSION} '\
          f'--machine-type {CLUSTER_MACHINE_TYPE} '\
          f'--service-account {SERVICE_ACCOUNT} '\
          f'--num-nodes=1 '\
          f'--zone={CLUSTER_ZONE} '\
          f'--project={CLUSTER_PROJECT} '\
          f'--async'
    call(cmd)


def delete_cluster_async():
    wait_for_cluster()
    print('\n# Delete cluster')
    cmd = f'yes | gcloud container clusters delete {CLUSTER_NAME} --zone ' \
          f'{CLUSTER_ZONE} --project {CLUSTER_PROJECT} --async --quiet'
    call(cmd)


def wait_for_cluster():
    print('# Waiting for GKE cluster to be ready ...', end='')

    while True:
        output = subprocess.check_output(
            [f'gcloud container clusters list --filter="name={CLUSTER_NAME}" 2>/dev/null'],
            shell=True
        ).decode().strip()

        try:
            status = output.split('\n')[1].split(' ')[-1]
            if status not in ['RUNNING', 'PROVISIONING']:
                raise Exception(f'Unknown status {status}')
        except Exception as e:
            print('\nFATAL: Error retrieving cluster status. Output was:')
            print(output)
            raise(e)

        if status == 'RUNNING':
            print('Done.')
            break

        print('.', end='', flush=True)
        time.sleep(5)


def setup_tiller():
    print('\n# Set up Tiller')
    call(f'kubectl --context {KUBE_CONTEXT} create serviceaccount --namespace kube-system tiller')
    call(f'kubectl --context {KUBE_CONTEXT} create clusterrolebinding tiller-cluster-rule ' +
         f'--clusterrole=cluster-admin --serviceaccount=kube-system:tiller')
    call(f'helm --kube-context {KUBE_CONTEXT} init --service-account tiller --upgrade --wait')


def clone_repos():
    if os.path.exists(SOURCE_DIR):
        shutil.rmtree(SOURCE_DIR)

    os.makedirs(SOURCE_DIR)

    print('\n# Clone repos')
    commit_backend = clone_substra_backend()
    commit_hlf = clone_hlf_k8s()
    commit_tests = clone_substra_tests()

    print(
        f'\nCommit hashes:\n'
        f'- substra-backend: \t{commit_backend}\n'
        f'- hlf-k8s: \t\t{commit_hlf}\n'
        f'- substra-tests: \t{commit_tests}\n'
    )
    return [
        {'name': 'hlf-k8s',
         'images': ['hlf-k8s'],
         'commit': commit_hlf,
         'branch': HLF_K8S_BRANCH},
        {'name': 'substra-backend',
         'images': ['substra-backend', 'celeryworker', 'celerybeat', 'flower'],
         'commit': commit_backend,
         'branch': SUBSTRA_BACKEND_BRANCH},
        {'name': 'substra-tests',
         'images': ['substra-tests'],
         'commit': commit_tests,
         'branch': SUBSTRA_TESTS_BRANCH},
    ]


def clone_repository(dirname, url, branch, commit=None):
    call(f'git clone -q --depth 1 {url} --branch "{branch}" {dirname}')

    if commit is None:
        commit = subprocess.check_output(
            [f'git --git-dir={dirname}/.git rev-parse origin/{branch}'],
            shell=True
        ).decode().strip()

    return commit


def clone_substra_backend():
    url = 'https://github.com/SubstraFoundation/substra-backend.git'
    return clone_repository(
        dirname=os.path.join(SOURCE_DIR, 'substra-backend'),
        url=url,
        branch=SUBSTRA_BACKEND_BRANCH
    )


def clone_hlf_k8s():
    url = 'https://github.com/SubstraFoundation/hlf-k8s.git'
    return clone_repository(
        dirname=os.path.join(SOURCE_DIR, 'hlf-k8s'),
        url=url,
        branch=HLF_K8S_BRANCH
    )


def clone_substra_tests():
    url = 'https://github.com/SubstraFoundation/substra-tests.git'
    return clone_repository(
        dirname=os.path.join(SOURCE_DIR, 'substra-tests'),
        url=url,
        branch=SUBSTRA_TESTS_BRANCH
    )


def build_images(configs):
    tag = f'substra-tests-{RUN_TAG}'
    images = {}

    print('# Queue docker image builds')
    for config in configs:
        for image in config['images']:
            build_id = build_image(
                tag=tag,
                image=image,
                branch=config['branch'],
                commit=config['commit']
            )
            images[build_id] = image

    wait_for_builds(tag, images)


def build_image(tag, image, branch, commit):
    config_file = os.path.join(DIR, f'cloudbuild/{image}.yaml')

    cmd = f'gcloud builds submit '\
        f'--config={config_file} '\
        f'--no-source '\
        f'--async --project={CLUSTER_PROJECT} '\
        f'--substitutions=_BUILD_TAG={tag},_BRANCH={branch},_COMMIT={commit},_KANIKO_CACHE_TTL={KANIKO_CACHE_TTL}'

    build_id = subprocess.check_output([cmd], shell=True)\
        .decode().strip().split('\n')[-1].split(' ')[0]

    return build_id


def wait_for_builds(tag, images):
    print('\n# Waiting for builds to complete ...', end='')
    do_wait = True
    while do_wait:
        build_list = subprocess.check_output(
            [f'gcloud builds list --filter="tags={tag}"'],
            shell=True
        ).decode().strip()

        builds = build_list.split('\n')[1:]

        num_builds = len(builds)
        num_success = build_list.count('SUCCESS')
        num_failed = build_list.count('TIMEOUT') + build_list.count('CANCELLED') + build_list.count('FAIL')

        do_wait = (num_builds != (num_success + num_failed))

        time.sleep(5)
        print('.', end='', flush=True)

    print(' Done.')

    if num_failed:
        print(f'FATAL: One or more builds failed. See logs for more details')
        for build in builds:
            if 'TIMEOUT' in build or 'CANCELLED' in build or 'FAIL' in build:
                build_id = build.split(' ')[0]
                image = images[build_id]
                print(f"- [{image}]: "
                      f"https://console.cloud.google.com/cloud-build/builds/{build_id}?project={CLUSTER_PROJECT}")
        raise Exception('docker image build(s) failed.')


def deploy_all(configs):
    print('\n# Deploy helm charts')
    for config in configs:
        deploy(config)


def deploy(config):
    artifacts_file = create_build_artifacts(config)
    skaffold_file = patch_skaffold_file(config)
    call(f'skaffold deploy --kube-context={KUBE_CONTEXT} '
         f'-f={skaffold_file} -a={artifacts_file} --status-check=true')


def create_build_artifacts(config):
    # Gcloud Build artifacts
    artifacts_file = os.path.join(SOURCE_DIR, config['name'], 'tags.json')

    with open(artifacts_file, 'w') as file:
        tags = {'builds': []}
        for image in config['images']:
            tags['builds'].append({
                'imageName': f'substrafoundation/{image}',
                'tag': f'gcr.io/{CLUSTER_PROJECT}/{image}:{config["commit"]}'
            })

        json.dump(tags, file)

    return artifacts_file


def patch_skaffold_file(config):

    skaffold_file = os.path.join(SOURCE_DIR, config['name'], 'skaffold.yaml')

    # Patch skaffold file
    with open(skaffold_file, 'r') as file:
        filedata = file.read()

    # Set statusCheckDeadlineSeconds to 300s instead of 120 default
    filedata = filedata.replace(
        'deploy:',
        'deploy:\n  statusCheckDeadlineSeconds: 300'
    )

    filedata = filedata.replace(
        'chartPath: charts/',
        f'chartPath: {os.path.join(SOURCE_DIR, config["name"],"charts/")}'
    )

    if config['name'] == 'substra-backend':
        # Default Domain
        filedata = filedata.replace(
            'defaultDomain: http://substra-backend.node-1.com',
            'defaultDomain: http://backend-org-1-substra-backend-server.org-1.svc.cluster.local:8000'
        )
        filedata = filedata.replace(
            'defaultDomain: http://substra-backend.node-2.com',
            'defaultDomain: http://backend-org-2-substra-backend-server.org-2.svc.cluster.local:8000'
        )

    with open(skaffold_file, 'w') as file:
        file.write(filedata)

    return skaffold_file


def run_tests():
    print('# Wait for the substra-tests pod to be ready')
    substra_tests_pod = subprocess.check_output(
        [f'kubectl --context {KUBE_CONTEXT} get pods -n substra-tests | grep substra-tests'],
        shell=True
    ).decode().strip().split(' ')[0]

    try:
        call(f'kubectl --context {KUBE_CONTEXT} wait pod/{substra_tests_pod} '
             f'-n substra-tests --for=condition=ready --timeout=590s')
    except Exception:
        print('ERROR: Timeout while waiting for the substra-tests pod. '
              'This means the `substra-backend-server` pods never reached the "ready" state.')

    print('\n# Run tests')
    call(f'kubectl --context {KUBE_CONTEXT} exec {substra_tests_pod} -n substra-tests -- make test')


def main():
    arg_parse()

    try:
        gcloud_login()
        create_cluster_async()
        configs = clone_repos()
        build_images(configs)
        wait_for_cluster()
        get_kube_context()
        setup_tiller()
        deploy_all(configs)
        run_tests()

    finally:

        if os.path.exists(SOURCE_DIR):
            shutil.rmtree(SOURCE_DIR)

        delete_cluster_async()


if __name__ == '__main__':
    main()
