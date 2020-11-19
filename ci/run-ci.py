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
#       - substra-chaincode
#     - Build the docker images using Google Cloud Builds
#     - Deploy these images using skaffold
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
#   - Travis build history: https://travis-ci.org/github/SubstraFoundation/substra-tests/builds
#   - Stale cluster deletion script:
#       https://console.cloud.google.com/functions/details/us-central1/clean-substra-tests-ci-deployment
import os
import time
import json
import shutil
import string
import random
import argparse
import subprocess
import sys
import yaml

CLUSTER_NAME_ALLOWED_PREFIX = 'substra-tests'
CLUSTER_NAME = ''
PVC_VOLUME_NAME_PREFIX = ''
CLUSTER_MACHINE_TYPE = 'n1-standard-8'

CLUSTER_PROJECT = 'substra-208412'

# Zone must be specific (e.g. "europe-west1-b" and not "europe-west1")
# or else several kubernetes nodes will be created instead of just one,
# which can lead to pod/volume affinity issues at runtime.
CLUSTER_ZONE = 'europe-west4-a'

SERVICE_ACCOUNT = 'substra-tests@substra-208412.iam.gserviceaccount.com'
KEY_SERVICE_ACCOUNT = 'substra-208412-3be0df12d87a.json'

SUBSTRA_TESTS_BRANCH = 'master'
SUBSTRA_BRANCH = 'master'
SUBSTRA_BACKEND_BRANCH = 'master'
SUBSTRA_CHAINCODE_BRANCH = 'master'
HLF_K8S_BRANCH = 'master'

CHAINCODE_COMMIT = ''

DIR = os.path.dirname(os.path.realpath(__file__))
CHARTS_DIR = os.path.realpath(os.path.join(DIR, '../charts/'))
KEYS_DIR = os.path.realpath(os.path.join(os.getenv('HOME'), '.local/'))

KUBE_CONTEXT = ''
RUN_TAG = ''.join(random.choice(string.ascii_letters + '0123456789') for _ in range(10))
SOURCE_DIR = os.path.realpath(os.path.join(DIR, 'src', RUN_TAG))

KANIKO_CACHE_TTL = '168h'  # 1 week

BACKEND_CELERY_CONCURRENCY = 4

TESTS_CONCURRENCY = 5
TESTS_FUTURE_TIMEOUT = 400


def call(cmd):
    print(f'+ {cmd}')
    return subprocess.check_call([cmd], shell=True)


def call_output(cmd, print_cmd=True):
    if print_cmd:
        print(f'+ {cmd}')
    return subprocess.check_output([cmd], shell=True, stderr=subprocess.STDOUT).decode().strip()


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
    global CLUSTER_MACHINE_TYPE
    global CLUSTER_NAME
    global PVC_VOLUME_NAME_PREFIX
    global SUBSTRA_TESTS_BRANCH
    global SUBSTRA_BRANCH
    global SUBSTRA_BACKEND_BRANCH
    global SUBSTRA_CHAINCODE_BRANCH
    global HLF_K8S_BRANCH
    global SUBSTRA_CHAINCODE_BRANCH
    global KANIKO_CACHE_TTL
    global BACKEND_CELERY_CONCURRENCY
    global TESTS_CONCURRENCY
    global TESTS_FUTURE_TIMEOUT

    parser = argparse.ArgumentParser()
    parser.add_argument('--machine-type', type=str, default=CLUSTER_MACHINE_TYPE,
                        help='The GKE machine type to use')
    parser.add_argument('-N', '--cluster-name', type=cluster_name, default=CLUSTER_NAME_ALLOWED_PREFIX,
                        help='The prefix name if the GKE kubernetes cluster to create')
    parser.add_argument('-K', '--keys-directory', type=str, default=KEYS_DIR,
                        help='The path to a folder containing the GKE service account credentials')
    parser.add_argument('--substra-tests', type=str, default=SUBSTRA_TESTS_BRANCH,
                        help='substra-tests branch', metavar='GIT_BRANCH')
    parser.add_argument('--substra', type=str, default=SUBSTRA_BRANCH,
                        help='substra branch', metavar='GIT_REF')
    parser.add_argument('--substra-backend', type=str, default=SUBSTRA_BACKEND_BRANCH,
                        help='substra-backend branch', metavar='GIT_BRANCH')
    parser.add_argument('--substra-chaincode', type=str, default=SUBSTRA_CHAINCODE_BRANCH,
                        help='substra-chaincode branch', metavar='GIT_BRANCH')
    parser.add_argument('--hlf-k8s', type=str, default=HLF_K8S_BRANCH,
                        help='hlf-k8s branch', metavar='GIT_BRANCH')
    parser.add_argument('--no-cache', action='store_true',
                        help='Use this option to disable kaniko caching')
    parser.add_argument('--backend-celery-concurrency', type=int, default=BACKEND_CELERY_CONCURRENCY,
                        help='The substra-backend worker task concurrency')
    parser.add_argument('--tests-concurrency', type=int, default=TESTS_CONCURRENCY,
                        help='The number of parallel test runners')
    parser.add_argument('--tests-future-timeout', type=int, default=TESTS_FUTURE_TIMEOUT,
                        help='In substra-tests, the number of seconds to wait for a training task to complete')

    args = vars(parser.parse_args())

    CLUSTER_NAME = args['cluster_name']
    # Add RUN_TAG to cluster name to make it non-deterministic in case of retry
    CLUSTER_NAME += f'-{RUN_TAG[:40-len(CLUSTER_NAME)-1]}'
    CLUSTER_NAME = CLUSTER_NAME.lower()   # Make it lower for gcloud compatibility
    PVC_VOLUME_NAME_PREFIX = args['cluster_name'] + f'-{RUN_TAG[:4]}'

    CLUSTER_MACHINE_TYPE = args['machine_type']
    KEYS_DIR = args['keys_directory']
    SUBSTRA_TESTS_BRANCH = args['substra_tests']
    SUBSTRA_BRANCH = args['substra']
    SUBSTRA_BACKEND_BRANCH = args['substra_backend']
    SUBSTRA_CHAINCODE_BRANCH = args['substra_chaincode']
    HLF_K8S_BRANCH = args['hlf_k8s']
    BACKEND_CELERY_CONCURRENCY = args['backend_celery_concurrency']
    TESTS_CONCURRENCY = args['tests_concurrency']
    TESTS_FUTURE_TIMEOUT = args['tests_future_timeout']
    if args['no_cache']:
        KANIKO_CACHE_TTL = '-1h'

    print('💃💃💃\n')
    print_args()


def print_args():
    print(
        f'CLUSTER_MACHINE_TYPE\t\t= {CLUSTER_MACHINE_TYPE}\n'
        f'CLUSTER_NAME\t\t\t= {CLUSTER_NAME}\n'
        f'KEYS_DIR\t\t\t= {KEYS_DIR}\n'
        f'SUBSTRA_TESTS_BRANCH\t\t= {SUBSTRA_TESTS_BRANCH}\n'
        f'SUBSTRA_BRANCH\t\t\t= {SUBSTRA_BRANCH}\n'
        f'SUBSTRA_BACKEND_BRANCH\t\t= {SUBSTRA_BACKEND_BRANCH}\n'
        f'SUBSTRA_CHAINCODE_BRANCH\t= {SUBSTRA_CHAINCODE_BRANCH}\n'
        f'HLF_K8S_BRANCH\t\t\t= {HLF_K8S_BRANCH}\n'
        f'KANIKO_CACHE_TTL\t\t= {KANIKO_CACHE_TTL}\n'
        f'BACKEND_CELERY_CONCURRENCY\t= {BACKEND_CELERY_CONCURRENCY}\n'
        f'TESTS_CONCURRENCY\t\t= {TESTS_CONCURRENCY}\n'
        f'TESTS_FUTURE_TIMEOUT\t\t= {TESTS_FUTURE_TIMEOUT}\n'
    )


def gcloud_login():
    print('# Log into Google Cloud')
    call(f'gcloud auth activate-service-account {SERVICE_ACCOUNT} --key-file={KEYS_DIR}/{KEY_SERVICE_ACCOUNT}')


def get_kube_context():
    global KUBE_CONTEXT

    old_ctx = None
    print('\n# Fetch kubernetes context')

    try:
        if call_output('kubectl config get-contexts --no-headers'):
            old_ctx = call_output('kubectl config current-context')
    except Exception:
        pass

    call(f'gcloud container clusters get-credentials {CLUSTER_NAME} --zone {CLUSTER_ZONE} --project {CLUSTER_PROJECT}')

    if old_ctx is not None:
        call(f'kubectl config use-context {old_ctx}')  # Restore old context

    KUBE_CONTEXT = f'gke_{CLUSTER_PROJECT}_{CLUSTER_ZONE}_{CLUSTER_NAME}'


def create_cluster_async():
    print('\n# Create GKE cluster')
    cmd = f'gcloud container clusters create {CLUSTER_NAME} '\
          f'--machine-type {CLUSTER_MACHINE_TYPE} '\
          f'--service-account {SERVICE_ACCOUNT} '\
          f'--num-nodes=1 '\
          f'--zone={CLUSTER_ZONE} '\
          f'--project={CLUSTER_PROJECT} '\
          f'--enable-ip-alias '\
          f'--no-enable-autoupgrade '\
          f'--enable-network-policy '\
          f'--async'
    call(cmd)


def delete_cluster():
    wait_for_cluster()
    print('# Delete cluster')
    cmd = f'yes | gcloud container clusters delete {CLUSTER_NAME} --zone ' \
          f'{CLUSTER_ZONE} --project {CLUSTER_PROJECT} --quiet'
    call(cmd)


def delete_disks():
    try:
        filter = f'name~^gke-{PVC_VOLUME_NAME_PREFIX}-pvc-.* AND -users:* AND -zones:({CLUSTER_ZONE})'
        cmd = f'gcloud compute disks list --project {CLUSTER_PROJECT} --format="table(name)" --filter="{filter}" '\
              '| sed 1d'
        disks = call_output(cmd)
        disks = disks.replace("\n", " ")
        if disks:
            call(f'gcloud compute disks delete --zone {CLUSTER_ZONE} --project {CLUSTER_PROJECT} --quiet {disks}')
    except Exception as ex:
        print('ERROR: Deletion of the GCP disks failed', ex)


def wait_for_cluster():
    print('# Waiting for GKE cluster to be ready ...', end='')

    while True:
        output = call_output(
            f'gcloud container clusters list --filter="name={CLUSTER_NAME}" --project {CLUSTER_PROJECT}',
            print_cmd=False
        )

        try:
            status = output.split('\n')[1].split(' ')[-1]
            if status not in ['RUNNING', 'PROVISIONING']:
                raise Exception(f'Unknown status {status}')
        except Exception as e:
            print('\nFATAL: Error retrieving cluster status. Output was:')
            print(output)
            raise(e)

        if status == 'RUNNING':
            print('done.')
            break

        print('.', end='', flush=True)
        time.sleep(5)


def setup_helm():
    print('\n# Setup Helm')
    call('helm repo add stable https://charts.helm.sh/stable')
    call('helm repo add owkin https://owkin.github.io/charts/')
    call('helm repo add bitnami https://charts.bitnami.com/bitnami')


def clone_repos():

    global CHAINCODE_COMMIT

    if os.path.exists(SOURCE_DIR):
        shutil.rmtree(SOURCE_DIR)

    os.makedirs(SOURCE_DIR)

    print(f'\n# Clone repos in {SOURCE_DIR}')
    commit_backend = clone_substra_backend()
    CHAINCODE_COMMIT = clone_substra_chaincode()
    commit_hlf = clone_hlf_k8s()
    commit_substra_tests = clone_substra_tests()
    commit_substra = get_remote_commit('https://github.com/SubstraFoundation/substra.git', SUBSTRA_BRANCH)

    print(
        f'\nCommit hashes:\n'
        f'- substra-backend: \t{commit_backend}\n'
        f'- substra-chaincode: \t{CHAINCODE_COMMIT}\n'
        f'- hlf-k8s: \t\t{commit_hlf}\n'
        f'- substra-tests: \t{commit_substra_tests}\n'
        f'- substra: \t\t{commit_substra}\n'
    )
    return [
        {'name': 'hlf-k8s',
         'images': ['fabric-tools', 'fabric-ca-tools', 'fabric-peer'],
         'commit': commit_hlf,
         'branch': HLF_K8S_BRANCH},
        {'name': 'substra-backend',
         'images': ['substra-backend'],
         'commit': commit_backend,
         'branch': SUBSTRA_BACKEND_BRANCH},
        {'name': 'substra-chaincode',
         'images': ['substra-chaincode'],
         'commit': CHAINCODE_COMMIT,
         'branch': SUBSTRA_CHAINCODE_BRANCH},
        {'name': 'substra-tests',
         'images': ['substra-tests'],
         'commit': commit_substra_tests,
         'branch': SUBSTRA_TESTS_BRANCH,
         'substra_commit': commit_substra}
    ]


def get_remote_commit(url, branch):
    commit = call_output(f'git ls-remote --refs {url} {branch}')
    commits = commit.split('\t')
    if len(commits) != 2:
        print(f'FATAL: On the repository {url}, the branch does not match one and only one commit: {commits}')
        raise Exception('Unable to get the right commit for the repository.')
    return commits[0]


def clone_repository(dirname, url, branch, commit=None):
    call(f'git clone -q --depth 1 {url} --branch "{branch}" {dirname}')

    if commit is None:
        commit = call_output(f'git --git-dir={dirname}/.git rev-parse origin/{branch}')

    return commit


def clone_substra_backend():
    url = 'https://github.com/SubstraFoundation/substra-backend.git'
    return clone_repository(
        dirname=os.path.join(SOURCE_DIR, 'substra-backend'),
        url=url,
        branch=SUBSTRA_BACKEND_BRANCH
    )


def clone_substra_chaincode():
    url = 'https://github.com/SubstraFoundation/substra-chaincode.git'
    return clone_repository(
        dirname=os.path.join(SOURCE_DIR, 'substra-chaincode'),
        url=url,
        branch=SUBSTRA_CHAINCODE_BRANCH
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
                config=config
            )
            images[build_id] = image

    wait_for_builds(tag, images)


def build_image(tag, image, config):
    branch = config["branch"]
    commit = config["commit"]
    name = config["name"]
    config_file = os.path.join(DIR, f'cloudbuild/{name}.yaml')

    extra_substitutions = ''
    if name == 'substra-tests':
        extra_substitutions = f',_SUBSTRA_GIT_COMMIT={config["substra_commit"]}'

    cmd = f'gcloud builds submit '\
        f'--config={config_file} '\
        f'--no-source '\
        f'--async '\
        f'--project={CLUSTER_PROJECT} '\
        f'--substitutions=_BUILD_TAG={tag},_IMAGE={image},_BRANCH={branch},_COMMIT={commit},'\
        f'_KANIKO_CACHE_TTL={KANIKO_CACHE_TTL}{extra_substitutions}'

    output = call_output(cmd)
    print(output)

    build_id = output.split('\n')[-1].split(' ')[0]

    return build_id


def wait_for_builds(tag, images):
    print('\n# Waiting for builds to complete ...', end='')
    do_wait = True
    while do_wait:
        build_list = call_output(
            f'gcloud builds list --filter="tags={tag}" --project={CLUSTER_PROJECT}',
            print_cmd=False
        )

        builds = build_list.split('\n')[1:]

        num_builds = len(builds)
        num_success = build_list.count('SUCCESS')
        num_failed = build_list.count('TIMEOUT') + build_list.count('CANCELLED') + build_list.count('FAIL')

        do_wait = (num_builds != (num_success + num_failed))

        time.sleep(5)
        print('.', end='', flush=True)

    print('done.')

    if num_failed:
        print('FATAL: One or more builds failed. See logs for more details')
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
        # Chaincode does not need to be deployed
        if config['name'] == 'substra-chaincode':
            continue

        wait = config['name'] != 'hlf-k8s'  # don't wait for hlf-k8s deployment to complete
        deploy(config, wait)


def deploy(config, wait=True):
    artifacts_file = create_build_artifacts(config)
    skaffold_file = patch_skaffold_file(config)

    path = os.path.dirname(skaffold_file)

    if config['name'] == 'hlf-k8s':
        call(f'KUBE_CONTEXT={KUBE_CONTEXT} {path}/examples/dev-secrets.sh create')

    call(f'cd {path} && skaffold deploy --kube-context={KUBE_CONTEXT} '
         f'-f=skaffold.yaml -a={artifacts_file} --status-check={"true" if wait else "false"}')


def create_build_artifacts(config):
    # Gcloud Build artifacts
    artifacts_file = os.path.join(SOURCE_DIR, config['name'], 'tags.json')

    with open(artifacts_file, 'w') as file:
        tags = {'builds': []}

        for image in config['images']:

            tag = f'eu.gcr.io/{CLUSTER_PROJECT}/{image}:ci-{config["commit"]}'

            if image == 'substra-tests':
                tag += f'-{config["substra_commit"]}'

            tags['builds'].append({
                'imageName': f'substrafoundation/{image}',
                'tag': tag
            })

            print(f'Created build artifact for {tag}')

        json.dump(tags, file)

    return artifacts_file


def patch_skaffold_file(config):

    skaffold_file = os.path.join(SOURCE_DIR, config['name'], 'skaffold.yaml')

    with open(skaffold_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    values_files = []

    for r in data['deploy']['helm']['releases']:
        if r['chartPath'].startswith('charts/'):
            r['chartPath'] = os.path.join(SOURCE_DIR, config["name"], r['chartPath'])
        if 'valuesFiles' in r:
            values_files.extend(r['valuesFiles'])

    with open(skaffold_file, 'w') as file:
        yaml.dump(data, file)

    for values_file in values_files:
        patch_values_file(config, os.path.join(SOURCE_DIR, config['name'], values_file))
    return skaffold_file


def patch_values_file(config, value_file):
    with open(value_file) as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    if config['name'] == 'substra-backend':
        data['celeryworker']['concurrency'] = BACKEND_CELERY_CONCURRENCY
    if config['name'] == 'hlf-k8s':
        if 'appChannels' in data:
            for i in range(len(data['appChannels'])):
                if 'chaincodes' in data['appChannels'][i]:
                    data['appChannels'][i]['chaincodes'][0]['image']['repository'] = \
                        f'eu.gcr.io/{CLUSTER_PROJECT}/substra-chaincode'
                    data['appChannels'][i]['chaincodes'][0]['image']['tag'] = f'ci-{CHAINCODE_COMMIT}'

    with open(value_file, 'w') as file:
        yaml.dump(data, file)


def run_tests():
    print('# Wait for the substra-tests pod to be ready')
    substra_tests_pod = call_output(
        f'kubectl --context {KUBE_CONTEXT} get pods -n substra-tests | grep substra-tests'
    ).split(' ')[0]

    try:
        call(f'kubectl --context {KUBE_CONTEXT} wait pod/{substra_tests_pod} '
             f'-n substra-tests --for=condition=ready --timeout=590s')
    except Exception:
        print('ERROR: Timeout while waiting for the substra-tests pod. '
              'This means the `substra-backend-server` pods never reached the "ready" state.')

    print('\n# Run tests')

    try:
        # Run the tests on the remote and local backend
        call(f'kubectl --context {KUBE_CONTEXT} exec {substra_tests_pod} -n substra-tests -- '
             f'env SUBSTRA_TESTS_FUTURE_TIMEOUT={TESTS_FUTURE_TIMEOUT} '
             f'make test-remote PARALLELISM={TESTS_CONCURRENCY}')
        return True
    except subprocess.CalledProcessError:
        print('FATAL: `make test-remote` completed with a non-zero exit code. Did some test(s) fail?')
        return False


def main():
    is_success = False
    arg_parse()

    try:
        gcloud_login()
        create_cluster_async()
        configs = clone_repos()
        build_images(configs)
        wait_for_cluster()
        get_kube_context()
        setup_helm()
        deploy_all(configs)
        is_success = run_tests()
        print("Completed test run:")
        print_args()

    except Exception as e:
        print(f'FATAL: {e}')
        is_success = False

    finally:
        print('\n# Perform final teardown')
        if os.path.exists(SOURCE_DIR):
            shutil.rmtree(SOURCE_DIR)
        delete_cluster()
        delete_disks()

    sys.exit(0 if is_success else 1)


if __name__ == '__main__':
    main()
