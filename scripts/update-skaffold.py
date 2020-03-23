import os
import glob

try:
    from ruamel import yaml
except ImportError:
    import yaml


dir_path = os.path.dirname(os.path.realpath(__file__))

skaffold_files = glob.glob(os.path.join(dir_path,
                                        'substra-resources/*/skaffold.yaml'))

print(skaffold_files)

for skaffold_file in skaffold_files:
    with open(skaffold_file) as f:
        skaffold = yaml.load(f)

    skaffold['build']['cluster'] = {
        'pullSecretName': 'kaniko-secret'
    }

    artifatcs = []

    for artifact in skaffold['build']['artifacts']:

        artifact['kaniko'] = {}

        if 'docker' in artifact:
            artifact['kaniko'] = artifact['docker']
            del artifact['docker']

        artifact['kaniko']['cache'] = {}

    with open(skaffold_file, 'w') as f:
        f.write(yaml.dump(skaffold,
                          default_flow_style=False,
                          indent=4,
                          line_break=None))
