import os
import glob

try:
    from ruamel import yaml
except ImportError:
    import yaml


dir_path = os.path.dirname(os.path.realpath(__file__))

skaffold_files = glob.glob(os.path.join(dir_path,
                                        './src/*/skaffold.yaml'))

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

        ## Disable cache: https://github.com/GoogleContainerTools/kaniko/issues/1039#issuecomment-588015578
        ## Uncomment to enable caching
        # artifact['kaniko']['cache'] = {}

        # Use debug version
        # Bug 1 (apt-get): https://github.com/GoogleContainerTools/kaniko/issues/793#issuecomment-582989625
        # Bug 2 (failed to get filesystem): https://github.com/GoogleContainerTools/kaniko/issues/1039#issuecomment-590974549
        artifact['kaniko']['image'] = 'gcr.io/kaniko-project/executor:debug-a1af057f997316bfb1c4d2d82719d78481a02a79'

    with open(skaffold_file, 'w') as f:
        f.write(yaml.dump(skaffold,
                          default_flow_style=False,
                          indent=4,
                          line_break=None))
