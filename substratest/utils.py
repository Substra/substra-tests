import os
import re
import zipfile


def zip_folder(path, destination=None):
    if not destination:
        destination = os.path.join(os.path.dirname(path),
                                   os.path.basename(path) + '.zip')
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(path):
            for f in files:
                abspath = os.path.join(root, f)
                archive_path = os.path.relpath(abspath, start=path)
                zf.write(abspath, arcname=archive_path)
    return destination


def create_archive(tmpdir, *files):
    tmpdir.mkdir()
    for path, content in files:
        with open(tmpdir / path, 'w') as f:
            f.write(content)
    return zip_folder(str(tmpdir))


def camel_to_snake(name):
    """Convert camel case to snake case."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def replace_dict_keys(d, converter):
    """Replace fields in a dict and return updated dict (recursive).

    Apply converter to each dict field.
    """
    assert isinstance(d, dict)
    new_d = {}
    for key, value in d.items():
        if isinstance(value, dict):
            value = replace_dict_keys(value, converter)
        elif isinstance(value, list):
            if all([isinstance(v, dict) for v in value]):
                value = [replace_dict_keys(v, converter) for v in value]

        new_d[converter(key)] = value
    return new_d
