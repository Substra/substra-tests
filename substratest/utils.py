import os
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
