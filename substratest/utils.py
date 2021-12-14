import os
import zipfile


def zip_folder(path, destination=None):
    if not destination:
        destination = os.path.join(os.path.dirname(path), os.path.basename(path) + ".zip")
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(path):
            for f in files:
                abspath = os.path.join(root, f)
                archive_path = os.path.relpath(abspath, start=path)
                info = zipfile.ZipInfo.from_file(abspath, arcname=archive_path)
                info.date_time = (1980, 1, 1, 0, 0, 0)
                with open(abspath) as src:
                    zf.writestr(info, src.read())
    return destination


def create_archive(tmpdir, *files):
    tmpdir.mkdir()
    for path, content in files:
        with open(tmpdir / path, "w") as f:
            f.write(content)
    return zip_folder(str(tmpdir))
