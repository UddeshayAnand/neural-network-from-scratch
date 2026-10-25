"""Small, atomic dataset downloads using the Python standard library."""
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


def download_file(url, data_dir, force=False):
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / Path(urlsplit(url).path).name
    if not force and target.is_file() and target.stat().st_size:
        return str(target)
    temporary = None
    try:
        with urlopen(url, timeout=30) as response:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
                temporary = Path(output.name)
                shutil.copyfileobj(response, output)
            expected = response.headers.get("Content-Length")
            if not temporary.stat().st_size:
                raise OSError("Downloaded file is empty: " + url)
            if expected is not None and temporary.stat().st_size != int(expected):
                raise OSError("Incomplete download: " + url)
        os.replace(temporary, target)
        return str(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
