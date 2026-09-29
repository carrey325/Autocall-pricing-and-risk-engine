"""Audit wheel/sdist contents, then install each into a fresh external venv."""

import argparse
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import venv
import zipfile


def audit(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            names = archive.getnames()
    for name in names:
        parts = PurePosixPath(name).parts
        assert not any(p in {"old", ".git", "__pycache__", ".pytest_tmp", ".pytest_cache"}
                       or p.startswith(".venv") for p in parts), name
        assert not name.endswith((".ipynb", ".pyc", ".pyo")), name
    assert sum("structured_products/data/" in n and n.endswith(".json") for n in names) == 6
    assert any(n.endswith("structured_products/py.typed") for n in names)
    assert any(n.endswith("LICENSE") for n in names)
    print(f"PASS contents: {path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheels = list(args.dist.glob("*.whl"))
    sources = list(args.dist.glob("*.tar.gz"))
    assert len(wheels) == len(sources) == 1, "Use a directory with exactly one wheel and one sdist"
    smoke = Path(__file__).with_name("installed_smoke.py").resolve()
    for artifact in wheels + sources:
        artifact = artifact.resolve()
        audit(artifact)
        with tempfile.TemporaryDirectory(prefix="autocall-installed-") as temporary:
            root = Path(temporary)
            environment = root / "env"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            subprocess.run(
                [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--quiet", str(artifact)],
                cwd=root, check=True,
            )
            subprocess.run([str(python), "-I", str(smoke)], cwd=root, check=True)
            subprocess.run([str(python), "-m", "pip", "check"], cwd=root, check=True)


if __name__ == "__main__":
    main()
