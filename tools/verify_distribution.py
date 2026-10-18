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
        assert not any(p in {"old", ".git", ".superpowers", "results", "__pycache__", ".pytest_tmp", ".pytest_research_check", ".pytest_cache"}
                       or p.startswith(".venv") for p in parts), name
        assert not name.endswith((".ipynb", ".pyc", ".pyo")), name
    # The six flat presets are top-level resources. Surface snapshots/templates
    # live below data/ and must not be counted as additional pricing presets.
    assert sum(PurePosixPath(n).parent.parts[-2:] == ("structured_products", "data")
               and n.endswith(".json") for n in names) == 6
    for resource in ("snapshot.json", "curves.csv", "calendar.csv", "quotes.csv", "research.json",
                     "generating_surface.json", "templates/snapshot.json", "templates/quotes.csv",
                     "templates/curves.csv", "templates/calendar.csv"):
        assert any(n.endswith("structured_products/data/market_surface/"+resource) for n in names), resource
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
    clean_environment=os.environ.copy()
    clean_environment.pop('PYTHONPATH',None)
    clean_environment.pop('PYTHONHOME',None)
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
                cwd=root, check=True, env=clean_environment,
            )
            subprocess.run([str(python), "-I", str(smoke)], cwd=root, check=True,env=clean_environment)
            subprocess.run([str(python), "-m", "pip", "check"], cwd=root, check=True,env=clean_environment)


if __name__ == "__main__":
    main()
