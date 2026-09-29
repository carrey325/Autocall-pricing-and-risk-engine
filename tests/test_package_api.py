"""Public package behavior, independent of the caller's working directory."""

from importlib.metadata import version
from importlib.resources import files
import json
import re
from pathlib import Path
import subprocess
import sys

import pytest

import structured_products as sp


def test_bundled_examples_work_outside_repository(tmp_path, monkeypatch):
    original = Path.cwd()
    monkeypatch.chdir(tmp_path)
    names = sp.list_examples()
    assert len(names) == 6
    for name in names:
        config = sp.load_example(name)
        reference = sp.load_config(original / "configs" / f"{name}.json")
        assert config == reference
        result = sp.price(config.product, config.market, "mc", sp.MCConfig(n_paths=100),
                          include_greeks=True)
        assert result.greeks.base_price == result.present_value
        json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("name", ["missing", "../classic_snowball", "classic_snowball.json"])
def test_unknown_example_has_actionable_error(name):
    with pytest.raises(ValueError, match="available examples"):
        sp.load_example(name)


def test_mapping_config_uses_same_validation_as_json_file():
    payload = {"product": {"template": "classic_snowball"},
               "market": {"spots": [100], "rate": 0.03, "volatilities": [0.2],
                          "dividend_yields": [0]}}
    config = sp.config_from_dict(payload)
    assert config.product.n_assets == 1
    payload["unknown"] = True
    with pytest.raises(ValueError, match="top-level"):
        sp.config_from_dict(payload)


def test_module_cli_works_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "structured_products", "price", "--example",
         "classic_snowball", "--paths", "100"],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert json.loads(result.stdout)["greeks"]["vanna"]


def test_version_matches_distribution_and_typing_marker_is_shipped():
    assert sp.__version__ == version("structured-products-pricer")
    assert files("structured_products").joinpath("py.typed").is_file()


@pytest.mark.parametrize("index", range(4))
def test_documented_tutorials_run_outside_checkout(index, tmp_path):
    document = Path(__file__).resolve().parents[1] / "docs" / "TUTORIALS.md"
    examples = re.findall(r"```python\n(.*?)```", document.read_text(encoding="utf-8"), re.S)
    assert len(examples) == 4
    subprocess.run([sys.executable, "-c", examples[index]], cwd=tmp_path,
                   text=True, capture_output=True, check=True)
