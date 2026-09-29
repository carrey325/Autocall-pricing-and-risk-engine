from __future__ import annotations

import json
from pathlib import Path

import pytest

from structured_products.cli import main


@pytest.fixture
def short_config(tmp_path: Path) -> Path:
    payload = {
        "product": {
            "template": "classic_snowball",
            "parameters": {
                "reference_spot": 100.0,
                "notional": 1000.0,
                "maturity_days": 21,
                "observation_days": [21],
                "coupon_rate": 0.12,
            },
        },
        "market": {
            "spots": [100.0],
            "rate": 0.03,
            "dividend_yields": [0.0],
            "volatilities": [0.2],
        },
        "engines": {
            "mc": {"n_paths": 200, "seed": 7},
            "pde": {"spot_grid_points": 201},
        },
    }
    path = tmp_path / "short.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "arguments,required_key",
    [
        (["price", "--method", "mc", "--paths", "100", "--seed", "9"], "price"),
        (["greeks", "--method", "mc"], "greeks"),
        (["run", "--method", "mc"], "price"),
        (["compare", "--grid-size", "201"], "comparison"),
    ],
)
def test_cli_subcommands(
    short_config: Path,
    capsys,
    arguments: list[str],
    required_key: str,
) -> None:
    code = main([*arguments, "--config", str(short_config)])
    captured = capsys.readouterr()
    assert code == 0
    assert required_key in json.loads(captured.out)


def test_cli_fair_coupon_and_output_file(
    short_config: Path, tmp_path: Path, capsys
) -> None:
    output = tmp_path / "result.json"
    code = main(
        [
            "fair-coupon",
            "--config",
            str(short_config),
            "--method",
            "mc",
            "--target-pv",
            "1000",
            "--lower",
            "0",
            "--upper",
            "0.5",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "fair_coupon" in json.loads(output.read_text())


def test_invalid_config_returns_nonzero(short_config: Path, capsys) -> None:
    payload = json.loads(short_config.read_text())
    payload["unknown"] = True
    short_config.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["price", "--config", str(short_config)]) == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["price", "run"])
@pytest.mark.parametrize("method", ["mc", "pde"])
def test_pricing_includes_extended_risk_by_default(short_config, capsys, command, method):
    assert main([command, "--config", str(short_config), "--method", method]) == 0
    result = json.loads(capsys.readouterr().out)
    risk = result["greeks"]
    assert risk["base_price"] == result["price"]["present_value"]
    for name in ("lambda", "omega", "vanna", "vomma", "volga", "charm", "veta"):
        assert len(risk[name]) == 1


def test_price_only_opt_out_accepts_zero_volatility(short_config, capsys):
    payload = json.loads(short_config.read_text())
    payload["market"]["volatilities"] = [0.0]
    short_config.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["price", "--config", str(short_config), "--no-greeks"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert "greeks" not in result
    assert result["price"]["present_value"] > 0
