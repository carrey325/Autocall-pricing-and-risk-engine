"""Exercise the finite diagnostic command and its reproducibility record."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_numerical_diagnostics_outputs_finite_reproducible_cases(tmp_path):
    output = tmp_path / "diagnostics.json"
    root = Path(__file__).resolve().parents[1]
    process = subprocess.run(
        [sys.executable, str(root / "tools/numerical_diagnostics.py"), "--output", str(output)],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    json.dumps(report, allow_nan=False)
    assert report["metadata"]["code_head"]
    assert report["metadata"]["numpy"]
    assert [row["config"]["seed"] for row in report["mc"]] == [7, 17, 29]
    for row in report["mc"]:
        assert row["result"]["standard_error"] == pytest.approx(row["independent_pair_se"])
        assert len(row["sample_sha256"]) == 64
    assert len(report["analytic_checks"]) == 2
    assert all(row["passed"] for row in report["analytic_checks"])
    for name in ("space", "time", "domain"):
        rows = report["pde_sweeps"][name]
        assert len(rows) == 15
        assert {row["spot"] for row in rows} == {79.9, 80.1, 99.9, 100.0, 100.1}
        for row in rows:
            config = row["config"]
            if name == "domain":
                assert config["spot_grid_upper"] / (config["spot_grid_points"] - 1) == .5
            else:
                assert config["spot_grid_upper"] == 400
            if name == "time":
                assert config["spot_grid_points"] == 1601
            else:
                assert config["time_steps_per_day"] == 2
    assert len(report["greek_bumps"]) == 6
    assert {row["greek_config"]["relative_spot_bump"] for row in report["greek_bumps"]} == {.005, .01, .02}
