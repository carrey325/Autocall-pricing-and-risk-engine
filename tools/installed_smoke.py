"""Exercise installed artifacts outside the checkout, with no test dependencies."""

from importlib.metadata import version
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys

import structured_products as sp


def main():
    assert Path(sys.prefix).resolve() in Path(sp.__file__).resolve().parents
    assert sp.__version__ == version("structured-products-pricer")
    assert files("structured_products").joinpath("py.typed").is_file()
    names = sp.list_examples()
    assert len(names) == 6
    for name in names:
        config = sp.load_example(name)
        engine = sp.MCConfig(n_paths=200, seed=7)
        result = sp.price(config.product, config.market, "mc", engine, include_greeks=True)
        assert result.present_value > 0
        assert result.greeks.base_price == result.present_value
        risk = result.to_dict()["greeks"]
        assert risk["lambda"] == risk["omega"]
        assert risk["vomma"] == risk["volga"]
        assert len(risk["vanna"]) == config.product.n_assets
        json.dumps(result.to_dict(), allow_nan=False)
        if config.product.n_assets == 1:
            assert sp.price(config.product, config.market, "pde", sp.PDEConfig(),
                            include_greeks=True).present_value > 0
    config = sp.load_example("classic_snowball")
    target = sp.price(config.product, config.market, "mc", engine).present_value
    solved = sp.solve_fair_coupon(config.product, config.market, target, "mc", engine, 0.0, 0.5)
    assert abs(solved.residual) < 1e-5
    console = Path(sys.executable).parent / (
        "structured-products-pricer.exe" if os.name == "nt" else "structured-products-pricer"
    )
    actual_version = subprocess.check_output([str(console), "--version"], text=True).strip()
    assert actual_version == sp.__version__
    for command in ([str(console)], [sys.executable, "-I", "-m", "structured_products"]):
        output = subprocess.check_output(
            [*command, "price", "--example", "classic_snowball", "--paths", "100"], text=True,
        )
        assert "veta" in json.loads(output)["greeks"]
    print(f"PASS installed {sp.__version__}: six presets, MC/PDE, risk, fair coupon, console/module CLI")


if __name__ == "__main__":
    main()
