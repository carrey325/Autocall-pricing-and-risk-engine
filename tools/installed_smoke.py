"""Exercise installed artifacts outside the checkout, with no test dependencies."""

from importlib.metadata import version
from importlib.resources import files
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from dataclasses import replace

import structured_products as sp


def main():
    assert Path(sys.prefix).resolve() in Path(sp.__file__).resolve().parents
    assert sp.__version__ == version("structured-products-pricer")
    assert files("structured_products").joinpath("py.typed").is_file()
    names = sp.list_examples()
    assert len(names) == 6
    inputs = sp.load_market_data(files("structured_products").joinpath("data", "market_surface"))
    quotes = sp.validate_quotes(inputs.rows, inputs.snapshot, inputs.calendar, inputs.curves)
    fit = sp.calibrate_ssvi(quotes)
    assert fit.accepted and len(quotes.accepted) == 90
    assert fit.overall["bid_ask_coverage"] == 1
    json.dumps(fit.to_dict(), allow_nan=False)
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
    from structured_products.research import default_analysis
    assert default_analysis()['pde']['spot_grid_points']==201
    model=sp.LocalVolModel.from_calibration(fit)
    note=sp.make_classic_snowball(maturity_days=2,observation_days=(2,))
    node=sp.transform_scenario(note,config.market,model,sp.ScenarioShock(spot=90.,roll_days=1))
    assert sp.Scenario.from_dict(node.to_dict()).model.identity==node.model.identity
    assert sp.price_scenario(node,'pde',sp.PDEConfig(101)).present_value>0
    stress=sp.stress_test(note,config.market,model,scenarios=(sp.StressSpec('zero'),),pde_config=sp.PDEConfig(101))
    assert stress.cells[0]['holder_loss']==0
    terms=sp.compare_terms(note,config.market,model,coupons=(.1,),ki_ratios=(.7,),ko_ratios=(1.,),
        scenarios=(),pde_config=sp.PDEConfig(101),fair_bracket=(0.,2.))
    assert terms.cells[0]['base_pv']>0
    console = Path(sys.executable).parent / (
        "structured-products-pricer.exe" if os.name == "nt" else "structured-products-pricer"
    )
    actual_version = subprocess.check_output([str(console), "--version"], text=True).strip()
    assert actual_version == sp.__version__
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'model.json'
        path.write_text(json.dumps(sp.LocalVolModel.theoretical(model.surface,model.curves).to_dict()),encoding='utf-8')
        output=subprocess.check_output([str(console),'price','--example','classic_snowball',
            '--model',str(path),'--paths','100','--no-greeks'],text=True)
        assert json.loads(output)['price']['present_value']>0
    for command in ([str(console)], [sys.executable, "-I", "-m", "structured_products"]):
        output = subprocess.check_output(
            [*command, "price", "--example", "classic_snowball", "--paths", "100"], text=True,
        )
        assert "veta" in json.loads(output)["greeks"]
    print(f"PASS installed {sp.__version__}: six presets, MC/PDE, risk, fair coupon, console/module CLI, SSVI resources/calibration, model/scenario/stress/terms/research config; no plot dependency")


if __name__ == "__main__":
    main()
