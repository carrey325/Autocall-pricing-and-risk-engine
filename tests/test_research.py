from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pytest
import structured_products as sp


def test_research_actual_artifacts_hashes_and_resume(tmp_path,monkeypatch):
    from structured_products.research import run_research
    phase={'started':False}
    def git(args,**kwargs):
        return ('later' if phase['started'] else 'start') if args[1]=='rev-parse' else ''
    monkeypatch.setattr('structured_products.research.subprocess.check_output',git)
    monkeypatch.setattr('structured_products.analytics.stress.source_hashes',lambda:{'changed_on_disk.py':'later'})
    monkeypatch.setattr('structured_products.analytics.terms.source_hashes',lambda:{'changed_on_disk.py':'later'})
    def progress(*args):phase['started']=True
    p=sp.make_classic_snowball(maturity_days=2,observation_days=(2,))
    c=sp.load_example('classic_snowball');c=replace(c,product=p,pde=sp.PDEConfig(101),mc=sp.MCConfig(n_paths=100,seed=7))
    plan={'title':'<script>alert("x")</script>','risk':dict(spot_ratios=[1.],rho_shifts=[0.],roll_days=[0,2],initial_ki=[False]),
          'terms':dict(coupons=[.1],ki_ratios=[.7],ko_ratios=[1.]),'scenarios':[{'name':'zero'}],
          'validation':False,'pde':dict(spot_grid_points=101,spot_grid_upper=450.),'mc':dict(n_paths=100,seed=9)}
    result=run_research(tmp_path,config=c,analysis=plan,progress=progress)
    assert result['code_revision']=='start'
    assert result['status']=='synthetic_completed; real_market_pending'
    assert result['config']['engines']['pde']['spot_grid_upper']==450.
    assert result['config']['engines']['mc']['seed']==9
    from structured_products.analytics.stress import content_hash
    expected_code=content_hash(result['source_hashes'])
    assert json.loads((tmp_path/'stress.json').read_text())['cells'][0]['code_hash']==expected_code
    term=json.loads((tmp_path/'terms.json').read_text())['cells'][0]
    assert term['code_hash']==expected_code and term['stress']['cells'][0]['code_hash']==expected_code
    html=(tmp_path/'report.html').read_text(encoding='utf-8')
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert 'Bump and resolution limits' in html
    for name in ('risk.json','risk.csv','terms.json','terms.csv','stress.json','stress.csv','overview.png','overview.svg','calibration.json',
                 'calibration-fit.png','local-vol.png','risk-metrics.png','stress-residuals.png','terms-fair-coupon.png'):
        assert (tmp_path/name).is_file()
        assert result['artifacts'][name]==hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
    assert 'src="overview.png"' in html
    assert len(json.loads((tmp_path/'risk.json').read_text())['cells'])==2
    before=(tmp_path/'risk.json').read_bytes()
    again=run_research(tmp_path,config=c,analysis=plan)
    assert (tmp_path/'risk.json').read_bytes()==before
    assert again['input_hash']==result['input_hash']
    from structured_products.research import refresh_report
    rendered=refresh_report(tmp_path)
    assert rendered['source_hashes']==result['source_hashes']
    assert rendered['report_render']['code_revision']=='later'
    payload=json.loads((tmp_path/'stress.json').read_text())
    payload['cells'][0]['holder_pnl']=1234567
    (tmp_path/'stress.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='checkpoint.*hash'):
        run_research(tmp_path,config=c,analysis=plan)
    with pytest.raises(ValueError,match='checkpoint'):
        run_research(tmp_path,config=c,analysis={**plan,'risk':dict(spot_ratios=[.9])})


def test_validation_records_independent_refinements(tmp_path):
    from structured_products.research import numerical_validation
    c=sp.load_example('classic_snowball')
    curves=sp.MarketCurves((0.,1.),(100.,101.5),(1.,.97))
    model=sp.LocalVolModel.theoretical(sp.SSVISurface((.25,1.),(.01,.04),-.4,.4),curves)
    rows=numerical_validation(c,model,quick=True)
    assert {'vanilla','product','risk_checkpoint','flat_control'}<=set(r['kind'] for r in rows)
    product=[r for r in rows if r['kind']=='product']
    assert len(product)==6
    for row in product:
        assert {'pde_space_change','pde_time_change','pde_domain_change','mc_substep_change','sampling_allowance','passed'}<=row.keys()
        assert row['acceptance_allowance']==pytest.approx(row['sampling_allowance']+row['discretization_allowance'])
    json.dumps(rows,allow_nan=False)


def test_rejected_report_calibration_diagnostics_only(tmp_path):
    from structured_products.research import render_report
    from importlib.resources import files
    d=sp.load_market_data(files('structured_products').joinpath('data/market_surface'))
    fit=sp.calibrate_ssvi(sp.validate_quotes(d.rows,d.snapshot,d.calendar,d.curves),sp.CalibrationConfig(max_price_rmse=1e-18))
    assert not fit.accepted
    render_report(tmp_path,'Real rejection',{'cells':[]},{'cells':[]},{'cells':[]},[],status='rejected_quality; pricing_pending',calibration=fit.to_dict())
    html=(tmp_path/'report.html').read_text(encoding='utf-8')
    assert 'No pricing performed' in html
    assert 'price_rmse' in html and '<table>' in html
    assert 'href="risk.json"' not in html
    assert (tmp_path/'calibration-fit.png').is_file()
    assert not (tmp_path/'local-vol.png').exists()


@pytest.mark.filterwarnings('error:No artists with labels')
def test_research_cli_precision_and_rejected_external_diagnostic(tmp_path,capsys):
    from structured_products.cli import main
    from structured_products.research import run_research
    from importlib.resources import files
    import shutil
    plan=dict(validation=False,pde=dict(spot_grid_points=101,spot_grid_upper=450.),mc=dict(n_paths=100,seed=11),
        risk=dict(spot_ratios=[1.],rho_shifts=[0.],roll_days=[0],initial_ki=[False]),
        terms=dict(coupons=[.1],ki_ratios=[.7],ko_ratios=[1.]),scenarios=[dict(name='zero')])
    analysis=tmp_path/'analysis.json';analysis.write_text(json.dumps(plan))
    config=tmp_path/'product.json'
    config.write_text(json.dumps(dict(product=dict(template='classic_snowball',parameters=dict(maturity_days=2,observation_days=[2])),
        market=dict(spots=[100.],rate=.03,dividend_yields=[0.],volatilities=[.2]))))
    assert main(['research','--config',str(config),'--analysis-config',str(analysis),'--output',str(tmp_path/'cli')])==0
    manifest=json.loads((tmp_path/'cli/manifest.json').read_text())
    assert manifest['config']['engines']['mc']['seed']==11
    capsys.readouterr()
    canonical=tmp_path/'external';canonical.mkdir()
    source=files('structured_products').joinpath('data/market_surface')
    for name in ('snapshot.json','quotes.csv','calendar.csv','curves.csv'):shutil.copy2(source.joinpath(name),canonical/name)
    snap=json.loads((canonical/'snapshot.json').read_text());snap['provenance']['kind']='external test fixture'
    (canonical/'snapshot.json').write_text(json.dumps(snap))
    rows=(canonical/'quotes.csv').read_text().splitlines()
    (canonical/'quotes.csv').write_text('\n'.join(rows[:2])+'\n')
    result=run_research(tmp_path/'rejected',analysis=plan,market_input=canonical)
    assert result['status']=='rejected_quality; pricing_pending'
    assert result['summary']['pricing_performed'] is False
    assert json.loads((tmp_path/'rejected/risk.json').read_text())['cells']==[]
