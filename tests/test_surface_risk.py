"""Surface risk tests use independently priced nodes and canonical cash events."""
from dataclasses import replace
from math import exp
import json

import numpy as np
import pytest

import structured_products as sp


def inputs(days=21):
    product = sp.make_classic_snowball(maturity_days=days, observation_days=(days,))
    market = sp.MarketData((100.,), .03, (.02,), (.2,))
    curves = sp.MarketCurves((0., .25, 1.), (100., 101., 103.), (1., .99, .95))
    model = sp.LocalVolModel.theoretical(
        sp.SSVISurface((.25, 1.), (.01, .06), -.4, .5), curves)
    return product, market, model


def test_zero_scenario_and_export_reconstruct_direct_price():
    product, market, model = inputs()
    node = sp.transform_scenario(product, market, model, sp.ScenarioShock())
    config = sp.PDEConfig(spot_grid_points=101)
    value = sp.price_scenario(node, 'pde', config)
    assert value.present_value == sp.price(product, market, 'pde', config, model=model).present_value
    restored = sp.Scenario.from_dict(json.loads(json.dumps(node.to_dict(), allow_nan=False)))
    assert restored.product == product
    assert sp.price_scenario(restored, 'pde', config).present_value == value.present_value
    assert restored.model.identity == model.identity


def test_spot_mark_freezes_history_and_roll_freezes_absolute_dynamics():
    product, market, model = inputs(63)
    mark = sp.transform_scenario(product, market, model, sp.ScenarioShock(spot=60.))
    assert mark.realized_knocked_in is False
    assert mark.product.reference_spots == (100.,)
    assert mark.model.curves.forwards == pytest.approx(np.array(model.curves.forwards)*.6)
    assert mark.model.local_vol(.1, 60.) == pytest.approx(model.local_vol(.1, 100.))
    roll = sp.transform_scenario(product, market, model, sp.ScenarioShock(roll_days=21))
    assert roll.model.local_vol(.02, 90.) == pytest.approx(model.local_vol(.02+21/252, 90.))
    assert roll.model.curves == model.curves
    assert roll.model.discount(.1) == pytest.approx(model.curves.discount(21/252+.1)/model.curves.discount(21/252))
    assert roll.product.elapsed_days == 21
    assert roll.product.autocall.barrier_ratios == product.autocall.barrier_ratios


@pytest.mark.parametrize('priority,ki', [(sp.SameDayPriority.KNOCK_OUT_FIRST, False), (sp.SameDayPriority.KNOCK_IN_FIRST, True)])
def test_same_day_roll_uses_payoff_priority_and_explicit_discounted_cash(priority, ki):
    product, market, model = inputs(3)
    product = replace(product, elapsed_days=10, same_day_priority=priority,
        knock_in=sp.KnockInFeature(1.), autocall=sp.AutocallFeature((1,3),(1.,1.),(.12,.12)),
        coupon=sp.CouponFeature(sp.CouponRule.FIXED_PERIODIC,(1,3),(.2,.2)))
    node = sp.transform_scenario(product, market, model, sp.ScenarioShock(roll_days=2))
    assert node.terminated and node.product is None
    assert node.realized_knocked_in is ki
    assert node.cash_sum == pytest.approx(1000+1000*.12*11/252)
    assert [c.kind for c in node.cashflows] == ['principal', 'autocall_coupon']
    assert all(c.day == 1 for c in node.cashflows)
    assert node.cash_pv == pytest.approx(node.cash_sum*model.discount(1/252))
    assert sp.price_scenario(node).present_value == 0.


def test_roll_coupon_accrual_delayed_ki_expiry_and_one_day_remaining():
    product, market, model = inputs(4)
    product = replace(product, autocall=None, elapsed_days=12, coupon_accrued_days=2,
        coupon=sp.CouponFeature(sp.CouponRule.FIXED_PERIODIC,(2,4),(.12,.12)),
        knock_in=sp.KnockInFeature(.7, monitoring_start_day=3))
    node = sp.transform_scenario(product, market, model, sp.ScenarioShock(spot=60., roll_days=2))
    assert not node.realized_knocked_in
    assert node.cash_sum == pytest.approx(1000*.12*4/252)
    assert node.product.coupon_accrued_days == 0
    assert node.product.knock_in.monitoring_start_day == 1
    assert node.product.reference_spots == product.reference_spots
    last = sp.transform_scenario(product, market, model, sp.ScenarioShock(spot=60., roll_days=3))
    assert last.realized_knocked_in and last.product.maturity_days == 1
    metrics = sp.scenario_risk(last, metrics=('present_value','delta','gamma'), pde_config=sp.PDEConfig(101))
    assert all(metrics['metric_status'][key]['status'] == 'ok' for key in metrics['metrics'])
    final = sp.transform_scenario(product, market, model, sp.ScenarioShock(spot=60., roll_days=4))
    assert final.terminated and final.cash_sum == pytest.approx(600+1000*.12*4/252)
    assert final.discount_to_roll == pytest.approx(model.discount(4/252))


def test_surface_metrics_equal_direct_centered_bumps_and_common_domain():
    product, market, model = inputs(3)
    config = sp.PDEConfig(101)
    surface = sp.risk_surface(product, market, model, spot_ratios=(.6,1.2),
        rho_shifts=(0.,), roll_days=(0,), initial_ki=(False,), pde_config=config)
    assert len(surface.cells) == 2
    assert surface.diagnostics['spot_grid_upper'] == pytest.approx(4*121.2)
    for cell in surface.cells:
        node = sp.Scenario.from_dict(cell['scenario'])
        common = replace(config, spot_grid_upper=surface.diagnostics['spot_grid_upper'])
        base = sp.price_scenario(node, 'pde', common).present_value
        bump = node.market.spots[0]*.01
        high = replace(node, market=replace(node.market,spots=(node.market.spots[0]+bump,)), model=node.model.rebase(node.market.spots[0]+bump))
        low = replace(node, market=replace(node.market,spots=(node.market.spots[0]-bump,)), model=node.model.rebase(node.market.spots[0]-bump))
        up = sp.price_scenario(high,'pde',common).present_value
        down = sp.price_scenario(low,'pde',common).present_value
        assert cell['metrics']['present_value'] == base
        assert cell['metrics']['delta'] == pytest.approx((up-down)/(2*bump))
        assert cell['metrics']['gamma'] == pytest.approx((up-2*base+down)/bump**2)
        assert cell['units']['atm_level_sensitivity'] == 'PV / ATM_level_multiplier'
    json.dumps(surface.to_dict(), allow_nan=False)


def test_invalid_surface_shocks_and_metric_only_failure_keep_price():
    product, market, model = inputs(2)
    model = sp.LocalVolModel.theoretical(replace(model.surface,rho=.995),model.curves)
    node = sp.transform_scenario(product,market,model,sp.ScenarioShock())
    risk = sp.scenario_risk(node,pde_config=sp.PDEConfig(101))
    assert risk['metrics']['present_value'] > 0
    assert risk['metrics']['rho_skew_sensitivity'] is None
    assert risk['metric_status']['rho_skew_sensitivity']['status'] == 'error'
    grid = sp.risk_surface(product,market,model,spot_ratios=(1.,),rho_shifts=(0.,.1),roll_days=(0,),initial_ki=(False,),metrics=('present_value',),pde_config=sp.PDEConfig(101))
    assert [c['status'] for c in grid.cells] == ['ok','error']
    with pytest.raises(ValueError):
        sp.transform_scenario(product,market,model,sp.ScenarioShock(level_multiplier=0.))
    with pytest.raises(ValueError):
        sp.transform_scenario(product,market,model,sp.ScenarioShock(spot=float('nan')))
    out = sp.risk_surface(product,market,model,spot_ratios=(1.2,),rho_shifts=(0.,),roll_days=(0,),initial_ki=(False,),pde_config=replace(sp.PDEConfig(101),spot_grid_upper=120.5))
    assert out.cells[0]['status'] == 'partial'
    assert out.cells[0]['metrics']['present_value'] > 0
    assert out.cells[0]['metrics']['delta'] is None


def test_default_axes_filter_expiry_add_barrier_neighborhood_and_mc_reproducibility():
    product,market,model=inputs(21)
    axes=sp.default_risk_axes(product)
    assert axes['roll_days'] == (0,21)
    assert .795 in axes['spot_ratios'] and .8 in axes['spot_ratios'] and .805 in axes['spot_ratios']
    assert .995 in axes['spot_ratios'] and 1.005 in axes['spot_ratios']
    progress=[]
    kwargs=dict(spot_ratios=(1.,),rho_shifts=(0.,),roll_days=(0,21,63),initial_ki=(False,True),metrics=('present_value',),pde_config=sp.PDEConfig(101),mc_checkpoints=(sp.ScenarioShock(spot=100.),),mc_config=sp.MCConfig(n_paths=100,seed=2))
    result=sp.risk_surface(product,market,model,progress=lambda i,n,c: progress.append((i,n)),**kwargs)
    again=sp.risk_surface(product,market,model,**kwargs)
    assert len(result.cells)==4 and progress[-1]==(4,4)
    assert result.mc_checkpoints == again.mc_checkpoints
    assert result.mc_checkpoints[0]['price']['confidence_interval'] is not None
    assert result.mc_checkpoints[0]['price']['knock_out_probability'] is not None
    assert result.mc_checkpoints[0]['price']['expected_redemption_time'] is not None


def test_scalar_only_certified_model_growth_protocol():
    product,market,_=inputs(2)
    class ScalarModel:
        anchor_spot=100.
        time_offset=0.
        day_count='ACT/252'
        def discount(self,t): return exp(-.03*t)
        def forward_growth(self,a,b): return exp(.01*(b-a))
        def local_vol(self,t,s): return 0.
        def diagnostics(self): return {'kind':'scalar_test'}
        def is_deterministic(self,h): return True
    assert sp.price(product,market,'pde',sp.PDEConfig(101),model=ScalarModel()).present_value > 0


def test_repeated_roll_cash_is_not_paid_twice_and_terminal_continuation():
    product,market,model=inputs(4)
    product=replace(product,autocall=None,coupon=sp.CouponFeature(sp.CouponRule.FIXED_PERIODIC,(1,3,4),(.12,.12,.12)))
    first=sp.transform_scenario(product,market,model,sp.ScenarioShock(spot=90.,roll_days=1))
    second=sp.advance_scenario(first,sp.ScenarioShock(roll_days=2))
    direct=sp.transform_scenario(product,market,model,sp.ScenarioShock(spot=90.,roll_days=3))
    assert [c.day for c in second.cashflows] == [1,3]
    assert second.cashflows == direct.cashflows
    assert second.model.identity == direct.model.identity
    assert second.product == direct.product
    assert second.initial_requested_knocked_in is False
    final=sp.advance_scenario(second,sp.ScenarioShock(spot=60.,roll_days=1))
    after=sp.advance_scenario(final,sp.ScenarioShock(spot=70.,roll_days=5))
    assert after.cashflows == final.cashflows
    assert after.terminated and after.realized_knocked_in
    assert after.discount_to_roll == pytest.approx(model.discount(9/252))
    assert sp.scenario_risk(after)['metrics'] == dict.fromkeys(sp.scenario_risk(after)['metrics'],0.)
    assert sp.Scenario.from_dict(after.to_dict()).terminated


def test_level_rho_units_match_direct_model_shocks():
    product,market,model=inputs(3)
    node=sp.transform_scenario(product,market,model,sp.ScenarioShock(level_multiplier=1.2,rho_shift=.1))
    assert node.model.surface.theta == pytest.approx(np.array(model.surface.theta)*1.44)
    assert node.model.surface.rho == pytest.approx(-.3)
    config=sp.PDEConfig(101,spot_grid_upper=420.)
    risk=sp.scenario_risk(node,pde_config=config)
    for key in ('atm_level_sensitivity','rho_skew_sensitivity'):
        prices=[]
        for sign in (1,-1):
            surface=replace(node.model.surface,**(
                {'theta':tuple(v*(1+sign*.01)**2 for v in node.model.surface.theta)}
                if key=='atm_level_sensitivity' else {'rho':node.model.surface.rho+sign*.01}))
            direct_model=sp.LocalVolModel.theoretical(surface,node.model.curves)
            prices.append(sp.price(product,market,'pde',config,model=direct_model).present_value)
        assert risk['metrics'][key] == pytest.approx((prices[0]-prices[1])/.02)


def test_invalid_axes_export_strict_json_csv_and_validation(tmp_path):
    product,market,model=inputs(2)
    result=sp.risk_surface(product,market,model,spot_ratios=(1.,float('nan'),-.1),rho_shifts=(0.,),roll_days=(0,),initial_ki=(False,),metrics=('present_value',),pde_config=sp.PDEConfig(101))
    assert [c['status'] for c in result] == ['ok','error','error']
    result.write_json(tmp_path/'grid.json')
    result.write_csv(tmp_path/'grid.csv')
    import csv
    rows=list(csv.DictReader((tmp_path/'grid.csv').open()))
    restored=sp.Scenario.from_dict(json.loads(rows[0]['scenario']))
    assert restored.market.spots == (100.,)
    with pytest.raises(ValueError,match='unknown|known'):
        sp.scenario_risk(restored,metrics=('vega',))
    bad_bump=sp.scenario_risk(restored,spot_bump=1.)
    assert bad_bump['metrics']['present_value']>0
    assert bad_bump['metrics']['delta'] is None
    assert 'bump' in bad_bump['metric_status']['delta']['error']
    payload=restored.to_dict()
    payload['original_product']['notional']=float('nan')
    with pytest.raises(ValueError,match='finite'):
        sp.Scenario.from_dict(payload)


def test_default_expiry_and_invalid_roll_are_never_silently_lost():
    product,market,model=inputs(4)
    assert sp.default_risk_axes(product)['roll_days'] == (0,4)
    result=sp.risk_surface(product,market,model,spot_ratios=(1.,),rho_shifts=(0.,),
        roll_days=(0,4,5,float('nan')),initial_ki=(False,),metrics=('present_value',),pde_config=sp.PDEConfig(101))
    assert [cell['status'] for cell in result] == ['ok','ok','error']
    assert result.cells[-1]['request']['roll_days'] is None
    json.dumps(result.to_dict(),allow_nan=False)


def test_product_elapsed_clock_cannot_advance_the_model_and_initial_ki_is_sticky():
    product,market,model=inputs(4)
    product=replace(product,elapsed_days=200,knock_in=replace(product.knock_in,initially_knocked_in=True))
    model=model.rolled(.25,100.)
    node=sp.transform_scenario(product,market,model,sp.ScenarioShock(spot=105.,roll_days=1))
    assert node.product.elapsed_days == 201
    assert node.model.time_offset == pytest.approx(.25+1/252)
    assert node.initial_requested_knocked_in and node.realized_knocked_in
    assert node.product.knock_in.initially_knocked_in
    continued=sp.advance_scenario(sp.Scenario.from_dict(node.to_dict()),sp.ScenarioShock(roll_days=1))
    assert continued.model.time_offset==pytest.approx(.25+2/252)
    assert continued.product.elapsed_days==202
    assert sp.Scenario.from_dict(continued.to_dict()).product==continued.product
    with pytest.raises(ValueError,match='reset'):
        sp.advance_scenario(node,sp.ScenarioShock(initial_knocked_in=False))


def test_scenario_loader_rejects_inconsistent_history_and_mutable_cash():
    product,market,model=inputs(2)
    node=sp.transform_scenario(product,market,model,sp.ScenarioShock())
    payload=node.to_dict()
    payload['realized_knocked_in']='false'
    with pytest.raises(ValueError,match='boolean'):
        sp.Scenario.from_dict(payload)
    payload=node.to_dict()
    payload['realized_knocked_in']=True
    with pytest.raises(ValueError,match='history'):
        sp.Scenario.from_dict(payload)
    mutable=[]
    detached=replace(node,cashflows=mutable)
    mutable.append(sp.ScenarioCashflow(1,1.,'principal',1.))
    assert detached.cashflows==()


def test_nonfinite_bumps_preserve_pv_and_export_individual_errors():
    product,market,model=inputs(2)
    result=sp.risk_surface(product,market,model,spot_ratios=(1.,),rho_shifts=(0.,),
        roll_days=(0,),initial_ki=(False,),pde_config=sp.PDEConfig(101),
        spot_bump=float('inf'),level_bump=float('nan'),rho_bump=0.)
    cell=result.cells[0]
    assert cell['status']=='partial'
    assert cell['metrics']['present_value']>0
    assert all(status['status']=='error' for key,status in cell['metric_status'].items() if key!='present_value')
    json.dumps(result.to_dict(),allow_nan=False)


def test_numpy_shock_scalars_export_as_native_json():
    product,market,model=inputs(2)
    node=sp.transform_scenario(product,market,model,
        sp.ScenarioShock(spot=np.float32(90.),rho_shift=np.float32(.1)))
    payload=json.loads(json.dumps(node.to_dict(),allow_nan=False))
    assert payload['market']['spots']==[90.]
    assert payload['shock']['rho_shift']==pytest.approx(.1)


def test_risk_cli_loads_surface_curves_and_keeps_legacy_commands(tmp_path,capsys):
    from structured_products.cli import main
    _,_,model=inputs(2)
    surface=tmp_path/'surface.json'
    curves=tmp_path/'curves.json'
    surface.write_text(json.dumps(model.surface.to_dict()))
    curves.write_text(json.dumps(model.curves.to_dict()))
    output=tmp_path/'risk.json'
    assert main(['risk-surface','--example','classic_snowball','--surface',str(surface),'--curves',str(curves),
        '--spot-ratios','1.0','--rho-shifts','0','--roll-days','0','--initial-ki','false',
        '--metrics','present_value','--grid-size','101','--output',str(output)]) == 0
    data=json.loads(output.read_text())
    assert data['cells'][0]['status']=='ok'
    assert list(data['cells'][0]['metrics'])==['present_value']
    capsys.readouterr()
    assert main(['price','--example','classic_snowball','--paths','20','--no-greeks'])==0
    assert 'price' in json.loads(capsys.readouterr().out)


def test_saved_calibration_scenario_and_cli_model_sources(tmp_path,capsys):
    from importlib.resources import files
    from structured_products.cli import main
    data=sp.load_market_data(files('structured_products').joinpath('data/market_surface'))
    fit=sp.calibrate_ssvi(sp.validate_quotes(data.rows,data.snapshot,data.calendar,data.curves))
    assert fit.accepted
    model=sp.LocalVolModel.from_calibration(fit)
    product,market,_=inputs(2)
    for shock in (sp.ScenarioShock(),sp.ScenarioShock(spot=90.,level_multiplier=1.2,rho_shift=-.1,roll_days=1)):
        scenario=sp.transform_scenario(product,market,model,shock)
        restored=sp.Scenario.from_dict(scenario.to_dict())
        assert restored.baseline_model_identity==model.identity
        assert restored.model.identity==scenario.model.identity
        assert sp.price_scenario(restored,'pde',sp.PDEConfig(101)).present_value==sp.price_scenario(scenario,'pde',sp.PDEConfig(101)).present_value
    fit.write_json(tmp_path/'fit.json')
    (tmp_path/'model.json').write_text(json.dumps(model.to_dict(),allow_nan=False))
    for source,path in (('--calibration','fit.json'),('--model','model.json')):
        code=main(['risk-surface','--example','classic_snowball',source,str(tmp_path/path),
            '--spot-ratios','1','--rho-shifts','0','--roll-days','0','--initial-ki','false',
            '--metrics','present_value','--grid-size','101'])
        captured=capsys.readouterr()
        assert code==0,captured.err
        assert json.loads(captured.out)['cells'][0]['status']=='ok'


def test_csv_curve_cli_checkpoint_export_and_errors(tmp_path,capsys):
    from importlib.resources import files
    from structured_products.cli import main
    _,_,model=inputs(2)
    (tmp_path/'surface.json').write_text(json.dumps(model.surface.to_dict()))
    (tmp_path/'points.json').write_text(json.dumps([{'spot':100.,'roll_days':252}]))
    args=['risk-surface','--example','classic_snowball','--surface',str(tmp_path/'surface.json'),
        '--curves',str(files('structured_products').joinpath('data/market_surface/curves.csv')),
        '--spot-ratios','1','--rho-shifts','0','--roll-days','252','--initial-ki','false',
        '--metrics','present_value','--grid-size','101','--mc-checkpoints',str(tmp_path/'points.json'),
        '--output',str(tmp_path/'out.csv'),'--csv-output',str(tmp_path/'extra.csv')]
    assert main(args)==0
    result=json.loads(capsys.readouterr().out)
    assert result['mc_checkpoints'][0]['price']['present_value']==0.
    assert (tmp_path/'out.csv').read_text()==(tmp_path/'extra.csv').read_text()
    assert main(['risk-surface','--example','classic_snowball','--surface',str(tmp_path/'surface.json')])==2
    assert '--curves' in capsys.readouterr().err


@pytest.mark.parametrize('field,value', [
    ('discount_to_roll',-1.), ('discount_to_roll',0.), ('roll_days',3),
    ('maturity_days',2), ('elapsed_days',2), ('coupon_accrued_days',1),
    ('cash_day',2), ('redemption_day',1),
])
def test_persisted_lifecycle_rejects_inconsistent_dates_and_discount(field,value):
    product,market,model=inputs(4)
    product=replace(product,autocall=None,elapsed_days=10,coupon_accrued_days=2,
        coupon=sp.CouponFeature(sp.CouponRule.FIXED_PERIODIC,(1,4),(.12,.12)))
    node=sp.transform_scenario(product,market,model,sp.ScenarioShock(roll_days=1))
    payload=node.to_dict()
    if field=='roll_days': payload['shock']['roll_days']=value
    elif field in ('maturity_days','elapsed_days','coupon_accrued_days'): payload['product'][field]=value
    elif field=='cash_day': payload['cashflows'][0]['day']=value
    else: payload[field]=value
    with pytest.raises(ValueError):
        sp.Scenario.from_dict(payload)


@pytest.mark.parametrize('section,key', [('knock_in','initially_knocked_in'),
    ('coupon','survives_knock_in'),('redemption','include_principal')])
@pytest.mark.parametrize('which', ['original_product','product'])
def test_persisted_product_boolean_types_are_strict(section,key,which):
    product,market,model=inputs(4)
    node=sp.transform_scenario(product,market,model,sp.ScenarioShock(initial_knocked_in=True))
    payload=node.to_dict()
    payload[which][section][key]='false'
    with pytest.raises(ValueError,match='boolean'):
        sp.Scenario.from_dict(payload)


def test_terminal_archive_bounds_and_post_redemption_continuation():
    product,market,model=inputs(4)
    node=sp.transform_scenario(product,market,model,sp.ScenarioShock(roll_days=4))
    after=sp.advance_scenario(node,sp.ScenarioShock(roll_days=5))
    restored=sp.Scenario.from_dict(after.to_dict())
    assert restored.shock.roll_days==9 and restored.redemption_day==4
    assert restored.cashflows==node.cashflows
    payload=after.to_dict()
    payload['redemption_day']=5
    with pytest.raises(ValueError,match='redemption'):
        sp.Scenario.from_dict(payload)
    payload=after.to_dict()
    payload['cashflows'][0]['day']=5
    with pytest.raises(ValueError,match='cash'):
        sp.Scenario.from_dict(payload)


@pytest.mark.parametrize('terminal', [False,True])
def test_unrepresentable_bumps_are_individual_errors_with_valid_pv(terminal):
    product,market,model=inputs(4)
    node=sp.transform_scenario(product,market,model,sp.ScenarioShock(roll_days=4 if terminal else 1))
    risk=sp.scenario_risk(node,pde_config=sp.PDEConfig(101),spot_bump=1e-300,
        level_bump=1e-300,rho_bump=1e-300)
    assert risk['metric_status']['present_value']['status']=='ok'
    assert risk['metrics']['present_value']==0. if terminal else risk['metrics']['present_value']>0
    for metric in ('delta','gamma','atm_level_sensitivity','rho_skew_sensitivity'):
        assert risk['metrics'][metric] is None
        assert risk['metric_status'][metric]['status']=='error'
    json.dumps(risk,allow_nan=False)


def test_overflowing_or_unsupported_spot_axis_is_isolated_before_domain(tmp_path):
    product,market,model=inputs(4)
    grid=sp.risk_surface(product,market,model,spot_ratios=(1.,1e308,1e306,'bad',None,1+2j),
        rho_shifts=(0.,),roll_days=(0,),initial_ki=(False,),metrics=('present_value',),pde_config=sp.PDEConfig(101))
    assert [cell['status'] for cell in grid]==['ok','error','error','error','error','error']
    assert grid.diagnostics['spot_grid_upper']==pytest.approx(404.)
    assert grid.cells[0]['metrics']['present_value']>0
    assert grid.cells[1]['request']['spot_ratio']==1e308
    grid.write_json(tmp_path/'finite.json')
    grid.write_csv(tmp_path/'finite.csv')


def test_terminal_invalid_controls_have_error_status_without_pricing():
    product,market,model=inputs(4)
    node=sp.transform_scenario(product,market,model,sp.ScenarioShock(roll_days=4))
    risk=sp.scenario_risk(node,spot_bump=float('nan'),level_bump=0.,rho_bump=float('inf'))
    assert risk['metrics']['present_value']==0.
    assert all(status['status']=='error' for metric,status in risk['metric_status'].items() if metric!='present_value')


def test_time_vol_scenario_retains_price_when_surface_derivatives_unsupported():
    product,market,model=inputs(4)
    node=sp.transform_scenario(product,market,sp.TimeVolModel(model.curves,(0.,),(.2,)))
    risk=sp.scenario_risk(node,pde_config=sp.PDEConfig(101))
    assert risk['metrics']['present_value']>0
    assert risk['metric_status']['rho_skew_sensitivity']['status']=='error'
