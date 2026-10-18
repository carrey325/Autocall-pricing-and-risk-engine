from dataclasses import replace
import csv
import json
import pytest
import numpy as np
import structured_products as sp


def inputs():
    p=sp.make_classic_snowball(maturity_days=4,observation_days=(2,4))
    m=sp.MarketData((100.,),.03,(.02,),(.2,))
    c=sp.MarketCurves((0.,1.),(100.,101.),(1.,.97))
    v=sp.LocalVolModel.theoretical(sp.SSVISurface((.25,1.),(.01,.04),-.4,.4),c)
    return p,m,v


def test_stress_zero_holder_loss_reconstruction_and_invalid_surface(tmp_path):
    p,m,v=inputs()
    specs=(sp.StressSpec('zero'),sp.StressSpec('down',sp.ScenarioShock(spot=70.)),
           sp.StressSpec('invalid',sp.ScenarioShock(rho_shift=-.7)))
    r=sp.stress_test(p,m,v,scenarios=specs,pde_config=sp.PDEConfig(101))
    assert r.cells[0]['holder_pnl']==0 and r.cells[0]['holder_loss']==0
    for c in r.cells[:2]:
        n=sp.Scenario.from_dict(c['scenario'])
        assert c['total_base_date_value']==pytest.approx(n.cash_pv+n.discount_to_roll*c['remaining_pv'])
        assert c['holder_loss']==pytest.approx(r.diagnostics['base_pv']-c['total_base_date_value'])
        assert c['full_revaluation_residual']==pytest.approx(c['holder_pnl']-c['local_approximation_pnl'])
        assert len(c['input_hash'])==len(c['code_hash'])==64
    assert r.cells[2]['status']=='error'
    r.write_json(tmp_path/'stress.json'); r.write_csv(tmp_path/'stress.csv')
    rows=list(csv.DictReader((tmp_path/'stress.csv').open()))
    assert sp.Scenario.from_dict(json.loads(rows[0]['scenario'])).product==p


def test_observed_close_explicit_history_and_no_day_zero_observation():
    p,m,v=inputs()
    with pytest.raises(ValueError,match='history'):
        sp.StressSpec('missing',sp.ScenarioShock(spot=100.),'observed_close')
    specs=(sp.StressSpec('mark',sp.ScenarioShock(spot=100.)),
        sp.StressSpec('day1',sp.ScenarioShock(spot=100.,roll_days=1),'observed_close',(100.,)),
        sp.StressSpec('day2',sp.ScenarioShock(spot=100.,roll_days=2),'observed_close',(70.,100.)))
    r=sp.stress_test(p,m,v,scenarios=specs,pde_config=sp.PDEConfig(101))
    assert not r.cells[0]['scenario']['terminated']
    assert not r.cells[1]['scenario']['terminated']
    assert r.cells[2]['scenario']['terminated'] and r.cells[2]['scenario']['realized_knocked_in']
    assert r.cells[2]['remaining_pv']==0 and r.cells[2]['cash_sum']>1000


def test_generic_terms_clone_preserves_schedule_history_and_fair_coupon():
    p,m,v=inputs()
    p=replace(p,elapsed_days=17,coupon_accrued_days=3)
    clone=sp.with_terms(p,coupon=.2,ki=.7,ko=1.05)
    assert clone.elapsed_days==17 and clone.coupon_accrued_days==3
    assert clone.autocall.observation_days==p.autocall.observation_days
    assert clone.autocall.barrier_ratios==(1.05,1.05)
    assert clone.reference_spots==p.reference_spots
    r=sp.compare_terms(p,m,v,coupons=(.1,.2),ki_ratios=(.7,),ko_ratios=(1.,),
        scenarios=(sp.StressSpec('zero'),),pde_config=sp.PDEConfig(101),fair_bracket=(0.,2.))
    assert len(r.cells)==2 and len(r.diagnostics['fair_coupons'])==1
    f=r.diagnostics['fair_coupons'][0]
    assert f['status']=='ok'
    fair=sp.with_terms(p,coupon=f['result']['coupon_rate'],ki=.7,ko=1.)
    price=sp.price(fair,m,'pde',sp.PDEConfig(101,spot_grid_upper=r.diagnostics['spot_grid_upper']),model=v)
    assert price.present_value==pytest.approx(p.issue_price,abs=1e-6)
    assert all(c['stress']['cells'][0]['holder_loss']==0 for c in r.cells)


def test_failed_fair_bracket_and_mc_stress_are_recorded():
    p,m,v=inputs()
    r=sp.compare_terms(p,m,v,coupons=(.1,),ki_ratios=(.7,),ko_ratios=(1.,),
        scenarios=(),pde_config=sp.PDEConfig(101),fair_bracket=(0.,1e-8))
    assert r.diagnostics['fair_coupons'][0]['status']=='error'
    assert r.diagnostics['fair_coupons'][0]['bracket']==[0.,1e-8]
    mc=sp.stress_test(p,m,v,scenarios=(sp.StressSpec('zero'),),method='mc',
        mc_config=sp.MCConfig(n_paths=100,seed=7),pde_config=sp.PDEConfig(101))
    assert mc.cells[0]['holder_pnl']==0
    assert mc.cells[0]['price']['standard_error'] is not None


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.1])
def test_coupon_nonfinite_rejected(value):
    p,_,_=inputs()
    with pytest.raises(ValueError): sp.with_flat_coupon(p,value)


def test_observed_endpoint_optional_spot_has_identical_taylor():
    p,m,v=inputs()
    specs=tuple(sp.StressSpec(str(s),sp.ScenarioShock(spot=s,roll_days=1),'observed_close',(90.,)) for s in (None,90.))
    r=sp.stress_test(p,m,v,scenarios=specs,pde_config=sp.PDEConfig(101))
    assert r.cells[0]['holder_pnl']==r.cells[1]['holder_pnl']
    assert r.cells[0]['local_approximation_pnl']==r.cells[1]['local_approximation_pnl']


def test_bad_term_rows_finite_and_fcn_absent_ko():
    p,m,v=inputs()
    r=sp.compare_terms(p,m,v,coupons=(.1,float('nan')),ki_ratios=(.7,),ko_ratios=(1.,),scenarios=(),pde_config=sp.PDEConfig(101))
    assert [c['status'] for c in r.cells]==['ok','error']
    assert r.to_dict()['cells'][1]['request']['coupon'] is None
    fcn=sp.make_standard_fcn(maturity_days=4,payment_days=(2,4))
    r=sp.compare_terms(fcn,m,v,coupons=(.1,),ki_ratios=(.7,),ko_ratios=(None,),scenarios=(),pde_config=sp.PDEConfig(101))
    assert r.cells[0]['status']=='ok'


def test_overflow_stress_isolated_from_valid_domain():
    p,m,v=inputs()
    r=sp.stress_test(p,m,v,scenarios=(sp.StressSpec('valid'),sp.StressSpec('overflow',sp.ScenarioShock(spot=1e308))),pde_config=sp.PDEConfig(101))
    assert r.cells[0]['status']=='ok' and r.cells[1]['status']=='error'
    json.dumps(r.to_dict(),allow_nan=False)


def test_prospective_ki_included_in_common_domain():
    p,m,v=inputs()
    r=sp.compare_terms(p,m,v,coupons=(.1,),ki_ratios=(5.,),ko_ratios=(1.,),scenarios=(),pde_config=sp.PDEConfig(101))
    assert r.diagnostics['spot_grid_upper']>=2000.
    assert r.cells[0]['status']=='ok'


def test_native_numeric_stress_request_hash_and_same_day_priority():
    p,m,v=inputs()
    p=replace(p,knock_in=replace(p.knock_in,barrier_ratio=1.),
        autocall=sp.AutocallFeature((1,4),(1.,1.),(.1,.1)))
    for priority,ki in ((sp.SameDayPriority.KNOCK_OUT_FIRST,False),(sp.SameDayPriority.KNOCK_IN_FIRST,True)):
        candidate=replace(p,same_day_priority=priority)
        spec=sp.StressSpec('overlap',sp.ScenarioShock(spot=np.float32(100.),roll_days=1),
            'observed_close',(np.float32(100.),))
        r=sp.stress_test(candidate,m,v,scenarios=(spec,),pde_config=sp.PDEConfig(101))
        assert r.cells[0]['scenario']['terminated']
        assert r.cells[0]['scenario']['realized_knocked_in'] is ki
        json.dumps(r.to_dict(),allow_nan=False)
