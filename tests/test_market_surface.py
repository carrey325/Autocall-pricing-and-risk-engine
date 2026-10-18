"""Market-clock, quote quality and smile lifecycle regression tests."""

from dataclasses import FrozenInstanceError, replace
from datetime import date
from pathlib import Path
import json

import numpy as np
import pytest

import structured_products as sp


def curves():
    return sp.MarketCurves((0, 0.5, 1), (100, 102, 105), (1, 0.98, 0.95))


def surface(eta=0.7):
    return sp.SSVISurface((0.1, 0.25, 0.5, 1, 2),
                          (0.004, 0.01, 0.022, 0.045, 0.09),
                          rho=-0.4, eta=eta, gamma=0.3)


def test_curves_interpolate_logs_and_use_elapsed_ratios():
    c = curves()
    assert c.forward(0.25) == pytest.approx(np.sqrt(100 * 102))
    assert c.discount(0.25) == pytest.approx(np.sqrt(0.98))
    assert c.discount_between(0.25, 0.75) == pytest.approx(c.discount(.75)/c.discount(.25))
    assert c.forward_growth(.25, .75) == pytest.approx(c.forward(.75)/c.forward(.25))
    assert c.forward(1.5) == pytest.approx(105**2 / 102)
    assert sp.MarketCurves.from_dict(c.to_dict()) == c
    with pytest.raises(FrozenInstanceError):
        c.forwards = (1, 2, 3)
    for args in [((0, 1), (100, -1), (1, .9)),
                 ((.1, 1), (100, 101), (1, .9)),
                 ((0, 0), (100, 101), (1, .9)),
                 ((0, 1), (100, 101), (1, np.nan))]:
        with pytest.raises(ValueError):
            sp.MarketCurves(*args)
    with pytest.raises(ValueError):
        c.discount(-.1)
    with pytest.raises(ValueError):
        c.forward_growth(.5, .4)


def test_ssvi_analytic_derivatives_match_independent_differences_and_flat_limit():
    s = surface()
    for k, t in [(-.3, .04), (0., .3), (.4, 1.4)]:
        d = s.derivatives(k, t)
        h = 1e-4
        assert d.w == s.total_variance(k, t)
        assert d.w_k == pytest.approx((s.total_variance(k+h,t)-s.total_variance(k-h,t))/(2*h), rel=1e-5, abs=1e-8)
        assert d.w_kk == pytest.approx((s.total_variance(k+h,t)-2*d.w+s.total_variance(k-h,t))/h**2, rel=2e-5)
        assert d.w_t == pytest.approx((s.total_variance(k,t+h)-s.total_variance(k,t-h))/(2*h), rel=2e-5)
    flat = sp.SSVISurface((.25, 1), (.01, .04), rho=0, eta=0, gamma=.5)
    for t in [0, .000001, .2, .7, 2]:
        d = flat.derivatives(.9, t)
        assert d.w == pytest.approx(.04*t)
        assert d.w_t == pytest.approx(.04)
        assert (d.w_k, d.w_kk) == (0, 0)
    d0 = s.derivatives(-.2, 0)
    assert d0.w == d0.w_k == d0.w_kk == 0
    assert np.isfinite(d0.w_t) and d0.w_t > 0
    assert s.total_variance(-.2, 1e-10)/1e-10 == pytest.approx(d0.w_t, rel=1e-5)
    assert sp.SSVISurface.from_dict(s.to_dict()) == s
    json.dumps(s.to_dict(), allow_nan=False)


def test_ssvi_rejects_global_arbitrage_and_nonmonotone_nodes():
    for change in [dict(eta=5), dict(rho=1), dict(gamma=.8),
                   dict(theta_floor=0), dict(theta=(.004,.003,.022,.045,.09))]:
        with pytest.raises(ValueError):
            replace(surface(), **change)
    s = surface()
    checks = s.arbitrage_diagnostics()
    assert checks['analytic_sufficient']
    assert checks['min_density_factor'] > 0
    assert checks['min_w_t'] >= 0
    assert checks['scope'] == 'all finite k and t >= 0; numerical grid supplemental'
    with pytest.raises(ValueError):
        s.derivatives(0, -.1)
    with pytest.raises(ValueError):
        s.total_variance(np.nan, 1)


def test_black_inversion_uses_model_time_and_bounds():
    # Hand-computed ATM forward call with total variance .04: 100*(2*N(.1)-1).
    p = sp.black_price(100, 100, .97, .04, 'call')
    assert p == pytest.approx(7.726600431743627)
    assert sp.implied_volatility(p, 100, 100, .97, .5, 'call') == pytest.approx(np.sqrt(.08))
    assert sp.black_price(100, 110, 1, 0, 'put') == 10
    with pytest.raises(ValueError):
        sp.implied_volatility(101, 100, 100, 1, 1, 'call')
    with pytest.raises(ValueError):
        sp.implied_volatility(1, 100, 100, 1, 0, 'call')


def market_inputs():
    snap = sp.MarketSnapshot('TEST',100,'USD','index_points',
                             '2026-01-02T21:00:00+00:00', 'PM',
                             (('source','unit-test'),('kind','synthetic')))
    cal = sp.TradingCalendar(('2026-01-02','2026-01-05','2026-01-06'), (0,1/252,2/252))
    c = sp.MarketCurves((0,2/252),(100,100),(1,1))
    p = sp.black_price(100,100,1,.04*2/252,'call')
    row = dict(symbol='TEST',quote_timestamp=snap.timestamp,expiry='2026-01-06',
               settlement='PM',strike='100',option_type='call',bid=str(p-.01),
               ask=str(p+.01),exercise_style='European')
    return snap,cal,c,row


def test_quote_validation_keeps_all_rows_and_reasons_and_bid_ask_parity():
    snap,cal,c,row = market_inputs()
    bad = [dict(row,bid='NaN'),dict(row,strike=''),dict(row,bid='2',ask='1'),
           dict(row,ask='100'),dict(row,exercise_style='American'),
           dict(row,expiry='2026-01-07'),dict(row,settlement='AM'),
           dict(row,quote_timestamp='2026-01-01T21:00:00+00:00'),
           dict(row,symbol='OTHER'),dict(row,option_type='other'),dict(row,volume='-1'),
           dict(row,bid='101',ask='102')]
    result = sp.validate_quotes([row, *bad],snap,cal,c)
    assert len(result.records)==13 and len(result.accepted)==1
    assert all(r.reasons for r in result.records[1:])
    assert result.accepted[0].model_time == pytest.approx(2/252)
    assert result.accepted[0].implied_vol == pytest.approx(.2)
    assert dict(result.records[1].raw)['bid'] == 'NaN'
    json.dumps(result.to_dict(), allow_nan=False)
    duplicate = sp.validate_quotes([row,dict(row)],snap,cal,c)
    assert len(duplicate.accepted)==0
    assert all('duplicate' in r.reasons for r in duplicate.records)
    # Midpoints need not satisfy parity if the executable bid/ask interval does.
    put = dict(row,option_type='put',bid=str(float(row['bid'])+.005),ask=str(float(row['ask'])+.005))
    assert len(sp.validate_quotes([row,put],snap,cal,c).accepted)==2
    put.update(bid='1',ask='1.02')
    parity = sp.validate_quotes([row,put],snap,cal,c)
    assert not parity.accepted
    assert all('put_call_parity' in r.reasons for r in parity.records)


def test_calendar_requires_explicit_act252_dates_and_snapshot_provenance():
    snap,cal,c,row = market_inputs()
    assert cal.time(date(2026,1,5)) == pytest.approx(1/252)
    assert sp.TradingCalendar.from_dict(cal.to_dict()) == cal
    assert sp.MarketSnapshot.from_dict(snap.to_dict()) == snap
    with pytest.raises(ValueError):
        sp.TradingCalendar(('2026-01-02','2026-01-05'),(0,1/365))
    with pytest.raises(ValueError):
        replace(snap,provenance=())
    with pytest.raises(ValueError):
        replace(snap,timestamp='2026-01-02T16:00:00')
    with pytest.raises(ValueError):
        sp.validate_quotes([row],snap,cal,sp.MarketCurves((0,1),(101,101),(1,1)))


def test_am_expiry_uses_explicit_intraday_calendar_fraction():
    snap,_,c,row = market_inputs()
    snap = replace(snap,timestamp='2026-01-02T15:45:00-05:00',settlement_convention='explicit_calendar')
    labels = (snap.timestamp,'2026-01-05T09:30:00-05:00','2026-01-06T09:30:00-05:00')
    cal = sp.TradingCalendar(labels,(0,.04/252,1.04/252),(.96,0,0))
    row.update(quote_timestamp=snap.timestamp,expiry=labels[-1],settlement='AM')
    p = sp.black_price(100,100,1,.04*1.04/252,'call')
    row.update(bid=p-.01,ask=p+.01)
    result = sp.validate_quotes([row],snap,cal,c)
    assert len(result.accepted)==1
    assert result.accepted[0].model_time == pytest.approx(1.04/252)
    assert result.accepted[0].implied_vol == pytest.approx(.2)
    with pytest.raises(ValueError):
        sp.TradingCalendar(labels,(0,1/252,2/252))


def test_synthetic_five_expiry_pipeline_calibrates_and_roundtrips(tmp_path):
    folder = Path(__file__).resolve().parents[1]/'src'/'structured_products'/'data'/'market_surface'
    data = sp.load_market_data(folder)
    validated = sp.validate_quotes(data.rows,data.snapshot,data.calendar,data.curves)
    assert len(validated.accepted) >= 45
    assert len({q.model_time for q in validated.accepted}) == 5
    for t in {q.model_time for q in validated.accepted}:
        assert len({q.strike for q in validated.accepted if q.model_time==t})>=9
    fit = sp.calibrate_ssvi(validated)
    assert fit.accepted, fit.to_dict()
    assert fit.overall['price_rmse'] < .005
    assert fit.overall['iv_rmse'] < .001
    assert fit.overall['bid_ask_coverage'] == 1
    assert len(fit.per_expiry)==5
    assert fit.optimizer['success']
    assert fit.surface.rho == pytest.approx(-.45,abs=1e-6)
    assert fit.surface.eta == pytest.approx(.65,abs=1e-6)
    assert fit.surface.gamma == pytest.approx(.3,abs=1e-6)
    reconstructed = sp.SSVISurface.from_dict(fit.to_dict()['surface'])
    assert reconstructed.total_variance(-.15,.4)==fit.surface.total_variance(-.15,.4)
    fit.write_json(tmp_path/'fit.json')
    assert json.loads((tmp_path/'fit.json').read_text())['accepted']
    strict = sp.calibrate_ssvi(validated,sp.CalibrationConfig(max_iv_rmse=1e-14,max_price_rmse=1e-14))
    assert not strict.accepted
    assert any(not g.passed for g in strict.gates)
    stopped = sp.calibrate_ssvi(validated,sp.CalibrationConfig(max_nfev=1))
    assert not stopped.accepted
    assert any(g.name=='optimizer_converged' and not g.passed for g in stopped.gates)


def test_ssvi_knots_have_continuous_derivatives_and_explicit_terminal_right_slope():
    s = surface()
    h = 1e-7
    for t in s.times[:-1]:
        left,right = s.derivatives(-.1,t-h),s.derivatives(-.1,t+h)
        assert left.w_t == pytest.approx(right.w_t,rel=1e-5)
        assert left.w_k == pytest.approx(right.w_k,rel=1e-5)
    flat_smile = sp.SSVISurface((.1,.5,1),(.004,.03,.04),0,0)
    assert flat_smile.derivatives(0,1).w_t == pytest.approx(.02)
    assert flat_smile.derivatives(0,2).w == pytest.approx(.06)
    ks = np.array([-.5,0,.5])
    ds = s.derivatives(ks,.4)
    assert np.allclose(ds.w,[s.total_variance(k,.4) for k in ks])
    assert np.all(ds.w_t>=0)


def test_market_parameters_copy_mutable_inputs_and_curves_supply_interval_rates():
    times,forward,discount = [0,1],[100,105],[1,.95]
    c = sp.MarketCurves(times,forward,discount)
    times[1],forward[1],discount[1] = 2,200,.5
    assert c.forward(1)==pytest.approx(105)
    assert c.instantaneous_rate(.5)==pytest.approx(-np.log(.95))
    assert c.instantaneous_carry(.5)==pytest.approx(np.log(1.05))
    assert np.allclose(c.discount(np.array([0,.5,1])),[1,np.sqrt(.95),.95])
    assert c.discount_between(.5,.5)==1
    with pytest.raises(ValueError):
        c.discount_between(1,0)
    theta = [.01,.04]
    s = sp.SSVISurface([.25,1],theta,0,0)
    theta[1]=1
    assert s.total_variance(0,1)==pytest.approx(.04)


@pytest.mark.parametrize('change,reason',[
    ({'bid':'NaN'},'invalid_numeric'), ({'strike':''},'missing_required'),
    ({'bid':'2','ask':'1'},'crossed_quote'), ({'ask':'100'},'wide_quote'),
    ({'exercise_style':'American'},'non_european'), ({'settlement':'AM'},'ambiguous_settlement'),
    ({'bid':'101','ask':'102'},'option_bounds'), ({'bid':'0','ask':'0.00000000000000000001','strike':'80'},'option_bounds'),
    ({'quote_timestamp':'2026-01-02T21:00:00'},'invalid_timestamp'),
    ({'expiry':'2026-01-02'},'expired')])
def test_rejection_codes_identify_actual_quote_failure(change,reason):
    snap,cal,c,row = market_inputs()
    row.update(change)
    result = sp.validate_quotes([row],snap,cal,c)
    assert not result.accepted
    assert reason in result.records[0].reasons


def test_black_bad_inputs_and_nonidentifiable_iv_are_rejected():
    for args in [(0,100,1,.1),(100,-1,1,.1),(100,100,-1,.1),(100,100,1,-.1),
                 (100,100,1,np.nan),(100,100,1,.1,'other')]:
        with pytest.raises(ValueError):
            sp.black_price(*args)
    with pytest.raises(ValueError):
        sp.implied_volatility(0,100,100,1,1)
    for kwargs in [dict(max_iv_rmse=-1),dict(min_bid_ask_coverage=1.1),dict(max_nfev=1.5)]:
        with pytest.raises(ValueError):
            sp.CalibrationConfig(**kwargs)
    with pytest.raises(ValueError):
        sp.QuotePolicy(max_absolute_spread=np.inf)


def test_empty_template_loads_and_has_explicit_failed_data_gate():
    p=Path(__file__).resolve().parents[1]/'src'/'structured_products'/'data'/'market_surface'/'templates'
    d=sp.load_market_data(p)
    v=sp.validate_quotes(d.rows,d.snapshot,d.calendar,d.curves)
    f=sp.calibrate_ssvi(v)
    assert not f.accepted and not f.validation.records


def test_calibration_refuses_sparse_data_and_does_not_confuse_optimizer_with_acceptance():
    snap,cal,c,row = market_inputs()
    quotes = sp.validate_quotes([row],snap,cal,c)
    fit = sp.calibrate_ssvi(quotes)
    assert not fit.accepted and fit.surface is None
    assert any(g.name=='data_coverage' and not g.passed for g in fit.gates)
    json.dumps(fit.to_dict(),allow_nan=False)
