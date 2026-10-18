from dataclasses import replace
from math import exp, sqrt

import numpy as np
import pytest
from importlib.resources import files

import structured_products as sp
from structured_products.engines.mc import _daily_independent_normals


def curves(rate=.03, carry=.01):
    return sp.MarketCurves((0., .25, 1., 2.),
        tuple(100*exp(carry*t) for t in (0., .25, 1., 2.)),
        tuple(exp(-rate*t) for t in (0., .25, 1., 2.)))


def market(vol=.2):
    return sp.MarketData((100.,), .03, (.02,), (vol,))


def bond_put(days=126, strike=1.):
    return sp.StructuredNote((100.,), 1000., days,
        sp.CouponFeature(sp.CouponRule.CONTINGENT_AT_REDEMPTION),
        sp.RedemptionFeature(strike_ratio=strike),
        knock_in=sp.KnockInFeature(.01, initially_knocked_in=True))


def test_constant_model_mc_flat_reduction_and_pair_error():
    product = bond_put(days=21)
    config = sp.MCConfig(n_paths=1000, seed=9)
    model = sp.TimeVolModel(curves(), (0.,), (.2,))
    flat = sp.price(product, market(), 'mc', config)
    local = sp.price(product, market(), 'mc', config, model=model)
    assert local.present_value == pytest.approx(flat.present_value, abs=2e-11)
    assert local.standard_error == pytest.approx(flat.standard_error, abs=2e-11)
    spots = np.full(config.n_paths, 100.)
    for i in range(21):
        z = _daily_independent_normals(seed=9, stream_index=i,
            n_paths=1000, n_assets=1, antithetic=True)[:, 0]
        spots *= np.exp((.01-.5*.2**2)/252+.2/sqrt(252)*z)
    values = 1000*np.minimum(spots/100, 1)*exp(-.03*21/252)
    pairs = (values[:500]+values[500:])/2
    assert local.standard_error == pytest.approx(np.std(pairs, ddof=1)/sqrt(500))


def test_dupire_flat_limit_and_clock():
    surface = sp.SSVISurface((.25, 1.), (.01, .04), -.4, 0.)
    model = sp.LocalVolModel.theoretical(surface, curves())
    assert model.local_vol(0., np.array([1e-30, 100., 1e30])) == pytest.approx([.2]*3)
    assert model.local_vol(.7, 90.) == pytest.approx(.2)
    rebuilt = sp.LocalVolModel.from_dict(model.to_dict())
    assert rebuilt.identity == model.identity
    rolled = model.rolled(.5, 90.)
    assert rolled.discount(.5) == pytest.approx(exp(-.03*.5))
    with pytest.raises(ValueError, match='anchor'):
        sp.price(bond_put(21), market(), 'mc', sp.MCConfig(n_paths=10), model=rolled)


def test_time_vol_pde_black_and_offset():
    model = sp.TimeVolModel(curves(), (0., .1, .3), (.1, .4, .2), time_offset=.2)
    product = bond_put(126)
    variance = .4**2*.1+.2**2*.4
    expected = 1000*exp(-.03*.5)-10*sp.black_price(100*exp(.01*.5),100,exp(-.03*.5),variance,'put')
    result = sp.price(product, market(0.), 'pde', sp.PDEConfig(spot_grid_points=1601, time_steps_per_day=5), model=model)
    assert result.present_value == pytest.approx(expected, abs=.06)
    assert result.diagnostics['model']['time_offset'] == .2


def test_model_rejects_greeks_and_single_pair_uncertainty():
    model = sp.TimeVolModel(curves(), (0.,), (.2,))
    with pytest.raises(ValueError, match='surface-risk'):
        sp.price(bond_put(21), market(), 'mc', model=model, include_greeks=True)
    result = sp.price(bond_put(21), market(), 'mc', sp.MCConfig(n_paths=2), model=model)
    assert result.standard_error is None
    assert result.confidence_interval is None
    assert 'one antithetic pair' in ' '.join(result.warnings)


@pytest.fixture(scope='module')
def smile():
    surface = sp.SSVISurface((.25, .5, 1.), (.013, .024, .045), -.45, .65)
    return sp.LocalVolModel.theoretical(surface, curves())


@pytest.mark.parametrize('days', [63, 126, 252])
@pytest.mark.parametrize('strike', [.9, 1., 1.1])
def test_nonflat_ssvi_vanilla_recovery_both_engines(smile, days, strike):
    product = bond_put(days, strike)
    t, k = days/252, strike*100
    variance = smile.surface.total_variance(np.log(k/smile.curves.forward(t)), t)
    expected = 1000*smile.discount(t)-1000/k*sp.black_price(
        smile.curves.forward(t), k, smile.discount(t), variance, 'put')
    coarse = sp.price(product, market(), 'pde', sp.PDEConfig(spot_grid_points=801), model=smile)
    fine = sp.price(product, market(), 'pde', sp.PDEConfig(spot_grid_points=1601, time_steps_per_day=2), model=smile)
    grid_allowance = 2*abs(fine.present_value-coarse.present_value)
    assert abs(fine.present_value-expected) < .045
    mc = sp.price(product, market(), 'mc', sp.MCConfig(n_paths=20000, steps_per_day=2), model=smile)
    assert abs(mc.present_value-expected) <= 3*mc.standard_error+grid_allowance
    print(f'vanilla days={days} K={k} black={expected:.8f} pde={fine.present_value:.8f} '
          f'grid_allowance={grid_allowance:.8f} mc={mc.present_value:.8f} se={mc.standard_error:.8f}')


@pytest.mark.parametrize('builder', [sp.make_classic_snowball, sp.make_standard_fcn, sp.make_stepdown_snowball])
@pytest.mark.parametrize('initial_ki', [False, True])
def test_nonflat_product_cross_engine_refinement(smile, builder, initial_ki):
    product = builder(maturity_days=42)
    if product.knock_in:
        product = replace(product, knock_in=replace(product.knock_in, initially_knocked_in=initial_ki))
    # Both marks sit near contractual barriers; use a fixed common domain.
    stressed = smile.rebase(99.)
    marked = replace(market(), spots=(99.,))
    coarse = sp.price(product, marked, 'pde', sp.PDEConfig(spot_grid_points=801, spot_grid_upper=400.), model=stressed)
    fine = sp.price(product, marked, 'pde', sp.PDEConfig(spot_grid_points=1601, time_steps_per_day=2, spot_grid_upper=400.), model=stressed)
    mc = sp.price(product, marked, 'mc', sp.MCConfig(n_paths=40000, steps_per_day=2), model=stressed)
    allowance = 2*abs(fine.present_value-coarse.present_value)
    assert abs(mc.present_value-fine.present_value) <= 3*mc.standard_error+allowance
    print(f'product={product.product_name} ki={initial_ki} pde={fine.present_value:.8f} '
          f'allowance={allowance:.8f} mc={mc.present_value:.8f} se={mc.standard_error:.8f}')


def test_mc_time_refinement_against_smile_black(smile):
    product = bond_put(63, 1.1)
    t, k = .25, 110.
    w = smile.surface.total_variance(np.log(k/smile.curves.forward(t)), t)
    expected = 1000*smile.discount(t)-1000/k*sp.black_price(smile.curves.forward(t), k, smile.discount(t), w, 'put')
    # Independent seeds give an empirical bound on refinement noise as well as bias.
    results = [sp.price(product, market(), 'mc', sp.MCConfig(n_paths=60000,
        steps_per_day=steps, seed=seed), model=smile) for steps, seed in [(1, 17), (4, 23)]]
    coarse, fine = results
    noise = 3*sqrt(coarse.standard_error**2+fine.standard_error**2)
    measured_allowance = abs(coarse.present_value-fine.present_value)
    assert abs(fine.present_value-expected) <= 3*fine.standard_error+measured_allowance
    print(f'mc_refinement black={expected:.8f} daily={coarse.present_value:.8f} '
          f'quarter_day={fine.present_value:.8f} joint_3se={noise:.8f} measured_substep_difference={measured_allowance:.8f}')


def test_time_vol_chronology_curve_cash_and_elapsed():
    # KI/KO at intermediate dates makes volatility ordering material, unlike a vanilla.
    product = sp.make_classic_snowball(maturity_days=42, observation_days=(21, 42), knock_in_ratio=.95)
    c = sp.MarketCurves((0., 21/252, 1.), (100., 100., 104.), (1., .98, .94))
    model = sp.TimeVolModel(c, (0., 21/252), (.5, .05))
    pde = sp.price(product, market(), 'pde', sp.PDEConfig(spot_grid_points=1601, time_steps_per_day=2), model=model)
    coarse = sp.price(product, market(), 'pde', sp.PDEConfig(spot_grid_points=801), model=model)
    mc = sp.price(product, market(), 'mc', sp.MCConfig(n_paths=60000, steps_per_day=2), model=model)
    assert abs(pde.present_value-mc.present_value) <= 3*mc.standard_error+2*abs(pde.present_value-coarse.present_value)
    reversed_model = replace(model, volatilities=(.05, .5))
    reversed_pv = sp.price(product, market(), 'pde', sp.PDEConfig(spot_grid_points=1601), model=reversed_model).present_value
    assert abs(pde.present_value-reversed_pv) > 2.
    rolled = model.rolled(21/252, 100.)
    elapsed = replace(product, elapsed_days=63)
    regular = sp.price(product, market(), 'mc', sp.MCConfig(n_paths=1000), model=rolled)
    accrued = sp.price(elapsed, market(), 'mc', sp.MCConfig(n_paths=1000), model=rolled)
    assert accrued.present_value > regular.present_value
    assert accrued.diagnostics['model']['time_offset'] == 21/252
    deterministic = sp.TimeVolModel(c, (0.,), (0.,))
    result = sp.price(product, market(.8), 'mc', sp.MCConfig(n_paths=10), model=deterministic)
    expected = 1000*(1+product.autocall.coupon_rates[0]*21/252)*.98
    assert result.present_value == pytest.approx(expected)


def test_model_domain_roll_rebase_immutability(smile):
    for t in [0., 1e-15, .2, 5.]:
        assert np.all(np.isfinite(smile.local_vol(t, [1e-100, 100., 1e100])))
    for t, spot in [(-1, 100), (np.nan,100), (0,0), (.5,-1), (.5,np.inf)]:
        with pytest.raises(ValueError):
            smile.local_vol(t, spot)
    rolled = smile.rolled(.4, 90.)
    assert rolled.local_vol(.1, 90.) == pytest.approx(smile.local_vol(.5, 90.))
    shifted = smile.rebase(90.)
    assert shifted.local_vol(.5, 90.) == pytest.approx(smile.local_vol(.5, 100.))
    assert smile.anchor_spot == 100.
    assert sp.TimeVolModel.from_dict(sp.TimeVolModel(curves(), [0.], [.2]).to_dict()).identity
    with pytest.raises(ValueError, match='day_count'):
        sp.price(replace(bond_put(21), day_count=365), market(), 'pde', model=smile)
    with pytest.raises(ValueError, match='single-asset'):
        sp.price(sp.make_worst_of_snowball(reference_spots=(100.,100.)),
            sp.MarketData((100.,100.), .03, (0.,0.), (.2,.2), ((1.,0.),(0.,1.))), 'mc', model=smile)
    payload = smile.to_dict()
    payload['surface']['theta'][0] = 99.
    assert smile.surface.theta[0] == .013


def test_calibration_reload_revalidates_and_detaches():
    data = sp.load_market_data(files('structured_products').joinpath('data/market_surface'))
    validated = sp.validate_quotes(data.rows, data.snapshot, data.calendar, data.curves)
    fit = sp.calibrate_ssvi(validated)
    saved = fit.to_dict()
    saved['optimizer']['starts'][0]['cost'] = -99
    saved['optimizer']['arbitrage']['min_density_factor'] = -99
    assert fit.to_dict()['optimizer']['starts'][0]['cost'] >= 0
    assert fit.to_dict()['optimizer']['arbitrage']['min_density_factor'] >= 0
    model = sp.LocalVolModel.from_calibration(fit)
    assert sp.LocalVolModel.from_dict(model.to_dict()).identity == model.identity
    saved = fit.to_dict()
    saved['accepted'] = True
    saved['validation']['records'] = saved['validation']['records'][:5]
    rejected = sp.CalibrationResult.from_dict(saved)
    assert not rejected.accepted
    with pytest.raises(ValueError, match='accepted calibration'):
        sp.LocalVolModel.from_calibration(rejected)
    payload = model.to_dict()
    payload['calibration'] = saved
    with pytest.raises(ValueError, match='revalidation'):
        sp.LocalVolModel.from_dict(payload)


def test_local_model_batch_invariance_and_flat_pde_reduction():
    product = bond_put(42)
    model = sp.TimeVolModel(curves(), (0.,), (.2,))
    a = sp.price(product, market(), 'mc', sp.MCConfig(n_paths=1000, batch_size=73), model=model)
    b = sp.price(product, market(), 'mc', sp.MCConfig(n_paths=1000, batch_size=511), model=model)
    assert a.present_value == b.present_value
    assert a.standard_error == b.standard_error
    config = sp.PDEConfig(spot_grid_points=1601, time_steps_per_day=4)
    a = sp.price(product, market(), 'pde', config)
    b = sp.price(product, market(), 'pde', config, model=model)
    # Daily KI projection smoothing adds BE half steps only in the model branch;
    # temporal refinement reduces its first-order diffusion error below .004.
    assert a.present_value == pytest.approx(b.present_value, abs=.004)


def test_model_validation_and_no_hidden_variance_repairs(smile, monkeypatch):
    for kwargs in [dict(times=(.1,), volatilities=(.2,)),
                   dict(times=(0.,), volatilities=(-.2,)),
                   dict(times=(0.,), volatilities=(np.nan,)),
                   dict(times=(0.,), volatilities=(.2,), time_offset=-.1),
                   dict(times=(0.,), volatilities=(.2,), anchor_spot=np.inf)]:
        with pytest.raises(ValueError):
            sp.TimeVolModel(curves(), **kwargs)
    with pytest.raises(ValueError):
        smile.rebase(0.)
    for field, value in [('w_t', -1e-16), ('w_kk', -100.), ('w', -1.)]:
        with monkeypatch.context() as patch:
            original = sp.SSVISurface.derivatives
            patch.setattr(sp.SSVISurface, 'derivatives', lambda self, k, t:
                replace(original(self, k, t), **{field: value}))
            with pytest.raises(ValueError, match='Dupire'):
                smile.local_vol(.5, 100.)
    class InvalidModel:
        anchor_spot, time_offset, day_count = 100., 0., 'ACT/252'
        def discount(self, t): return exp(-.03*t)
        def forward_growth(self, a, b): return 1.
        def local_vol(self, t, s): return -1.
    for method, config in [('mc', sp.MCConfig(n_paths=10)), ('pde', sp.PDEConfig())]:
        with pytest.raises(ValueError, match='invalid local volatility'):
            sp.price(bond_put(2), market(), method, config, model=InvalidModel())


def test_fair_coupon_uses_model_and_relative_curve_discount():
    product = sp.make_classic_snowball(maturity_days=21, observation_days=(21,))
    model = sp.TimeVolModel(curves(.1, .01), (0.,), (0.,)).rolled(.5, 100.)
    result = sp.solve_fair_coupon(product, market(), 1000., 'mc', sp.MCConfig(n_paths=10),
        0., .3, model=model)
    assert result.coupon_rate == pytest.approx((exp(.1*21/252)-1)/(21/252))


def test_pde_midpoints_rannacher_and_interior_spots():
    samples = []
    class RecordingModel:
        anchor_spot, time_offset, day_count = 100., .5, 'ACT/252'
        def discount(self, t): return exp(-.03*t)
        def forward_growth(self, a, b): return exp(.01*(b-a))
        def local_vol(self, t, s):
            assert np.min(s)>0
            samples.append(t*252)
            return .2
        def diagnostics(self): return {'kind':'recording_test'}
    sp.price(bond_put(2), market(), 'pde',
        sp.PDEConfig(spot_grid_points=101, time_steps_per_day=2), model=RecordingModel())
    assert samples == pytest.approx([1.875,1.625,1.25,.875,.625,.25])


def test_dupire_zero_time_right_limit_and_zero_forward_variance(smile):
    spots = [30., 100., 300.]
    assert smile.local_vol(0., spots) == pytest.approx(smile.local_vol(1e-12, spots), rel=1e-7)
    plateau = sp.LocalVolModel.theoretical(sp.SSVISurface((.25,.5),(.01,.01),-.4,.6), curves())
    assert plateau.local_vol(.7, spots) == pytest.approx([0.,0.,0.])


@pytest.mark.parametrize('bad', [0., -1., np.inf, np.nan])
def test_engine_rejects_invalid_curve_discounts(bad):
    class BadDiscountModel:
        anchor_spot, time_offset, day_count = 100., 0., 'ACT/252'
        def discount(self, t): return bad
        def forward_growth(self, a, b): return 1.
        def local_vol(self, t, s): return .2
        def diagnostics(self): return {}
    for method, config in [('mc', sp.MCConfig(n_paths=10)), ('pde', sp.PDEConfig())]:
        with pytest.raises(ValueError, match='discount'):
            sp.price(bond_put(2), market(), method, config, model=BadDiscountModel())


def test_explicit_deterministic_pde_inclusive_ko_and_curve_discounts():
    carry = np.log(100/99)/(21/252)
    times = (0.,21/252,42/252,1.)
    c = sp.MarketCurves(times, (99.,100.,100*exp(carry*21/252),99*exp(carry)),
        tuple(exp(-.03*t) for t in times))
    model = sp.TimeVolModel(c, (0.,), (0.,), anchor_spot=99.)
    marked = replace(market(.2), spots=(99.,))
    product = sp.make_classic_snowball(maturity_days=42, observation_days=(21,42))
    expected = 1000*(1+product.autocall.coupon_rates[0]*21/252)*exp(-.03*21/252)
    mc = sp.price(product, marked, 'mc', sp.MCConfig(n_paths=10), model=model)
    pde = sp.price(product, marked, 'pde', sp.PDEConfig(spot_grid_points=401,
        time_steps_per_day=2, spot_grid_upper=400.), model=model)
    assert mc.present_value == pytest.approx(expected, abs=1e-10)
    assert pde.present_value == pytest.approx(expected, abs=1e-10)
    assert pde.diagnostics['deterministic_zero_volatility']


def test_theoretical_and_direct_constructor_cannot_forge_accepted_provenance(smile):
    for constructor in (sp.LocalVolModel, sp.LocalVolModel.theoretical):
        with pytest.raises(ValueError, match='calibration|theoretical'):
            constructor(smile.surface, smile.curves, calibration_json='{}')
    rebuilt = sp.LocalVolModel.from_dict(smile.to_dict())
    assert rebuilt.identity == smile.identity
    assert rebuilt.diagnostics()['provenance'] == 'research/theoretical'


def test_calibrated_constructor_rejects_surface_replacement_without_refit():
    data = sp.load_market_data(files('structured_products').joinpath('data/market_surface'))
    fit = sp.calibrate_ssvi(sp.validate_quotes(data.rows,data.snapshot,data.calendar,data.curves))
    model = sp.LocalVolModel.from_calibration(fit)
    with pytest.raises(ValueError, match='calibration|surface'):
        replace(model, surface=replace(model.surface, rho=model.surface.rho+.01))


@pytest.mark.parametrize('kind', ['time_vol', 'dupire_plateau'])
def test_remaining_deterministic_horizon_daily_ki_coupon_and_roll_discounts(kind):
    offset = .25
    times = (0.,offset,offset+7/252,offset+14/252,offset+21/252,1.)
    discounts = (1.,.99,.985,.98,.97,.94)
    c = sp.MarketCurves(times, (100.,100.,101.,95.,97.,98.), discounts)
    if kind == 'time_vol':
        model = sp.TimeVolModel(c,(0.,offset),(.4,0.)).rolled(offset,100.)
    else:
        surface = sp.SSVISurface((.1,.25,.5,1.),(.01,.02,.02,.04),-.4,.6)
        model = sp.LocalVolModel.theoretical(surface,c).rolled(offset,100.)
    assert model.is_deterministic(21/252)
    assert not model.rolled(0.,100.).is_deterministic(offset+21/252)
    product = sp.StructuredNote((100.,),1000.,21,
        sp.CouponFeature(sp.CouponRule.FIXED_PERIODIC,(7,14,21),(.12,.12,.12)),
        sp.RedemptionFeature(),knock_in=sp.KnockInFeature(.98))
    expected = 1000*.12*7/252*.985/.99 + 970*.97/.99
    result = sp.price(product,market(.8),'pde',model=model)
    assert result.present_value == pytest.approx(expected,abs=1e-10)
    assert result.diagnostics['deterministic_zero_volatility']
    assert result.diagnostics['model']['time_offset'] == offset


def test_zero_initial_vol_but_future_nonzero_does_not_take_deterministic_shortcut():
    model = sp.TimeVolModel(curves(),(0.,7/252),(0.,.4))
    assert model.is_deterministic(7/252)
    assert not model.is_deterministic(14/252)
    result = sp.price(bond_put(14),market(0.),'pde',sp.PDEConfig(spot_grid_points=401),model=model)
    assert not result.diagnostics.get('deterministic_zero_volatility',False)


def test_accepted_roll_rebase_keep_proof_without_recalibration(monkeypatch):
    data = sp.load_market_data(files('structured_products').joinpath('data/market_surface'))
    fit = sp.calibrate_ssvi(sp.validate_quotes(data.rows,data.snapshot,data.calendar,data.curves))
    model = sp.LocalVolModel.from_calibration(fit)
    original_identity = model.identity
    def forbidden(*args, **kwargs):
        raise AssertionError('roll/rebase should not rerun calibration')
    monkeypatch.setattr(sp.CalibrationResult,'from_dict',forbidden)
    for moved in (model.rolled(.2,90.),model.rebase(90.),model.rolled(.2,90.).rebase(80.)):
        assert moved.diagnostics()['provenance'] == 'accepted calibration'
        assert moved._validated_calibration is model._validated_calibration
    assert model.identity == original_identity
