"""Reconstructable single-asset scenarios and full-revaluation surface risk."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
from itertools import product as cartesian_product
import json
from math import isfinite
from pathlib import Path

import numpy as np

from ..config import MCConfig, PDEConfig
from ..enums import BasketRule, CouponRule, SameDayPriority
from ..market import MarketData
from ..models import LocalVolModel, TimeVolModel, validate_model
from ..payoffs import VectorizedPayoffState, evaluate_path
from ..pricing import price
from ..products import (StructuredNote, CouponFeature, RedemptionFeature,
                        KnockInFeature, AutocallFeature)
from ..results import PricingResult, _jsonify
from ..engines.pde.grid import build_spot_grid
from .greeks import _roll_product


METRIC_UNITS = {
    'present_value': 'PV', 'delta': 'PV / spot', 'gamma': 'PV / spot^2',
    'atm_level_sensitivity': 'PV / ATM_level_multiplier',
    'rho_skew_sensitivity': 'PV / SSVI_rho',
}
CONVENTION = 'sticky_log_forward_moneyness; constant_spot_daily_close_ex_cash; frozen_dynamics_time_roll'


def _finite_payload(data):
    """Reject NaN/Infinity even in otherwise permissive legacy domain fields."""
    if isinstance(data, dict):
        for value in data.values():
            _finite_payload(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            _finite_payload(value)
    elif isinstance(data, (int, float, np.number)) and not isfinite(data):
        raise ValueError('scenario inputs and outputs must be finite')


def _audit_coordinates(data):
    """Invalid numeric requests use explicit nulls; each corresponding cell fails."""
    if isinstance(data, dict):
        return {key:_audit_coordinates(value) for key,value in data.items()}
    if isinstance(data, (list,tuple)):
        return [_audit_coordinates(value) for value in data]
    if isinstance(data, np.generic):
        return _audit_coordinates(data.item())
    if isinstance(data, (float,np.floating)) and not isfinite(data):
        return None
    return data if data is None or isinstance(data,(str,bool,int,float)) else repr(data)


def _product_dict(note):
    return None if note is None else _jsonify(asdict(note))


def _validate_product_booleans(note):
    flags = [note.coupon.survives_knock_in, note.redemption.include_principal]
    if note.knock_in is not None:
        flags.append(note.knock_in.initially_knocked_in)
    if any(type(flag) is not bool for flag in flags):
        raise ValueError('nested product flags must be boolean')


def _product_from_dict(data):
    if data is None:
        return None
    data = dict(data)
    data['reference_spots'] = tuple(data['reference_spots'])
    data['metadata'] = tuple(tuple(row) for row in data['metadata'])
    data['basket_rule'] = BasketRule(data['basket_rule'])
    data['same_day_priority'] = SameDayPriority(data['same_day_priority'])
    coupon = dict(data['coupon'])
    coupon['rule'] = CouponRule(coupon['rule'])
    for key in ('payment_days', 'rate_schedule'):
        coupon[key] = tuple(coupon[key])
    data['coupon'] = CouponFeature(**coupon)
    data['redemption'] = RedemptionFeature(**data['redemption'])
    data['knock_in'] = None if data['knock_in'] is None else KnockInFeature(**data['knock_in'])
    if data['autocall'] is not None:
        data['autocall'] = AutocallFeature(**{k: tuple(v) for k,v in data['autocall'].items()})
    return StructuredNote(**data)


@dataclass(frozen=True)
class ScenarioShock:
    """Spot is an absolute mark. Roll days are additional trading closes."""
    spot: float | None = None
    roll_days: int = 0
    level_multiplier: float = 1.
    rho_shift: float = 0.
    initial_knocked_in: bool | None = None

    def __post_init__(self):
        if (self.spot is not None and (not isfinite(self.spot) or self.spot <= 0)
                or not isfinite(self.level_multiplier) or self.level_multiplier <= 0
                or not isfinite(self.rho_shift)):
            raise ValueError('spot/ATM multiplier must be finite positive and rho shift finite')
        if type(self.roll_days) is not int or self.roll_days < 0:
            raise ValueError('roll_days must be a nonnegative integer')
        if self.initial_knocked_in is not None and type(self.initial_knocked_in) is not bool:
            raise ValueError('initial_knocked_in must be a boolean or null')


@dataclass(frozen=True)
class ScenarioCashflow:
    day: int
    amount: float
    kind: str
    base_present_value: float


@dataclass(frozen=True)
class Scenario:
    original_product: StructuredNote
    product: StructuredNote | None
    market: MarketData
    model: LocalVolModel | TimeVolModel
    shock: ScenarioShock
    initial_requested_knocked_in: bool
    realized_knocked_in: bool
    terminated: bool
    knocked_out: bool
    redemption_day: int | None
    cashflows: tuple[ScenarioCashflow, ...]
    discount_to_roll: float
    baseline_model_identity: str
    convention: str = CONVENTION

    def __post_init__(self):
        object.__setattr__(self, 'cashflows', tuple(self.cashflows))
        _validate_product_booleans(self.original_product)
        _finite_payload(_product_dict(self.original_product))
        if self.product is not None:
            _validate_product_booleans(self.product)
            _finite_payload(_product_dict(self.product))
        flags = (self.initial_requested_knocked_in, self.realized_knocked_in,
                 self.terminated, self.knocked_out)
        if any(type(flag) is not bool for flag in flags):
            raise ValueError('scenario lifecycle flags must be boolean')
        if not isfinite(self.discount_to_roll) or self.discount_to_roll <= 0:
            raise ValueError('discount_to_roll must be finite and positive')
        if self.terminated != (self.product is None) or self.knocked_out and not self.terminated:
            raise ValueError('terminated scenario cannot have future product exposure')
        initial = bool(self.original_product.knock_in and self.original_product.knock_in.initially_knocked_in)
        if self.initial_requested_knocked_in != initial:
            raise ValueError('requested history does not match original product history')
        if (self.initial_requested_knocked_in and not self.realized_knocked_in
                or self.product is not None and self.realized_knocked_in != bool(
                    self.product.knock_in and self.product.knock_in.initially_knocked_in)):
            raise ValueError('scenario realized history does not match future product history')
        days, maturity = self.shock.roll_days, self.original_product.maturity_days
        if not self.terminated:
            if days >= maturity:
                raise ValueError('alive remaining maturity is inconsistent with cumulative roll')
            expected = self.original_product if days == 0 else _roll_product(self.original_product, days)
            if expected.knock_in is not None:
                expected = replace(expected, knock_in=replace(expected.knock_in,
                    initially_knocked_in=self.realized_knocked_in))
            if self.product != expected or days == 0 and self.realized_knocked_in != initial:
                raise ValueError('remaining contract/history/accrual is inconsistent with cumulative roll')
            if self.redemption_day is not None:
                raise ValueError('alive scenario cannot have a redemption day')
        else:
            if type(self.redemption_day) is not int or not 1 <= self.redemption_day <= min(days, maturity):
                raise ValueError('redemption day must lie within observed contract history')
            if self.knocked_out:
                if self.original_product.autocall is None or self.redemption_day not in self.original_product.autocall.observation_days:
                    raise ValueError('knock-out redemption must occur on an observation day')
            elif self.redemption_day != maturity:
                raise ValueError('non-autocall redemption must occur at expiry')
        last_day = 0
        for cash in self.cashflows:
            bound = min(days, maturity, self.redemption_day if self.terminated else maturity)
            if type(cash.day) is not int or not 1 <= cash.day <= bound or cash.day < last_day:
                raise ValueError('cash day must be ordered and within observed/redemption history')
            if (not isfinite(cash.amount) or cash.amount < 0 or not isfinite(cash.base_present_value)
                    or cash.base_present_value < 0):
                raise ValueError('cash amount and base PV must be finite and nonnegative')
            if cash.kind == 'periodic_coupon':
                if cash.day not in self.original_product.coupon.payment_days:
                    raise ValueError('cash coupon day is absent from the original schedule')
            elif cash.kind in ('principal','autocall_coupon','maturity_coupon'):
                if not self.terminated or cash.day != self.redemption_day:
                    raise ValueError('redemption cash must occur at actual redemption')
            else:
                raise ValueError('unsupported cash kind')
            last_day = cash.day

    @property
    def cash_sum(self):
        return sum(c.amount for c in self.cashflows)

    @property
    def cash_pv(self):
        return sum(c.base_present_value for c in self.cashflows)

    def to_dict(self):
        data = dict(schema='surface_scenario_v1', original_product=_product_dict(self.original_product),
            product=_product_dict(self.product), market=_jsonify(asdict(self.market)), model=self.model.to_dict(),
            shock=asdict(self.shock), initial_requested_knocked_in=self.initial_requested_knocked_in,
            realized_knocked_in=self.realized_knocked_in, terminated=self.terminated,
            knocked_out=self.knocked_out, redemption_day=self.redemption_day,
            cashflows=[asdict(c) for c in self.cashflows], cash_sum=self.cash_sum, cash_pv=self.cash_pv,
            discount_to_roll=self.discount_to_roll, baseline_model_identity=self.baseline_model_identity,
            convention=self.convention)
        _finite_payload(data)
        return _jsonify(data)

    @classmethod
    def from_dict(cls, data):
        _finite_payload(data)
        if data['schema'] != 'surface_scenario_v1' or data['convention'] != CONVENTION:
            raise ValueError('unsupported scenario schema or convention')
        loaders = {'ssvi_dupire': LocalVolModel, 'time_volatility': TimeVolModel}
        try:
            loader = loaders[data['model']['kind']]
        except KeyError as exc:
            raise ValueError('unsupported scenario model') from exc
        node = cls(_product_from_dict(data['original_product']), _product_from_dict(data['product']),
            MarketData(**data['market']), loader.from_dict(data['model']), ScenarioShock(**data['shock']),
            data['initial_requested_knocked_in'], data['realized_knocked_in'], data['terminated'],
            data['knocked_out'], data['redemption_day'],
            tuple(ScenarioCashflow(**c) for c in data['cashflows']), data['discount_to_roll'],
            data['baseline_model_identity'], data['convention'])
        if node.terminated != (node.product is None):
            raise ValueError('terminated scenario cannot have future product exposure')
        validate_model(node.model, node.original_product, node.market)
        if node.cash_sum != data['cash_sum'] or node.cash_pv != data['cash_pv']:
            raise ValueError('inconsistent scenario cash totals')
        return node


def shock_model(model, level_multiplier=1., rho_shift=0.):
    """Rebuild Dupire after a validated theta*m^2 / additive-rho shock."""
    ScenarioShock(level_multiplier=level_multiplier, rho_shift=rho_shift)
    if level_multiplier == 1. and rho_shift == 0.:
        return model
    if not isinstance(model, LocalVolModel):
        raise ValueError('ATM/rho surface shocks require an SSVI LocalVolModel')
    surface = replace(model.surface, theta=tuple(t*level_multiplier**2 for t in model.surface.theta),
                      rho=model.surface.rho+rho_shift)
    diagnostic = surface.arbitrage_diagnostics()
    if diagnostic['min_density_factor'] <= 0 or diagnostic['min_w_t'] < 0:
        raise ValueError('shocked SSVI fails numerical density/calendar checks')
    return LocalVolModel.theoretical(surface, model.curves, time_offset=model.time_offset,
        anchor_spot=model.anchor_spot, observed_support=model.observed_support)


def transform_scenario(product, market, model, shock=None):
    """Apply an instantaneous mark, then actual daily closes at the marked spot.

    Contract elapsed days never set model time. Historical KI is unchanged at
    roll zero. Future diffusion is the original absolute coefficient field after
    rebasing, evaluated at the advanced market clock.
    """
    shock = shock or ScenarioShock()
    validate_model(model, product, market)
    _finite_payload(dict(product=_product_dict(product), market=asdict(market)))
    if shock.roll_days > product.maturity_days:
        raise ValueError('roll exceeds remaining maturity')
    initial = bool(product.knock_in and product.knock_in.initially_knocked_in)
    if shock.initial_knocked_in is not None:
        initial = shock.initial_knocked_in
        if initial and product.knock_in is None:
            raise ValueError('initial KI history requires a knock-in feature')
        if product.knock_in:
            product = replace(product, knock_in=replace(product.knock_in, initially_knocked_in=initial))
    baseline_identity = model.identity
    shifted = shock_model(model, shock.level_multiplier, shock.rho_shift)
    spot = market.spots[0] if shock.spot is None else shock.spot
    marked_market = replace(market, spots=(spot,))
    if spot != market.spots[0]:
        shifted = shifted.rebase(spot)
    state = VectorizedPayoffState(product, 1, 0.)
    performance = np.array([spot/product.reference_spots[0]])
    for day in range(1, shock.roll_days+1):
        state.process_day(day, performance)
        if not state.alive[0]:
            break
    terminated = not bool(state.alive[0])
    knocked_in = bool(state.knocked_in[0])
    cashflows = ()
    if shock.roll_days:
        # Canonical path evaluator records individual cash kinds and event order.
        evaluation = evaluate_path(np.full((product.maturity_days+1,1), spot), product)
        cashflows = tuple(ScenarioCashflow(c.day, c.amount, c.kind,
            c.amount*float(model.discount(c.day/product.day_count)))
            for c in evaluation.cashflows if c.day <= shock.roll_days)
    remaining = None if terminated else product
    if shock.roll_days and not terminated:
        remaining = _roll_product(product, shock.roll_days)
        if remaining.knock_in:
            remaining = replace(remaining, knock_in=replace(remaining.knock_in, initially_knocked_in=knocked_in))
    shifted = shifted.rolled(model.time_offset+shock.roll_days/product.day_count, spot)
    discount = float(model.discount(shock.roll_days/product.day_count))
    if not isfinite(discount) or discount <= 0:
        raise ValueError('invalid discount to scenario roll date')
    node = Scenario(product, remaining, marked_market, shifted, shock, initial, knocked_in,
        terminated, bool(state.knocked_out[0]), int(state.redemption_days[0]) if terminated else None,
        cashflows, discount, baseline_identity)
    _finite_payload(node.to_dict())
    return node


def price_scenario(scenario, method='pde', engine_config=None):
    """Future ex-cash PV in roll-date currency; paid cash is stored separately."""
    if scenario.terminated:
        return PricingResult(method=str(getattr(method, 'value', method)), present_value=0., net_value=0.,
            standard_error=0. if method == 'mc' else None,
            confidence_interval=(0.,0.) if method == 'mc' else None,
            knock_in_probability=float(scenario.realized_knocked_in),
            knock_out_probability=float(scenario.knocked_out), expected_redemption_time=0.,
            expected_coupon_cashflow=0., diagnostics={'terminated': True})
    return price(scenario.product, scenario.market, method, engine_config, model=scenario.model)


def advance_scenario(scenario, shock=None):
    """Continue an existing scenario without resetting history or paying cash twice.

    Recorded cashflow days and PVs always remain relative to the first base date.
    Already redeemed notes can advance beyond expiry, with zero future exposure.
    """
    shock = shock or ScenarioShock()
    if shock.initial_knocked_in is not None:
        raise ValueError('continuation cannot reset initial knock-in history')
    total_shock = ScenarioShock(spot=scenario.market.spots[0] if shock.spot is None else shock.spot,
        roll_days=scenario.shock.roll_days+shock.roll_days,
        level_multiplier=scenario.shock.level_multiplier*shock.level_multiplier,
        rho_shift=scenario.shock.rho_shift+shock.rho_shift,
        initial_knocked_in=scenario.initial_requested_knocked_in)
    if scenario.terminated:
        model = shock_model(scenario.model,shock.level_multiplier,shock.rho_shift)
        if total_shock.spot != scenario.market.spots[0]:
            model = model.rebase(total_shock.spot)
        discount = scenario.discount_to_roll*float(scenario.model.discount(shock.roll_days/scenario.original_product.day_count))
        return replace(scenario,market=replace(scenario.market,spots=(total_shock.spot,)),
            model=model.rolled(model.time_offset+shock.roll_days/scenario.original_product.day_count,total_shock.spot),
            shock=total_shock,discount_to_roll=discount)
    next_node = transform_scenario(scenario.product,scenario.market,scenario.model,shock)
    new_cash = tuple(ScenarioCashflow(c.day+scenario.shock.roll_days,c.amount,c.kind,
        c.base_present_value*scenario.discount_to_roll) for c in next_node.cashflows)
    return replace(next_node,original_product=scenario.original_product,shock=total_shock,
        initial_requested_knocked_in=scenario.initial_requested_knocked_in,
        redemption_day=None if next_node.redemption_day is None else next_node.redemption_day+scenario.shock.roll_days,
        cashflows=scenario.cashflows+new_cash,discount_to_roll=scenario.discount_to_roll*next_node.discount_to_roll,
        baseline_model_identity=scenario.baseline_model_identity)


def _difference_coordinates(center, step, *, squared=False):
    """A positive requested step need not change a float or yield a denominator."""
    center, step = float(center), float(step)
    high, low = center+step, center-step
    denominator = step*step if squared else 2*step
    if (not isfinite(step) or step <= 0 or not isfinite(high) or not isfinite(low)
            or high == center or low == center or high == low
            or not isfinite(denominator) or denominator <= 0):
        raise ValueError('bump must change both coordinates with a finite nonzero denominator')
    return high, low, denominator


def scenario_risk(scenario, *, metrics=None, pde_config=None, spot_bump=.01,
                  level_bump=.01, rho_bump=.01, _cache=None):
    """Centered derivatives; bumps change future inputs with history held fixed."""
    selected = tuple(METRIC_UNITS) if metrics is None else tuple(dict.fromkeys(metrics))
    if not selected or set(selected)-METRIC_UNITS.keys():
        raise ValueError('metrics must select known surface risk fields')
    config = pde_config or PDEConfig()
    if config.spot_grid_upper is None:
        # Invalid bumps have no valuation spot; report their metric error below.
        margin = spot_bump if isfinite(spot_bump) and 0 < spot_bump < 1 else 0.
        high_market = replace(scenario.market, spots=(scenario.market.spots[0]*(1+margin),))
        upper = float(build_spot_grid(scenario.original_product, high_market, config)[-1])
        config = replace(config, spot_grid_upper=upper)
    # Check even deterministic/terminal nodes against an explicitly supplied domain.
    build_spot_grid(scenario.original_product, scenario.market, config)
    cache = {} if _cache is None else _cache

    def value(node):
        if node.terminated:
            return 0.
        build_spot_grid(node.product, node.market, config)
        key = (node.product, node.market, node.model.identity, config)
        if key not in cache:
            cache[key] = price_scenario(node, 'pde', config).present_value
        result = cache[key]
        if not isfinite(result):
            raise ValueError('nonfinite scenario price')
        return result

    values, statuses = {}, {}
    base = value(scenario)
    spot_values = None
    for metric in selected:
        try:
            if metric == 'present_value':
                result = base
            elif metric in ('delta', 'gamma'):
                if not isfinite(spot_bump) or not 0 < spot_bump < 1:
                    raise ValueError('relative spot bump must be finite and in (0, 1)')
                step = scenario.market.spots[0]*spot_bump
                high, low, denominator = _difference_coordinates(scenario.market.spots[0], step,
                    squared=metric == 'gamma')
                if scenario.terminated:
                    result = 0.
                elif spot_values is None:
                    bumped = []
                    for spot in (high,low):
                        node = replace(scenario, market=replace(scenario.market, spots=(spot,)),
                                       model=scenario.model.rebase(spot))
                        bumped.append(value(node))
                    spot_values = bumped
                if not scenario.terminated:
                    up,down = spot_values
                    result = (up-down)/denominator if metric == 'delta' else (up-2*base+down)/denominator
            else:
                bump = level_bump if metric == 'atm_level_sensitivity' else rho_bump
                if not isfinite(bump) or bump <= 0:
                    raise ValueError('surface sensitivity bump must be finite positive')
                center = 1. if metric == 'atm_level_sensitivity' else getattr(getattr(scenario.model,'surface',None),'rho',0.)
                _, _, denominator = _difference_coordinates(center,bump)
                prices = []
                for sign in (1,-1):
                    model = shock_model(scenario.model, **({'level_multiplier':1+sign*bump}
                        if metric == 'atm_level_sensitivity' else {'rho_shift':sign*bump}))
                    prices.append(0. if scenario.terminated else value(replace(scenario,model=model)))
                result = (prices[0]-prices[1])/denominator
            if not isfinite(result):
                raise ValueError('nonfinite risk metric')
            values[metric], statuses[metric] = float(result), {'status':'ok'}
        except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
            values[metric], statuses[metric] = None, {'status':'error', 'error':str(exc)}
    return dict(metrics=values, metric_status=statuses, units={k:METRIC_UNITS[k] for k in selected},
        diagnostics=dict(spot_grid_upper=config.spot_grid_upper, engine=asdict(config), bumps=_audit_coordinates(dict(relative_spot=spot_bump,
            atm_level_multiplier=level_bump,ssvi_rho=rho_bump)), convention=CONVENTION,
            model=scenario.model.diagnostics(), atm_level_basis='local multiplier on scenario theta',
            unique_valuations=len(cache)), cash_sum=scenario.cash_sum, cash_pv=scenario.cash_pv,
        discount_to_roll=scenario.discount_to_roll,
        total_base_date_value=scenario.cash_pv+scenario.discount_to_roll*base)


def default_risk_axes(product):
    ratios = set(round(.60+i*.05, 10) for i in range(13))
    levels = []
    if product.knock_in:
        levels.append(product.knock_in.barrier_ratio)
    if product.autocall:
        levels.extend(product.autocall.barrier_ratios)
    for level in levels:
        ratios.update(round(level+shift,10) for shift in (-.005,0.,.005) if level+shift > 0)
    return dict(spot_ratios=tuple(sorted(ratios)),rho_shifts=(-.1,0.,.1),
        roll_days=tuple(sorted({d for d in (0,21,63,126,189,product.maturity_days)
                                if d<=product.maturity_days})),initial_ki=(False,True))


@dataclass(frozen=True)
class RiskSurfaceResult:
    cells: tuple[dict, ...]
    mc_checkpoints: tuple[dict, ...]
    diagnostics: dict

    def __iter__(self):
        return iter(self.cells)

    def to_dict(self):
        result = _jsonify(asdict(self))
        _finite_payload(result)
        return result

    def write_json(self, path):
        Path(path).write_text(json.dumps(self.to_dict(),indent=2,allow_nan=False)+'\n',encoding='utf-8')

    def write_csv(self, path):
        """Each CSV row includes full JSON scenario inputs for independent repricing."""
        rows = []
        for cell in self.cells:
            row = {k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list)) else v
                   for k,v in cell.items() if k not in ('metrics','metric_status','units','diagnostics')}
            row.update(cell.get('metrics', {}))
            for key in ('metric_status','units','diagnostics'):
                row[key] = json.dumps(cell.get(key,{}),allow_nan=False)
            rows.append(row)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with Path(path).open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def risk_surface(product, market, model, *, spot_ratios=None, rho_shifts=None, roll_days=None,
                 initial_ki=None, metrics=None, pde_config=None, mc_checkpoints=(),mc_config=None,
                 progress=None, spot_bump=.01,level_bump=.01,rho_bump=.01):
    """PDE Cartesian grid with explicit independent MC checkpoint scenarios.

    ``progress(completed, total, cell)`` fires after every cell, including failures.
    Bump prices are cached across identical nodes; calibration is never repeated.
    """
    validate_model(model,product,market)
    defaults=default_risk_axes(product)
    axes = dict(spot_ratios=tuple(defaults['spot_ratios'] if spot_ratios is None else spot_ratios),
        rho_shifts=tuple(defaults['rho_shifts'] if rho_shifts is None else rho_shifts),
        roll_days=tuple(d for d in (defaults['roll_days'] if roll_days is None else roll_days)
                        if type(d) is not int or d<=product.maturity_days),
        initial_ki=tuple(defaults['initial_ki'] if initial_ki is None else initial_ki))
    config=pde_config or PDEConfig()
    reference=product.reference_spots[0]
    levels=[reference,reference*product.redemption.strike_ratio]
    if product.knock_in is not None:
        levels.append(reference*product.knock_in.barrier_ratio)
    if product.autocall is not None:
        levels.extend(reference*b for b in product.autocall.barrier_ratios)
    coordinates, limits = [], []
    margin = spot_bump if isfinite(spot_bump) and 0 < spot_bump < 1 else 0.
    for ratio in axes['spot_ratios']:
        try:
            if isinstance(ratio,(bool,np.bool_)) or not isinstance(ratio,(int,float,np.integer,np.floating)):
                raise ValueError('spot ratio must be a finite positive real number')
            spot=float(ratio)*reference
            if not isfinite(spot) or spot <= 0:
                raise ValueError('derived spot must be finite positive')
            if config.spot_grid_upper is None:
                high=spot*(1+margin)
                upper=config.upper_spot_multiple*max(high,*levels)
                if not isfinite(high) or not isfinite(upper) or upper <= 0:
                    raise ValueError('derived bump spot/common domain is outside finite range')
                limits.append(upper)
            coordinates.append((ratio,spot,None))
        except (ValueError,TypeError,ArithmeticError) as exc:
            coordinates.append((ratio,None,str(exc)))
    if config.spot_grid_upper is None:
        config=replace(config,spot_grid_upper=max(limits,default=config.upper_spot_multiple*max(levels)))
    specs=list(cartesian_product(coordinates,axes['rho_shifts'],axes['roll_days'],axes['initial_ki']))
    cells=[]
    cache={}
    for index,((ratio,spot,coordinate_error),rho,days,ki) in enumerate(specs,1):
        # Invalid nonfinite coordinates remain auditable without invalid JSON tokens.
        request=_audit_coordinates(dict(spot_ratio=ratio,rho_shift=rho,
                     roll_days=days,initial_knocked_in=ki))
        cell=dict(request=request)
        try:
            if coordinate_error is not None:
                raise ValueError(coordinate_error)
            node=transform_scenario(product,market,model,ScenarioShock(
                spot=spot,rho_shift=rho,roll_days=days,initial_knocked_in=ki))
            cell['scenario']=node.to_dict()
            cell.update(scenario_risk(node,metrics=metrics,pde_config=config,spot_bump=spot_bump,
                                     level_bump=level_bump,rho_bump=rho_bump,_cache=cache))
            cell['status']='ok' if all(s['status']=='ok' for s in cell['metric_status'].values()) else 'partial'
        except (ValueError,TypeError,RuntimeError,OverflowError) as exc:
            cell.update(status='error',error=str(exc))
        cells.append(cell)
        if progress is not None:
            progress(index,len(specs),cell)
    checkpoints=[]
    for shock in mc_checkpoints:
        entry=dict(request=asdict(shock))
        try:
            node=transform_scenario(product,market,model,shock)
            entry.update(status='ok',scenario=node.to_dict(),price=price_scenario(node,'mc',mc_config or MCConfig()).to_dict())
        except (ValueError,TypeError,RuntimeError,OverflowError) as exc:
            entry.update(status='error',error=str(exc))
        checkpoints.append(entry)
    return RiskSurfaceResult(tuple(cells),tuple(checkpoints),dict(axes=_audit_coordinates(axes),
        spot_grid_upper=config.spot_grid_upper,engine=asdict(config),baseline_model=model.diagnostics(),
        convention=CONVENTION,unique_valuations=len(cache),
        bumps=_audit_coordinates(dict(relative_spot=spot_bump,atm_level_multiplier=level_bump,ssvi_rho=rho_bump))))
