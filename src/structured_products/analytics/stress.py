"""Full revaluation stresses with explicit event histories and holder value dates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from itertools import product as cartesian_product
import json
from math import isfinite
from pathlib import Path

from ..config import MCConfig, PDEConfig
from ..results import _jsonify
from .surface_risk import (ScenarioShock, RiskSurfaceResult, transform_scenario,
    advance_scenario, price_scenario, scenario_risk, _product_dict, _audit_coordinates)


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*.py'))}


@dataclass(frozen=True)
class StressSpec:
    name: str
    shock: ScenarioShock = ScenarioShock()
    observation: str = 'mark'
    close_history: tuple[float,...] = ()

    def __post_init__(self):
        object.__setattr__(self,'close_history',tuple(self.close_history))
        if not self.name or self.observation not in ('mark','observed_close'):
            raise ValueError('stress needs a name and mark/observed_close convention')
        if self.observation=='observed_close':
            if (self.shock.roll_days<=0 or len(self.close_history)!=self.shock.roll_days
                    or any(not isfinite(s) or s<=0 for s in self.close_history)
                    or self.shock.spot is not None and self.close_history[-1]!=self.shock.spot):
                raise ValueError('observed_close requires explicit positive day count and matching daily close history')
        elif self.shock.roll_days or self.close_history:
            raise ValueError('mark is instantaneous; time advancement requires observed_close history')

    def to_dict(self): return _jsonify(asdict(self))

    @classmethod
    def from_dict(cls,data):
        return cls(data['name'],ScenarioShock(**data.get('shock',{})),
                   data.get('observation','mark'),tuple(data.get('close_history',())))


def default_stresses(product,market,*,observed_day=None):
    """23 nonzero spot/ATM/rho combinations, each as mark and declared constant-close roll.

    Default observed horizon is the first remaining autocall date (or expiry).
    These are hypothetical histories, never actual historical market observations.
    """
    day=observed_day if observed_day is not None else (
        product.autocall.observation_days[0] if product.autocall else product.maturity_days)
    result=[]
    for ds,level,rho in cartesian_product((0.,-.1,-.2,-.3),(1.,1.25,1.5),(0.,-.1)):
        if (ds,level,rho)==(0.,1.,0.): continue
        spot=market.spots[0]*(1+ds)
        name=f'spot{ds:+.2f}_atm{level:.2f}_rho{rho:+.2f}'
        result.append(StressSpec(name+'_mark',ScenarioShock(spot=spot,level_multiplier=level,rho_shift=rho)))
        result.append(StressSpec(name+f'_close_day{day}',ScenarioShock(spot=spot,roll_days=day,
            level_multiplier=level,rho_shift=rho),'observed_close',(spot,)*day))
    return tuple(result)


def common_domain(product,market,config,scenarios=(),ko_levels=()):
    if config.spot_grid_upper is not None: return config
    levels=[*product.reference_spots,market.spots[0]*1.01,
            product.reference_spots[0]*product.redemption.strike_ratio]
    if product.knock_in: levels.append(product.reference_spots[0]*product.knock_in.barrier_ratio)
    if product.autocall: levels.extend(product.reference_spots[0]*v for v in product.autocall.barrier_ratios)
    for value in ko_levels:
        if value is not None:
            try:
                level=product.reference_spots[0]*value
                if isfinite(level*config.upper_spot_multiple) and level>0: levels.append(level)
            except (TypeError,ValueError,ArithmeticError): pass
    for spec in scenarios:
        candidates=list(spec.close_history)+([] if spec.shock.spot is None else [spec.shock.spot])
        # Invalid coordinates do not influence the common domain. They will fail
        # their own engine/domain validation, never be substituted or clipped.
        levels.extend(s*1.01 for s in candidates if isfinite(s*1.01*config.upper_spot_multiple))
    return replace(config,spot_grid_upper=config.upper_spot_multiple*max(levels))


def stress_test(product,market,model,*,scenarios=None,method='pde',pde_config=None,
                mc_config=None,progress=None,_risk=None,_base=None,_code_hash=None):
    """Holder loss = base PV - (paid cash PV + D(roll)*remaining PV).

    Local Taylor approximation uses initial mark Greeks only; it omits time,
    realized event changes and all cross/higher-order terms. The residual includes
    these effects and diffusion/numerical differences. MC CIs are price sampling
    intervals and are never presented as Greek uncertainty.
    """
    if method not in ('pde','mc'): raise ValueError('method must be pde or mc')
    specs=default_stresses(product,market) if scenarios is None else tuple(scenarios)
    config=common_domain(product,market,pde_config or PDEConfig(),specs)
    engine=config if method=='pde' else (mc_config or MCConfig())
    base=transform_scenario(product,market,model)
    base_price=price_scenario(base,method,engine).present_value if _base is None else _base
    risk=scenario_risk(base,pde_config=config) if _risk is None else _risk
    code_hash=_code_hash or content_hash(source_hashes())
    cells=[]
    for index,spec in enumerate(specs,1):
        cell=dict(request=spec.to_dict(),code_hash=code_hash,engine=asdict(engine))
        cell['input_hash']=content_hash(dict(product=_product_dict(product),market=asdict(market),
            model=model.to_dict(),stress=spec.to_dict(),method=method,engine=asdict(engine)))
        try:
            if spec.observation=='mark':
                node=transform_scenario(product,market,model,spec.shock)
            else:
                initial=replace(spec.shock,roll_days=0,spot=None)
                node=transform_scenario(product,market,model,initial)
                # A declared constant path uses the exact same canonical roll in one step.
                if len(set(spec.close_history))==1:
                    node=advance_scenario(node,ScenarioShock(spot=spec.close_history[0],roll_days=len(spec.close_history)))
                else:
                    for spot in spec.close_history:
                        node=advance_scenario(node,ScenarioShock(spot=spot,roll_days=1))
            value=price_scenario(node,method,engine)
            total=node.cash_pv+node.discount_to_roll*value.present_value
            pnl=total-base_price
            ds=node.market.spots[0]-market.spots[0]
            terms=dict(delta=ds,gamma=.5*ds**2,atm_level_sensitivity=spec.shock.level_multiplier-1,
                       rho_skew_sensitivity=spec.shock.rho_shift)
            approximation=sum(risk['metrics'][k]*v for k,v in terms.items() if v) if all(
                risk['metrics'][k] is not None for k,v in terms.items() if v) else None
            cell.update(status='ok',scenario=node.to_dict(),price=value.to_dict(),remaining_pv=value.present_value,
                cash_sum=node.cash_sum,cash_pv=node.cash_pv,discount_to_roll=node.discount_to_roll,
                total_base_date_value=total,holder_pnl=pnl,holder_loss=-pnl,
                local_approximation_pnl=approximation,
                full_revaluation_residual=None if approximation is None else pnl-approximation)
            if not all(isfinite(v) for v in (total,pnl,value.present_value)):
                raise ValueError('nonfinite stress value')
        except (ValueError,TypeError,RuntimeError,ArithmeticError) as exc:
            cell.update(status='error',error=str(exc))
        cells.append(cell)
        if progress: progress(index,len(specs),cell)
    return RiskSurfaceResult(tuple(cells),(),dict(base_pv=base_price,base_risk=risk,method=method,
        spot_grid_upper=config.spot_grid_upper,units=dict(holder_pnl='base-date currency',holder_loss='base-date currency'),
        approximation='initial instantaneous mark Delta/Gamma + ATM multiplier/rho first derivatives; omits time, realized events, cross and higher orders',
        observed_history='explicit hypothetical daily closes; event priority from contract',code_hash=code_hash))
