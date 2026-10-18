"""Contract-preserving coupon/barrier comparisons and model-aware fair coupons."""
from dataclasses import asdict, replace
from itertools import product as cartesian_product
from math import isfinite

from ..config import MCConfig, PDEConfig
from .fair_coupon import with_flat_coupon, solve_fair_coupon
from .surface_risk import RiskSurfaceResult, transform_scenario, scenario_risk, price_scenario, _product_dict, _audit_coordinates
from .stress import stress_test, default_stresses, common_domain, content_hash, source_hashes


def with_terms(product,*,coupon=None,ki=None,ko=None):
    """Override flat coupon/KI/flat KO; preserve observations, reference and accrual."""
    if product.n_assets!=1: raise ValueError('terms comparison supports single-asset notes')
    for name,value in (('ki',ki),('ko',ko)):
        if value is not None and (not isfinite(value) or value<=0): raise ValueError(f'{name} must be finite positive')
    if coupon is not None: product=with_flat_coupon(product,coupon)
    if ki is not None:
        if product.knock_in is None: raise ValueError('KI override requires knock-in feature')
        product=replace(product,knock_in=replace(product.knock_in,barrier_ratio=ki))
    if ko is not None:
        if product.autocall is None: raise ValueError('KO override requires autocall feature')
        product=replace(product,autocall=replace(product.autocall,barrier_ratios=(ko,)*len(product.autocall.observation_days)))
    return product


def compare_terms(product,market,model,*,coupons=(.1,.15,.2),ki_ratios=(.6,.7,.8),
                  ko_ratios=(.95,1.,1.05),scenarios=None,method='pde',pde_config=None,
                  mc_config=None,fair_bracket=(0.,1.),progress=None,_code_hash=None):
    specs=default_stresses(product,market) if scenarios is None else tuple(scenarios)
    coupons,ki_ratios,ko_ratios=tuple(coupons),tuple(ki_ratios),tuple(ko_ratios)
    config=common_domain(product,market,pde_config or PDEConfig(),specs,ko_ratios+ki_ratios)
    engine=config if method=='pde' else (mc_config or MCConfig())
    code_hash=_code_hash or content_hash(source_hashes())
    fair=[]
    for ki,ko in cartesian_product(ki_ratios,ko_ratios):
        entry=_audit_coordinates(dict(ki=ki,ko=ko,bracket=list(fair_bracket),target_pv=product.issue_price))
        try:
            candidate=with_terms(product,ki=ki,ko=ko)
            entry.update(status='ok',result=solve_fair_coupon(candidate,market,product.issue_price,method,
                engine,*fair_bracket,model=model).to_dict())
        except (ValueError,TypeError,RuntimeError,ArithmeticError) as exc:
            entry.update(status='error',error=str(exc))
        fair.append(entry)
    cells=[]
    total=len(coupons)*len(ki_ratios)*len(ko_ratios)
    for coupon,ki,ko in cartesian_product(coupons,ki_ratios,ko_ratios):
        cell=dict(request=_audit_coordinates(dict(coupon=coupon,ki=ki,ko=ko)),code_hash=code_hash)
        try:
            candidate=with_terms(product,coupon=coupon,ki=ki,ko=ko)
            node=transform_scenario(candidate,market,model)
            risk=scenario_risk(node,pde_config=config)
            base=price_scenario(node,method,engine).present_value if method!='pde' else risk['metrics']['present_value']
            cell.update(status='ok' if all(s['status']=='ok' for s in risk['metric_status'].values()) else 'partial',
                scenario=node.to_dict(),metrics=risk['metrics'],risk=risk,base_pv=base,
                input_hash=content_hash(dict(product=_product_dict(candidate),market=asdict(market),model=model.to_dict(),engine=asdict(engine))),
                stress=stress_test(candidate,market,model,scenarios=specs,method=method,pde_config=config,
                    mc_config=mc_config,_risk=risk,_base=base,_code_hash=code_hash).to_dict())
        except (ValueError,TypeError,RuntimeError,ArithmeticError) as exc:
            cell.update(status='error',error=str(exc))
        cells.append(cell)
        if progress: progress(len(cells),total,cell)
    return RiskSurfaceResult(tuple(cells),(),dict(fair_coupons=fair,spot_grid_upper=config.spot_grid_upper,
        engine=asdict(engine),method=method,code_hash=code_hash,
        coupon_basis='decimal annual rate; flat replacement retains original dates/accrual',
        barrier_basis='contractual reference ratio; KO replacement is flat at all original observation dates'))
