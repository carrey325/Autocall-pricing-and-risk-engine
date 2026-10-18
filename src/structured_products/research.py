"""Reproducible offline market-surface research; optional plotting imports are lazy."""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from html import escape
from importlib.resources import files
from importlib.metadata import version
import hashlib
import json
from math import sqrt
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from .config import load_example, PDEConfig, MCConfig
from .marketdata import load_market_data,validate_quotes,QuotePolicy,MarketCurves
from .volatility import calibrate_ssvi,CalibrationConfig,black_price,SSVISurface
from .models import LocalVolModel,TimeVolModel
from .products import StructuredNote,CouponFeature,RedemptionFeature,KnockInFeature
from .enums import CouponRule
from .builders import make_classic_snowball,make_stepdown_snowball,make_standard_fcn
from .pricing import price
from .analytics.surface_risk import risk_surface,ScenarioShock,transform_scenario,scenario_risk
from .analytics.stress import StressSpec,stress_test,content_hash,source_hashes
from .analytics.terms import compare_terms
from .workflow_io import read_json,write_json,write_csv


def default_analysis():
    return read_json(files('structured_products').joinpath('data/market_surface/research.json'))


def numerical_validation(config,model,*,quick=False,progress=None):
    """Record independent space/time/domain changes; acceptance uses measured error.

    Measured refinement differences are diagnostics, not mathematical bias bounds.
    MC refinement uses independent Philox seeds; its difference includes sampling
    noise. No tolerance is enlarged after looking at results. Failures are rows.
    """
    rows=[]
    n=101 if quick else 801
    fine_n=201 if quick else 1601
    paths=200 if quick else 40000
    base=PDEConfig(n,time_steps_per_day=2,spot_grid_upper=400.)
    fine=replace(base,spot_grid_points=fine_n)
    daily=replace(fine,time_steps_per_day=1)
    domain=replace(fine,spot_grid_points=2*(fine_n-1)+1,spot_grid_upper=800.)
    mc1=MCConfig(n_paths=paths,steps_per_day=1,seed=17)
    mc4=replace(mc1,steps_per_day=4,seed=23)
    def values(p,m,v):
        coarse=price(p,m,'pde',base,model=v).present_value
        f=price(p,m,'pde',fine,model=v).present_value
        t=price(p,m,'pde',daily,model=v).present_value
        d=price(p,m,'pde',domain,model=v).present_value
        a=price(p,m,'mc',mc1,model=v); b=price(p,m,'mc',mc4,model=v)
        dx,dt,dd,dm=abs(f-coarse),abs(f-t),abs(f-d),abs(a.present_value-b.present_value)
        allowance=2*(dx+dt+dd)+dm
        sampling=3*b.standard_error
        return dict(pde=f,mc=b.present_value,mc_standard_error=b.standard_error,
            pde_space_change=dx,pde_time_change=dt,pde_domain_change=dd,mc_substep_change=dm,
            mc_refinement_joint_3se=3*sqrt(a.standard_error**2+b.standard_error**2),
            sampling_allowance=sampling,discretization_allowance=allowance,
            acceptance_allowance=sampling+allowance,cross_engine_difference=abs(f-b.present_value),
            passed=abs(f-b.present_value)<=sampling+allowance,
            precision=dict(space_coarse=asdict(base),fine=asdict(fine),time_coarse=asdict(daily),domain=asdict(domain),
                mc_daily=asdict(mc1),mc_fine=asdict(mc4)),
            allowance_basis='3 pair-aware SE + 2*(independent PDE space/time/domain changes) + observed independent-seed MC substep change; empirical, not a guaranteed bound')
    def record(row):
        rows.append(row)
        if progress: progress(len(rows),None,row)
    days=(21,) if quick else (63,126,252)
    strikes=(1.,) if quick else (.9,1.,1.1)
    flat=LocalVolModel.theoretical(replace(model.surface,rho=0.,eta=0.,theta=tuple(.04*t for t in model.surface.times)),model.curves)
    for label,v in (('nonflat',model),('constant_vol',flat)):
        for maturity in days:
            for strike in strikes:
                p=StructuredNote((100.,),1000.,maturity,CouponFeature(CouponRule.CONTINGENT_AT_REDEMPTION),
                    RedemptionFeature(strike_ratio=strike),knock_in=KnockInFeature(.01,initially_knocked_in=True))
                row=dict(kind='vanilla',model=label,days=maturity,strike=100*strike)
                try:
                    row.update(values(p,config.market,v))
                    t=maturity/252;k=100*strike
                    w=v.surface.total_variance(np.log(k/v.curves.forward(t)),t)
                    analytic=1000*v.discount(t)-1000/k*black_price(v.curves.forward(t),k,v.discount(t),w,'put')
                    row.update(analytic=analytic,pde_analytic_error=abs(row['pde']-analytic),
                        mc_analytic_error=abs(row['mc']-analytic),status='ok')
                    row['passed']=row['passed'] and row['mc_analytic_error']<=row['acceptance_allowance']
                except (ValueError,TypeError,RuntimeError,ArithmeticError) as exc: row.update(status='error',error=str(exc),passed=False)
                record(row)
    for builder in (make_classic_snowball,make_stepdown_snowball,make_standard_fcn):
        for ki in (False,True):
            p=builder(maturity_days=4 if quick else 42)
            p=replace(p,knock_in=replace(p.knock_in,initially_knocked_in=ki))
            m=replace(config.market,spots=(99.,));v=model.rebase(99.)
            row=dict(kind='product',product=p.product_name,initial_knocked_in=ki,spot=99.)
            try: row.update(values(p,m,v),status='ok')
            except (ValueError,TypeError,RuntimeError,ArithmeticError) as exc: row.update(status='error',error=str(exc),passed=False)
            record(row)
    # Both controls use identical curves/seed/domain; original legacy example is
    # explicitly a separate reference because it uses different carry.
    t=config.product.maturity_years
    sigma=sqrt(model.surface.total_variance(0.,t)/t)
    control=TimeVolModel(model.curves,(0.,),(sigma,))
    for method,engine in (('pde',config.pde),('mc',config.mc)):
        row=dict(kind='flat_control',method=method,sigma=sigma,curves=model.curves.to_dict(),engine=asdict(engine))
        try:
            row.update(local_vol=price(config.product,config.market,method,engine,model=model).to_dict(),
                same_curves_flat=price(config.product,config.market,method,engine,model=control).to_dict(),
                legacy_flat_reference=price(config.product,config.market,method,engine).to_dict(),status='ok')
            row['local_minus_same_curves_flat']=row['local_vol']['present_value']-row['same_curves_flat']['present_value']
        except (ValueError,TypeError,RuntimeError,ArithmeticError) as exc: row.update(status='error',error=str(exc))
        record(row)
    for ratio in ((1.,) if quick else (.795,.805,.995,1.005,1.1)):
        node=transform_scenario(config.product,config.market,model,ScenarioShock(spot=100*ratio))
        checks=[]
        for grid in (config.pde.spot_grid_points,201 if quick else 801):
            for bump in (.01,.005):
                checks.append(dict(grid=grid,bump=bump,**scenario_risk(node,pde_config=replace(config.pde,spot_grid_points=grid),
                    spot_bump=bump,level_bump=bump,rho_bump=bump)))
        # Predeclared per-1000-notional warning scales; no MC price CI as Greek CI.
        limits=dict(present_value=.5,delta=.1,gamma=.1,atm_level_sensitivity=10.,rho_skew_sensitivity=10.)
        differences={key:max(c['metrics'][key] for c in checks if c['metrics'][key] is not None)-
            min(c['metrics'][key] for c in checks if c['metrics'][key] is not None)
            for key in limits if all(c['metrics'][key] is not None for c in checks)}
        record(dict(kind='risk_checkpoint',spot_ratio=ratio,checks=checks,ranges=differences,warning_thresholds=limits,
            material_disagreement=any(v>limits[k]*config.product.notional/1000 for k,v in differences.items()),
            interpretation='bump/resolution sensitivity diagnostic; MC price CI does not quantify Greek uncertainty',status='ok'))
    return rows


def _table(rows,columns):
    def cell(value):
        return escape(f'{value:.6g}' if isinstance(value,float) else str(value),quote=True)
    return '<table><thead><tr>'+''.join('<th>'+cell(k)+'</th>' for k in columns)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+cell(row.get(k,''))+'</td>' for k in columns)+'</tr>' for row in rows)+'</tbody></table>'


def _save(fig,directory,name,plt):
    fig.savefig(directory/(name+'.png'),dpi=150)
    fig.savefig(directory/(name+'.svg'),metadata={'Date':None})
    plt.close(fig)


def _calibration_plot(directory,calibration,plt):
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    lookup={r['row_number']:r['quote'] for r in calibration['validation']['records'] if r['accepted']}
    fitted=calibration.get('fitted_quotes',[])
    times=sorted({q['model_time'] for q in lookup.values()})
    for t in times:
        group=[(lookup[r['row_number']],r) for r in fitted if lookup[r['row_number']]['model_time']==t]
        if not group: continue
        group.sort(key=lambda v:v[0]['strike'])
        ks=[np.log(q['strike']/q['forward']) for q,r in group]
        label=group[0][0]['expiry'][:10]
        line=axes[0,0].plot(ks,[q['implied_vol']*100 for q,r in group],'.',markersize=2,label=label)[0]
        axes[0,0].plot(ks,[r['model_iv']*100 for q,r in group],color=line.get_color(),linewidth=.8)
        axes[0,1].scatter(ks,[r['price_error'] for q,r in group],s=4,label=label)
    axes[0,0].set(xlabel='log(K/F)',ylabel='Implied vol (%) ACT/252',title='Observed IV points / SSVI lines')
    if axes[0,0].get_legend_handles_labels()[0]: axes[0,0].legend(fontsize=8)
    axes[0,1].axhline(0,color='black',linewidth=.6)
    axes[0,1].set(xlabel='log(K/F)',ylabel='Model minus quote midpoint (normalized price)',title='Price residuals; no quote repair')
    per=calibration.get('per_expiry',[])
    axes[1,0].bar(range(len(per)),[r['bid_ask_coverage']*100 for r in per])
    axes[1,0].axhline(calibration['config']['min_bid_ask_coverage']*100,color='crimson',linestyle='--',label='Frozen coverage gate')
    axes[1,0].set(xlabel='Expiry index (see table)',ylabel='Bid/ask coverage (%)',ylim=(0,105),title='Independent fit quality by expiry')
    axes[1,0].legend()
    if per:
        axes[1,1].plot([r['model_time'] for r in per],[r['iv_rmse']*100 for r in per],marker='.',label='IV RMSE')
        axes[1,1].axhline(calibration['config']['max_iv_rmse']*100,color='crimson',linestyle='--',label='Frozen IV gate')
    axes[1,1].set(xlabel='ACT/252 model time',ylabel='Volatility points (%)',title='IV error / frozen gate')
    if axes[1,1].get_legend_handles_labels()[0]: axes[1,1].legend()
    fig.suptitle('ACCEPTED synthetic fit' if calibration.get('real_data_acceptance','').startswith('synthetic_accepted')
        else ('ACCEPTED fit' if calibration['accepted'] else 'REJECTED SSVI fit — No pricing performed'))
    _save(fig,directory,'calibration-fit',plt)


def _analysis_plots(directory,risk,stress,terms,calibration,plt):
    model=LocalVolModel.theoretical(SSVISurface.from_dict(calibration['surface']),MarketCurves.from_dict(calibration['validation']['curves']))
    ts=np.linspace(.002,model.surface.times[-1],90);spots=np.linspace(.5,1.5,100)*model.anchor_spot
    tt,ss=np.meshgrid(ts,spots);k=np.log(ss/model.curves.forward(tt))
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,z,title in ((axes[0],100*np.sqrt(model.surface.total_variance(k,tt)/tt),'SSVI implied volatility (%)'),
                       (axes[1],100*model.local_vol(tt,ss),'Dupire local volatility (%)')):
        image=ax.pcolormesh(tt,ss/model.anchor_spot,z,shading='auto',cmap='viridis')
        fig.colorbar(image,ax=ax,label='Annual vol (%) ACT/252');ax.set(xlabel='ACT/252 model time',ylabel='Spot / snapshot spot',title=title)
    fig.suptitle('Accepted surface: interpolation and modeled wings / tenor extension')
    _save(fig,directory,'local-vol',plt)
    fig,axes=plt.subplots(2,3,figsize=(17,10),layout='constrained')
    names=('present_value','delta','gamma','atm_level_sensitivity','rho_skew_sensitivity')
    units=('PV / note','PV / spot','PV / spot²','PV / ATM multiplier','PV / SSVI rho')
    cells=[c for c in risk['cells'] if c.get('status')!='error' and c['request']['rho_shift']==0]
    days=sorted({c['request']['roll_days'] for c in cells})
    colors=plt.get_cmap('viridis')(np.linspace(0,.9,max(len(days),1)))
    handles=[]
    for ax,name,unit in zip(axes.flat,names,units):
        for day,color in zip(days,colors):
            for ki in (False,True):
                group=sorted([c for c in cells if c['request']['roll_days']==day and c['request']['initial_knocked_in']==ki
                    and c['metrics'].get(name) is not None],key=lambda c:c['request']['spot_ratio'])
                if group:
                    line=ax.plot([c['request']['spot_ratio'] for c in group],[c['metrics'][name] for c in group],
                        color=color,linestyle='--' if ki else '-',label=f'day {day}; initial KI {ki}')[0]
                    if name=='present_value':handles.append(line)
        ax.set(xlabel='Spot / contractual reference',ylabel=unit,
            title='Remaining ex-cash PV (KO/expiry → 0)' if name=='present_value' else name)
        ax.axvline(.8,color='grey',linewidth=.6);ax.axvline(1.,color='grey',linewidth=.6)
    axes[1,2].axis('off')
    if handles:axes[1,2].legend(handles=handles,loc='center',fontsize=9)
    fig.suptitle('All surface metrics: rho shift 0; roll and initial-KI states (moderate precision)')
    _save(fig,directory,'risk-metrics',plt)
    cells=[c for c in stress['cells'] if c.get('status')=='ok']
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for observation,color in (('mark','#2455a4'),('observed_close','#bc511d')):
        group=[(i,c) for i,c in enumerate(cells) if c['request']['observation']==observation]
        axes[0].scatter([i for i,c in group],[c['holder_loss'] for i,c in group],s=14,label=observation,color=color)
        approx=[c for i,c in group if c['local_approximation_pnl'] is not None]
        axes[1].scatter([c['local_approximation_pnl'] for c in approx],[c['holder_pnl'] for c in approx],s=14,label=observation,color=color)
        axes[2].scatter([i for i,c in group if c['full_revaluation_residual'] is not None],
            [c['full_revaluation_residual'] for i,c in group if c['full_revaluation_residual'] is not None],s=14,color=color)
    axes[0].set(xlabel='Serialized scenario index',ylabel='Base-date holder loss',title='All full-revaluation stresses')
    axes[1].set(xlabel='Initial Taylor P&L (omits time/events)',ylabel='Full base-date P&L',title='Approximation vs revaluation')
    axes[2].set(xlabel='Serialized scenario index',ylabel='Full P&L minus Taylor P&L',title='Nonlinear / time / event residual')
    axes[0].legend();axes[1].legend()
    _save(fig,directory,'stress-residuals',plt)
    cells=[c for c in terms['cells'] if c.get('base_pv') is not None]
    fair=terms.get('diagnostics',{}).get('fair_coupons',[])
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    term_image=None
    for index,ko in enumerate(sorted({c['request']['ko'] for c in cells if c['request']['ko'] is not None})):
        group=[c for c in cells if c['request']['ko']==ko]
        term_image=axes[0].scatter([c['request']['coupon']*100 for c in group],[c['base_pv'] for c in group],
            c=[c['request']['ki'] for c in group],vmin=.6,vmax=.8,label=f'KO {ko:g}',cmap='viridis',marker=('o','s','^')[index%3])
    axes[0].set(xlabel='Annual coupon (%)',ylabel='PV / note',title='Coupon / KI / KO comparisons')
    if cells:axes[0].legend()
    if term_image is not None:fig.colorbar(term_image,ax=axes[0],label='KI / reference')
    good=[r for r in fair if r['status']=='ok' and r['ki'] is not None and r['ko'] is not None]
    if good:
        image=axes[1].scatter([r['ko'] for r in good],[r['ki'] for r in good],c=[r['result']['coupon_rate']*100 for r in good],s=700,cmap='viridis')
        for r in good:axes[1].annotate(f"{r['result']['coupon_rate']*100:.2f}%",(r['ko'],r['ki']),ha='center',va='center',
            color='black' if r['result']['coupon_rate']>.17 else 'white',fontsize=8)
        fig.colorbar(image,ax=axes[1],label='Fair annual coupon (%)')
        axes[1].margins(x=.2,y=.15)
    axes[1].set(xlabel='KO / reference',ylabel='KI / reference',title='Fair coupon to issue price (bracket failures in table)')
    worst=[max((s['holder_loss'] for s in c['stress']['cells'] if s['status']=='ok'),default=0.) for c in cells]
    axes[2].bar(range(len(cells)),worst)
    axes[2].set(xlabel='Serialized term combination index',ylabel='Worst full-stress holder loss',title='All terms / all stress combinations')
    _save(fig,directory,'terms-fair-coupon',plt)


def render_report(directory,title,risk,stress,terms,validation,*,status,calibration):
    """Write real plots plus standalone local HTML, with all content escaped."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc: raise RuntimeError('research figures require optional install: pip install structured-products-pricer[plot]') from exc
    directory=Path(directory)
    _calibration_plot(directory,calibration,plt)
    gates=_table(calibration.get('gates',[]),('name','observed','threshold','passed','detail'))
    per_expiry=_table(calibration.get('per_expiry',[]),('model_time','count','price_rmse','iv_rmse','bid_ask_coverage','spread_rmse'))
    provenance=escape(json.dumps(calibration['validation']['snapshot'],indent=2,allow_nan=False))
    if not calibration['accepted']:
        # The rejected branch displays measured calibration diagnostics only.
        for suffix in ('.png','.svg'):
            (directory/('overview'+suffix)).write_bytes((directory/('calibration-fit'+suffix)).read_bytes())
        html=f'''<!doctype html><meta charset="utf-8"><title>{escape(title)}</title><style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:20px;line-height:1.6}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccd}}pre{{white-space:pre-wrap}}</style><h1>{escape(title)}</h1><h2>REJECTED — No pricing performed</h2><p>The observed quotes failed the frozen calibration quality gates. Convergence and arbitrage checks alone do not establish acceptable market fit. Pricing, stress, terms and numerical pricing validation are pending; no zero count is evidence of a pass.</p><img src="calibration-fit.png" alt="Observed IV, SSVI fit residuals and rejected quality gates"><h2>Acceptance gates</h2>{gates}<h2>Per-expiry errors and bid/ask coverage</h2>{per_expiry}<h2>Snapshot, selection and source hash</h2><pre>{provenance}</pre><p>One root/settlement convention; synchronous15:45 fields; K/spot[.8,1.2]. Curves use quality-screened paired quotes in[.9,1.1]spot and executable parity intervals. Calendar and normalization are explicit in the saved inputs. Raw/derived vendor data stay local.</p><p><a href="calibration.json">Complete calibration/cleaning evidence</a> · <a href="calibration-fit.svg">Vector figure</a> · <a href="manifest.json">Manifest</a></p>'''
        (directory/'report.html').write_text(html,encoding='utf-8')
        return dict(status=status,calibration_accepted=False,real_data_acceptance=calibration['real_data_acceptance'],
            failed_calibration_gates=[g['name'] for g in calibration['gates'] if not g['passed']],
            pricing_performed=False,numerical_validation='not run; calibration rejected')
    _analysis_plots(directory,risk,stress,terms,calibration,plt)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    cells=[c for c in risk.get('cells',[]) if c.get('metrics',{}).get('present_value') is not None
        and c['request'].get('rho_shift')==0 and c['request'].get('roll_days')==0
        and c['request'].get('initial_knocked_in') is False]
    axes[0,0].plot([c['request']['spot_ratio'] for c in cells],[c['metrics']['present_value'] for c in cells],marker='.')
    axes[0,0].set(xlabel='Spot / contractual reference',ylabel='PV per note',title='Initial PV; sticky log-forward moneyness')
    cells=[c for c in stress.get('cells',[]) if c.get('status')=='ok'][:12]
    axes[0,1].bar(range(len(cells)),[c['holder_loss'] for c in cells]);axes[0,1].axhline(0,color='black',linewidth=.5)
    axes[0,1].set(xlabel='First 12 serialized stress scenarios',ylabel='Base-date holder loss',title='Full revaluation; cash included')
    cells=[c for c in terms.get('cells',[]) if c.get('base_pv') is not None]
    for ki in sorted({c['request']['ki'] for c in cells if c['request']['ki'] is not None}):
        group=[c for c in cells if c['request']['ki']==ki]
        axes[1,0].scatter([c['request']['coupon'] for c in group],[c['base_pv'] for c in group],label=f'KI {ki:g}')
    axes[1,0].set(xlabel='Annual coupon (decimal)',ylabel='PV per note',title='Full coupon / KI / KO grid')
    if cells: axes[1,0].legend()
    checked=[r for r in validation if 'cross_engine_difference' in r]
    axes[1,1].plot([r['cross_engine_difference'] for r in checked],label='|MC - PDE|',marker='.')
    axes[1,1].plot([r['acceptance_allowance'] for r in checked],label='3 SE + measured refinement')
    axes[1,1].set(xlabel='Validation row',ylabel='PV units',title='Independent numerical evidence')
    if checked: axes[1,1].legend()
    fig.suptitle('Market surface risk research — calibration and precision study')
    fig.savefig(directory/'overview.png',dpi=150)
    fig.savefig(directory/'overview.svg',metadata={'Date':None})
    plt.close(fig)
    summary=dict(status=status,risk_cells=len(risk.get('cells',[])),stress_cells=len(stress.get('cells',[])),
        terms_cells=len(terms.get('cells',[])),validation_rows=len(validation),
        failed_numerical_checks=sum(r.get('passed') is False or r.get('status')=='error' for r in validation),
        risk_checkpoint_warnings=sum(r.get('material_disagreement',False) for r in validation),
        calibration_accepted=calibration.get('accepted'),real_data_acceptance=calibration.get('real_data_acceptance'))
    controls=[r for r in validation if r.get('kind')=='flat_control' and r.get('status')=='ok']
    losses=[c for c in stress.get('cells',[]) if c.get('status')=='ok']
    fair=terms.get('diagnostics',{}).get('fair_coupons',[])
    good=[r['result']['coupon_rate'] for r in fair if r['status']=='ok']
    observations=[]
    for row in controls:observations.append(f"{row['method'].upper()} local-vol minus SAME-CURVES flat control: {row['local_minus_same_curves_flat']:.6f} PV units.")
    if losses:
        worst=max(losses,key=lambda c:c['holder_loss'])
        observations.append(f"Worst of {len(losses)} full-revaluation stresses: loss {worst['holder_loss']:.6f}, request {worst['request']['name']}.")
    if good:observations.append(f"Fair annual coupon range {min(good)*100:.4f}% to {max(good)*100:.4f}% across {len(good)} accepted brackets; {len(fair)-len(good)} bracket failures.")
    observations.append(f"Numerical checks failed: {summary['failed_numerical_checks']}; material bump/grid risk disagreements: {summary['risk_checkpoint_warnings']}. Barrier sensitivity peaks require refinement, not a price CI.")
    findings='<ul>'+''.join('<li>'+escape(s)+'</li>' for s in observations)+'</ul>'
    numeric=_table([r for r in validation if 'cross_engine_difference' in r],
        ('kind','model','product','days','strike','initial_knocked_in','pde','mc','mc_standard_error','cross_engine_difference','acceptance_allowance','passed'))
    fairs=_table([dict(ki=r['ki'],ko=r['ko'],status=r['status'],coupon=r.get('result',{}).get('coupon_rate'),
        bracket=r['bracket'],residual=r.get('result',{}).get('residual'),error=r.get('error','')) for r in fair],
        ('ki','ko','status','coupon','bracket','residual','error'))
    refinements=[]
    for checkpoint in (r for r in validation if r.get('kind')=='risk_checkpoint'):
        checks=checkpoint['checks'];grids=sorted({c['grid'] for c in checks})
        coarse=next(c for c in checks if c['grid']==grids[0] and c['bump']==.01)
        fine=next(c for c in checks if c['grid']==grids[-1] and c['bump']==.01)
        small=next(c for c in checks if c['grid']==grids[-1] and c['bump']==.005)
        for metric in checkpoint['warning_thresholds']:
            a,b,d=(c['metrics'][metric] for c in (coarse,fine,small))
            refinements.append(dict(spot_ratio=checkpoint['spot_ratio'],metric=metric,coarse_grid=grids[0],fine_grid=grids[-1],
                coarse_1pct=a,fine_1pct=b,fine_halfpct=d,
                grid_change=None if a is None or b is None else abs(a-b),
                fine_bump_change=None if b is None or d is None else abs(b-d),warning_scale=checkpoint['warning_thresholds'][metric]))
    refine_table=_table(refinements,('spot_ratio','metric','coarse_grid','fine_grid','coarse_1pct','fine_1pct','fine_halfpct','grid_change','fine_bump_change','warning_scale'))
    warning_plot=''
    if refinements:
        fig,axs=plt.subplots(1,3,figsize=(16,5),layout='constrained')
        for ax,metric,unit in zip(axs,('present_value','delta','gamma'),('PV / note','PV / spot','PV / spot²')):
            group=[r for r in refinements if r['metric']==metric]
            ax.plot([r['spot_ratio'] for r in group],[r['grid_change'] for r in group],'.-',label='Coarse→fine, bump1%')
            ax.plot([r['spot_ratio'] for r in group],[r['fine_bump_change'] for r in group],'.-',label='Fine grid, bump1%→0.5%')
            ax.axhline(group[0]['warning_scale'],color='crimson',linestyle='--',label='Predeclared warning scale')
            ax.set(xlabel='Spot / contractual reference',ylabel='Absolute change: '+unit,title=metric);ax.legend(fontsize=8)
        fig.suptitle('Risk checkpoints: resolution changes versus remaining bump sensitivity')
        _save(fig,directory,'risk-checkpoints',plt)
        warning_plot='<img src="risk-checkpoints.png" alt="Grid changes versus fine-grid bump sensitivity">'
    addendum=''
    addendum_path=directory/'risk-refinement-addendum.json'
    if addendum_path.is_file():
        extra=read_json(addendum_path)
        extra_rows=[dict(grid=r['grid'],bump=r['bump'],origin=r['origin'],**r['metrics']) for r in extra['rows']]
        addendum='<h3>Limited spot/reference0.805 refinement addendum</h3><p>Only this existing state was refined to1601/3201 points at the same upper500 and daily time step, with bumps1%/0.5%. Original five warnings remain. This limited check does not certify every grid cell or eliminate time/discretization error.</p>'+_table(extra_rows,
            ('origin','grid','bump','present_value','delta','gamma','atm_level_sensitivity','rho_skew_sensitivity'))+'<p><a href="risk-refinement-addendum.json">Addendum JSON and source hashes</a> · <a href="risk-refinement-addendum.csv">CSV</a></p>'
    text='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>body{{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;line-height:1.6;color:#182230}}img{{width:100%;height:auto}}pre{{white-space:pre-wrap;background:#f3f5f8;padding:20px}}a{{color:#174ea6}}h1,h2{{line-height:1.25}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ccd;padding:6px}}</style>
<h1>{title}</h1><p>{status}</p><p>Research demonstration. Full-grid PDE precision is moderate; barrier Greeks require the separate bump/grid checkpoints. Price MC confidence intervals measure sampling error, excluding discretization bias. Failed checks remain visible.</p>
<img src="overview.png" alt="PV surface, stresses, term comparisons and numerical validation"><p><a href="overview.svg">Vector figure</a> · <a href="manifest.json">Reproducibility manifest</a></p>
<h2>Measured findings</h2>{findings}<pre>{summary}</pre>
<h2>Calibration / fit quality</h2><img src="calibration-fit.png" alt="IV smiles and fit residuals">{gates}{per_expiry}
<h2>SSVI and Dupire fields</h2><img src="local-vol.png" alt="Implied and local volatility fields">
<h2>PV and every surface sensitivity</h2><p>The PV panel is remaining ex-cash PV: paid cash is excluded and KO/expiry gives zero future exposure. It is not total holder value or a loss; holder value includes cash PV plus discounted remaining PV.</p><img src="risk-metrics.png" alt="Remaining ex-cash PV Delta Gamma ATM rho across roll and KI states">
<h2>Bump and resolution limits</h2><p>The full-grid spot Greeks are not fully converged. Each flagged checkpoint must be read as a numerical limitation. The table separates coarse→fine resolution at a fixed1% bump from1%→0.5% bump changes within the fine grid; large Gamma changes or sign changes must not be interpreted as economic precision.</p>{warning_plot}{refine_table}{addendum}
<h2>Full stresses and approximation residuals</h2><img src="stress-residuals.png" alt="All full stresses and nonlinear residuals">
<h2>Coupon / barriers / fair coupon</h2><img src="terms-fair-coupon.png" alt="Term comparison and fair coupon">{fairs}
<h2>Conventions</h2><p>ACT/252 explicit supplied sessions. Contract age and model clock are separate. Instant spot marks use sticky log-forward moneyness and preserve KI history. Observed-close stresses use the serialized hypothetical daily closes and canonical contract event priorities, never a day-zero KO. Holder P&amp;L = cash PV + D(roll) × remaining PV − initial PV.</p>
<p>ATM multiplier scales theta by its square; rho shock is additive SSVI correlation. Taylor approximation uses initial Delta/Gamma plus first ATM/rho derivatives and omits time, realized events, cross and higher orders. Residual includes all omitted effects. No VaR/ES.</p>
<p>Flat-vol controls use the same F/D curves, seed and domain, with sigma = sqrt(w(0,T)/T). The original flat example is a separately labeled legacy reference with different carry. SSVI wings and terminal theta extrapolation remain modeled; finite domain/refinement evidence is empirical.</p>
<h2>Reconstructable outputs</h2><p><a href="risk.json">Risk JSON</a> · <a href="risk.csv">Risk CSV</a> · <a href="stress.json">Stress JSON</a> · <a href="stress.csv">Stress CSV</a> · <a href="terms.json">Terms JSON</a> · <a href="terms.csv">Terms CSV</a> · <a href="validation.json">Convergence JSON</a> · <a href="validation.csv">Convergence CSV</a> · <a href="calibration.json">Calibration gates and quotes audit</a></p>
<h2>Numerical validation</h2>{numeric}<p>Full independent refinement inputs and all risk checkpoint ranges are in validation.json/CSV. Flags remain visible; no threshold was widened.</p></html>'''
    (directory/'report.html').write_text(text.format(title=escape(title,quote=True),status=escape(status),
        summary=escape(json.dumps(summary,indent=2,allow_nan=False)),
        findings=findings,gates=gates,per_expiry=per_expiry,fairs=fairs,numeric=numeric,
        warning_plot=warning_plot,refine_table=refine_table,addendum=addendum),encoding='utf-8')
    return summary


def run_research(output_directory,*,config=None,analysis=None,market_input=None,progress=None):
    """Run calibration -> risk/stress/terms -> validation -> local report.

    Completed stages are restartable only under the same input/source hash.
    Raw/derived external vendor quote artifacts remain the caller's local files.
    A rejected fit generates diagnostics and never enters pricing.
    """
    directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=True)
    try:
        head=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],stderr=subprocess.DEVNULL,text=True).strip())
    except (OSError,subprocess.CalledProcessError):head=None;dirty=None
    plan=default_analysis() if analysis is None else analysis
    config=config or load_example('classic_snowball')
    config=replace(config,pde=PDEConfig(**plan['pde']) if 'pde' in plan else config.pde,
                   mc=MCConfig(**plan['mc']) if 'mc' in plan else config.mc)
    source=files('structured_products').joinpath('data/market_surface') if market_input is None else Path(market_input)
    inputs=load_market_data(source)
    code=source_hashes()
    input_files={name:hashlib.sha256(source.joinpath(name).read_bytes()).hexdigest()
                 for name in ('snapshot.json','curves.csv','calendar.csv','quotes.csv')}
    key=content_hash(dict(config=config.to_dict(),analysis=plan,inputs=input_files,code=code))
    checkpoint=directory/'checkpoint.json'
    state=read_json(checkpoint) if checkpoint.exists() else dict(input_hash=key,inputs=input_files,source_hashes=code,
        code_revision=head,dirty_source=dirty,stage_hashes={})
    if state['input_hash']!=key:
        raise ValueError('checkpoint input/source hash differs; use a fresh output directory')
    write_json(checkpoint,state)
    def log(stage,current,total=None,cell=None):
        with (directory/'progress.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(dict(stage=stage,completed=current,total=total),allow_nan=False)+'\n')
        if progress: progress(stage,current,total,cell)
    def seal(name):
        for suffix in ('.json','.csv'):
            path=directory/(name+suffix)
            if path.exists():state['stage_hashes'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(checkpoint,state)
    def stage(name,callback):
        path=directory/(name+'.json')
        if path.exists():
            for suffix in ('.json','.csv'):
                cached=directory/(name+suffix)
                expected=state.get('stage_hashes',{}).get(cached.name)
                if cached.exists() or expected is not None:
                    if not cached.exists() or expected!=hashlib.sha256(cached.read_bytes()).hexdigest():
                        raise ValueError(f'checkpoint output hash mismatch or absent: {cached.name}')
            return read_json(path)
        log(name,0)
        result=callback()
        payload=result.to_dict() if hasattr(result,'to_dict') else result
        write_json(path,payload)
        if hasattr(result,'write_csv'):result.write_csv(directory/(name+'.csv'))
        seal(name)
        return payload
    policy=QuotePolicy(**plan.get('quote_policy',{}))
    quality=CalibrationConfig(**plan.get('calibration',{}))
    fit=stage('calibration',lambda:calibrate_ssvi(validate_quotes(inputs.rows,inputs.snapshot,inputs.calendar,inputs.curves,policy),quality))
    write_csv(directory/'cleaning.csv',fit['validation']['records'])
    if not fit['accepted']:
        status=fit['real_data_acceptance']+'; pricing_pending'
        risk=stress=terms={'cells':[]};validation=[]
    else:
        # In-memory accepted fit is reconstructed once when resuming, never in loops.
        from .volatility import CalibrationResult
        model=LocalVolModel.from_calibration(CalibrationResult.from_dict(fit))
        specs=None if 'scenarios' not in plan else tuple(StressSpec.from_dict(s) for s in plan['scenarios'])
        callback=lambda name:lambda i,n,c:log(name,i,n,c)
        checkpoints=tuple(ScenarioShock(**s) for s in plan.get('mc_checkpoints',[{'spot':80.},{'spot':100.}]))
        risk=stage('risk',lambda:risk_surface(config.product,config.market,model,pde_config=config.pde,
            mc_config=config.mc,mc_checkpoints=checkpoints,progress=callback('risk'),**plan.get('risk',{})))
        # Attach stable baseline input/code hashes to each exported row.
        for cell in risk['cells']:
            cell['input_hash']=content_hash(dict(inputs=input_files,request=cell['request'],config=config.to_dict()))
            cell['code_hash']=content_hash(code)
        write_json(directory/'risk.json',risk);write_csv(directory/'risk.csv',risk['cells'])
        seal('risk')
        stress=stage('stress',lambda:stress_test(config.product,config.market,model,scenarios=specs,
            pde_config=config.pde,mc_config=config.mc,progress=callback('stress'),_code_hash=content_hash(code)))
        terms=stage('terms',lambda:compare_terms(config.product,config.market,model,scenarios=specs,
            pde_config=config.pde,mc_config=config.mc,progress=callback('terms'),_code_hash=content_hash(code),**plan.get('terms',{})))
        validation=stage('validation',lambda:numerical_validation(config,model,progress=callback('validation'))
                         if plan.get('validation',True) else [])
        status=('synthetic_completed; real_market_pending' if dict(inputs.snapshot.provenance).get('kind')=='synthetic'
                else 'real_accepted_research_completed')
    for name,payload in (('risk',risk),('stress',stress),('terms',terms)):
        if not (directory/(name+'.json')).exists():write_json(directory/(name+'.json'),payload);write_csv(directory/(name+'.csv'),[])
    write_json(directory/'validation.json',validation);write_csv(directory/'validation.csv',validation)
    if fit['accepted']:seal('validation')
    summary=render_report(directory,plan.get('title','Market surface risk research'),risk,stress,terms,validation,status=status,calibration=fit)
    manifest=dict(status=status,input_hash=key,input_files=input_files,source_hashes=code,
        code_revision=state.get('code_revision',head),dirty_source=state.get('dirty_source',dirty),config=config.to_dict(),analysis=plan,
        environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
            numpy=np.__version__,scipy=version('scipy'),matplotlib=version('matplotlib')),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),summary=summary,
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())
                   if p.is_file() and p.name not in ('manifest.json','progress.jsonl')})
    write_json(directory/'manifest.json',manifest)
    return manifest


def refresh_report(output_directory):
    """Verify saved computation hashes and redraw only, preserving its provenance.

    This also supports reports computed before the stage-checkpoint hash schema;
    the completed manifest's output hashes must still validate. It never reprices
    or silently attributes previous computations to the rendering revision.
    """
    directory=Path(output_directory)
    manifest=read_json(directory/'manifest.json')
    for name,expected in manifest['artifacts'].items():
        path=directory/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'computation artifact hash mismatch: {name}')
    summary=render_report(directory,manifest['analysis'].get('title','Market surface risk research'),
        read_json(directory/'risk.json'),read_json(directory/'stress.json'),read_json(directory/'terms.json'),
        read_json(directory/'validation.json'),status=manifest['status'],calibration=read_json(directory/'calibration.json'))
    try:head=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
    except (OSError,subprocess.CalledProcessError):head=None
    manifest['report_render']=dict(code_revision=head,source_hashes=source_hashes(),
        rendered_at_utc=datetime.now(timezone.utc).isoformat(),pricing_performed=False)
    manifest['summary']=summary
    manifest['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())
        if p.is_file() and p.name not in ('manifest.json','progress.jsonl')}
    write_json(directory/'manifest.json',manifest)
    return manifest
