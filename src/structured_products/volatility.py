"""Price-calibrated regularized modified-power-law SSVI on an ACT/252 clock."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr

from .marketdata import (_finite_tuple, _scalar, _time, MarketSnapshot, MarketCurves,
                         TradingCalendar, QuotePolicy, validate_quotes)


def _detached(value):
    if isinstance(value, Mapping):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_detached(item) for item in value]
    return value


def black_price(forward, strike, discount, total_variance, option_type='call'):
    """Discounted European Black price; total_variance is sigma**2 * model_time."""
    f,k,d,w = np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (forward,strike,discount,total_variance)])
    if (any(np.any(~np.isfinite(v)) for v in (f,k,d,w)) or np.any(f<=0)
            or np.any(k<=0) or np.any(d<=0) or np.any(w<0) or option_type not in ('call','put')):
        raise ValueError("Black inputs need finite positive F/K/D, nonnegative variance and call/put type")
    root = np.sqrt(np.where(w>0,w,1))
    a = np.log(f/k)/root + root/2
    b = a-root
    sign = 1 if option_type=='call' else -1
    price = d*sign*(f*ndtr(sign*a)-k*ndtr(sign*b))
    price = np.where(w==0,d*np.maximum(sign*(f-k),0),price)
    return _scalar(price)


def implied_volatility(price, forward, strike, discount, model_time, option_type='call'):
    """Invert price on the supplied model clock; boundary IV is not identifiable."""
    t = float(_time(model_time))
    if not np.isfinite(price) or t<=0:
        raise ValueError("IV needs a finite price and positive model time")
    lower = black_price(forward,strike,discount,0,option_type)
    upper = discount*(forward if option_type=='call' else strike)
    if not lower < price < upper:
        raise ValueError("price must lie strictly inside European option bounds")
    # Root in total standard deviation avoids any ACT/365 or vendor-IV conversion.
    hi = 1.
    while black_price(forward,strike,discount,hi*hi,option_type) < price:
        hi *= 2
        if hi>64:
            raise ValueError("IV root cannot be bracketed")
    root = brentq(lambda x: black_price(forward,strike,discount,x*x,option_type)-price,0,hi,xtol=1e-13)
    return root/np.sqrt(t)


def _butterfly_factor(gamma):
    a = 1-2*gamma
    return 1. if a==0 else a**a/(1+a)**(1+a)


def _eta_cap(rho,gamma):
    return min(4/(1+abs(rho)),np.sqrt(4/((1+abs(rho))*_butterfly_factor(gamma))))


@dataclass(frozen=True)
class VarianceDerivatives:
    w: float | np.ndarray
    w_k: float | np.ndarray
    w_kk: float | np.ndarray
    w_t: float | np.ndarray


@dataclass(frozen=True)
class SSVISurface:
    """Shared-rho SSVI with globally bounded regularized modified power law.

    phi(theta)=eta/((theta+theta_floor)**gamma*(1+theta)**(1-gamma)).
    The positive floor removes the t=0 off-ATM derivative singularity. ATM total
    variance uses monotone PCHIP through (0,0) and quoted nodes; beyond the final
    node it extends with the last secant slope (right derivative at that node).
    """

    times: tuple[float, ...]
    theta: tuple[float, ...]
    rho: float
    eta: float
    gamma: float = .3
    theta_floor: float = 1e-4
    day_count: str = 'ACT/252'

    def __post_init__(self):
        object.__setattr__(self,'times',_finite_tuple(self.times))
        object.__setattr__(self,'theta',_finite_tuple(self.theta))
        for name in ('rho','eta','gamma','theta_floor'):
            object.__setattr__(self,name,float(getattr(self,name)))
        if (len(self.times)!=len(self.theta) or len(self.times)<2 or self.times[0]<=0
                or np.any(np.diff(self.times)<=0) or self.theta[0]<=0 or np.any(np.diff(self.theta)<0)
                or not np.all(np.isfinite((self.rho,self.eta,self.gamma,self.theta_floor)))
                or abs(self.rho)>=1 or self.eta<0 or not 0<=self.gamma<=.5
                or self.theta_floor<=0 or self.day_count!='ACT/252'):
            raise ValueError("invalid ACT/252 SSVI nodes or shape parameters")
        # Conservative all-theta sufficient bounds, not a sampled-grid assertion.
        wing = self.eta*(1+abs(self.rho))
        density = self.eta**2*(1+abs(self.rho))*_butterfly_factor(self.gamma)
        if wing>=4 or density>4:
            raise ValueError("SSVI fails global sufficient butterfly constraints")

    def _atm(self,t):
        t = _time(t)
        x = (0.,)+self.times
        y = (0.,)+self.theta
        interpolator = PchipInterpolator(x,y,extrapolate=False)
        inside = np.minimum(t,self.times[-1])
        value = interpolator(inside)
        derivative = interpolator.derivative()(inside)
        slope = (y[-1]-y[-2])/(x[-1]-x[-2])
        value = np.where(t>=x[-1],y[-1]+slope*(t-x[-1]),value)
        derivative = np.where(t>=x[-1],slope,derivative)
        return value,derivative

    def atm_total_variance(self,t):
        return _scalar(self._atm(t)[0])

    def derivatives(self,k,t):
        k,t = np.broadcast_arrays(np.asarray(k,dtype=float),_time(t))
        if np.any(~np.isfinite(k)):
            raise ValueError("log-forward-moneyness must be finite")
        theta,theta_t = self._atm(t)
        phi = self.eta/((theta+self.theta_floor)**self.gamma*(1+theta)**(1-self.gamma))
        phi_theta = -phi*(self.gamma/(theta+self.theta_floor)+(1-self.gamma)/(1+theta))
        x = phi*k
        root = np.sqrt((x+self.rho)**2+1-self.rho**2)
        shape = 1+self.rho*x+root
        shape_x = self.rho+(x+self.rho)/root
        w = theta*shape/2
        wk = theta*phi*shape_x/2
        wkk = theta*phi**2*(1-self.rho**2)/(2*root**3)
        wt = theta_t*(shape+theta*shape_x*k*phi_theta)/2
        values = (w,wk,wkk,wt)
        if any(np.any(~np.isfinite(v)) for v in values):
            raise ValueError("SSVI evaluation is outside finite numerical range")
        return VarianceDerivatives(*(_scalar(v) for v in values))

    def total_variance(self,k,t):
        return self.derivatives(k,t).w

    def arbitrage_diagnostics(self):
        times = np.unique(np.r_[np.geomspace(1e-9,self.times[-1]*4,80),self.times])
        ks = np.r_[np.linspace(-3,3,121),-100.,100.]
        k,t = np.meshgrid(ks,times)
        d = self.derivatives(k,t)
        g = (1-k*d.w_k/(2*d.w))**2-d.w_k**2/4*(1/d.w+.25)+d.w_kk/2
        return dict(analytic_sufficient=True,
                    scope='all finite k and t >= 0; numerical grid supplemental',
                    wing_bound=self.eta*(1+abs(self.rho)),
                    butterfly_bound=self.eta**2*(1+abs(self.rho))*_butterfly_factor(self.gamma),
                    min_density_factor=float(np.min(g)), min_w_t=float(np.min(d.w_t)),
                    numerical_k_range=[float(ks.min()),float(ks.max())],
                    numerical_time_range=[float(times.min()),float(times.max())])

    def to_dict(self):
        return dict(times=list(self.times),theta=list(self.theta),rho=self.rho,eta=self.eta,
                    gamma=self.gamma,theta_floor=self.theta_floor,day_count=self.day_count)

    @classmethod
    def from_dict(cls,data):
        return cls(**data)


@dataclass(frozen=True)
class CalibrationConfig:
    min_expiries: int = 5
    min_strikes: int = 9
    min_wing_log_moneyness: float = .1
    max_atm_log_moneyness: float = .05
    min_tenor_ratio: float = 4.
    max_price_rmse: float = .25
    max_iv_rmse: float = .01
    min_bid_ask_coverage: float = .8
    max_spread_rmse: float = 1.
    spread_floor: float = .005
    theta_floor: float = 1e-4
    max_nfev: int = 2000

    def __post_init__(self):
        values = tuple(vars(self).values())
        if (not np.all(np.isfinite(values)) or min(values)<=0
                or self.min_expiries<2 or self.min_strikes<3 or self.min_bid_ask_coverage>1
                or any(not isinstance(v,int) for v in (self.min_expiries,self.min_strikes,self.max_nfev))):
            raise ValueError("invalid calibration quality configuration")


@dataclass(frozen=True)
class QualityGate:
    name: str
    passed: bool
    observed: float | int | bool | None
    threshold: float | int | None
    detail: str = ''

    def to_dict(self):
        return dict(vars(self))


@dataclass(frozen=True)
class CalibrationResult:
    surface: SSVISurface | None
    validation: object
    config: CalibrationConfig
    gates: tuple[QualityGate, ...]
    overall: object
    per_expiry: tuple
    optimizer: object
    fitted_quotes: tuple

    @property
    def accepted(self):
        return self.surface is not None and all(g.passed for g in self.gates)

    def to_dict(self):
        synthetic=dict(self.validation.snapshot.provenance).get('kind')=='synthetic'
        status=('synthetic_accepted' if self.accepted else 'synthetic_rejected')+'; real_market_pending' if synthetic else ('accepted' if self.accepted else 'rejected_quality')
        return dict(schema_version=1,accepted=self.accepted,
                    surface=None if self.surface is None else self.surface.to_dict(),
                    validation=self.validation.to_dict(),config=dict(vars(self.config)),
                    gates=[g.to_dict() for g in self.gates],overall=dict(self.overall),
                    per_expiry=[dict(v) for v in self.per_expiry],optimizer=_detached(self.optimizer),
                    fitted_quotes=[dict(v) for v in self.fitted_quotes],
                    real_data_acceptance=status)

    def write_json(self,path):
        Path(path).write_text(json.dumps(self.to_dict(),indent=2,allow_nan=False)+'\n',encoding='utf-8')

    @classmethod
    def from_dict(cls, data):
        """Rebuild raw validation and deterministically refit; serialized gates are evidence only."""
        if data.get('schema_version') != 1:
            raise ValueError('unsupported calibration schema')
        raw = data['validation']
        validation = validate_quotes([record['raw'] for record in raw['records']],
            MarketSnapshot.from_dict(raw['snapshot']), TradingCalendar.from_dict(raw['calendar']),
            MarketCurves.from_dict(raw['curves']), QuotePolicy(**raw['policy']))
        return calibrate_ssvi(validation, CalibrationConfig(**data['config']))

    @classmethod
    def read_json(cls, path):
        def reject(value):
            raise ValueError(f'nonfinite JSON constant: {value}')
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject))


def calibrate_ssvi(validation, config=None):
    """Fit prices with inverse half-spread weights and independent acceptance gates.

    Shape bounds enforce the analytic sufficient inequalities for all theta>0.
    Three deterministic starts include the eta=0 flat limit. No quote repair,
    randomized initialization, or implicit acceptance from optimizer success.
    """
    config = config or CalibrationConfig()
    quotes = validation.accepted
    times = sorted({q.model_time for q in quotes})
    coverage_details = []
    for t in times:
        group = [q for q in quotes if q.model_time==t]
        ks = [np.log(q.strike/q.forward) for q in group]
        if (len({q.strike for q in group})<config.min_strikes
                or min(ks)>-config.min_wing_log_moneyness or max(ks)<config.min_wing_log_moneyness
                or min(abs(k) for k in ks)>config.max_atm_log_moneyness):
            coverage_details.append(f"time={t}: insufficient strikes/ATM/wings")
    enough = (len(times)>=config.min_expiries and not coverage_details
              and times[-1]/times[0]>=config.min_tenor_ratio)
    gates = [QualityGate('data_coverage',bool(enough),len(times),config.min_expiries,'; '.join(coverage_details))]
    if not enough:
        return CalibrationResult(None,validation,config,tuple(gates),MappingProxyType({}),(),
                                 MappingProxyType(dict(success=False,message='data coverage failed; not optimized')),())
    t = np.array([q.model_time for q in quotes])
    f = np.array([q.forward for q in quotes])
    strikes = np.array([q.strike for q in quotes])
    discount = np.array([q.discount for q in quotes])
    mids = np.array([q.mid for q in quotes])
    bids,asks = np.array([q.bid for q in quotes]),np.array([q.ask for q in quotes])
    iv = np.array([q.implied_vol for q in quotes])
    call = np.array([q.option_type=='call' for q in quotes])
    k = np.log(strikes/f)
    weights = np.maximum((asks-bids)/2,config.spread_floor)
    initial_theta = []
    for expiry in times:
        group = np.flatnonzero(t==expiry)
        near = group[np.argmin(np.abs(k[group]))]
        initial_theta.append(iv[near]**2*expiry)
    initial_theta = np.maximum.accumulate(initial_theta)
    increments = np.maximum(np.diff([0.]+list(initial_theta)),1e-8)

    def unpack(x):
        rho,eta_fraction,gamma = x[:3]
        eta = eta_fraction*_eta_cap(rho,gamma)*(1-1e-8)
        return SSVISurface(tuple(times),tuple(np.cumsum(x[3:])),rho,eta,gamma,config.theta_floor)

    def predictions(s):
        w = s.total_variance(k,t)
        return np.where(call,black_price(f,strikes,discount,w,'call'),black_price(f,strikes,discount,w,'put'))

    def residual(x):
        return (predictions(unpack(x))-mids)/weights

    runs = []
    # Finite bounds prevent unidentifiable parameter excursions; acceptance still
    # measures errors in original quote price and decimal-volatility units.
    for rho,eta_fraction in ((-.5,.25),(0.,0.),(.5,.25)):
        x0 = np.r_[rho,eta_fraction,.3,np.minimum(increments,3.9)]
        runs.append(least_squares(residual,x0,bounds=(np.r_[-.98,0.,0.,np.full(len(times),1e-12)],
                                                     np.r_[.98,1.,.5,np.full(len(times),4.)]),
                                  max_nfev=config.max_nfev,ftol=1e-11,xtol=1e-11,gtol=1e-11))
    best = min(runs,key=lambda r: float(np.dot(r.fun,r.fun)))
    fitted = unpack(best.x)
    prices = predictions(fitted)
    fitted_iv = np.sqrt(fitted.total_variance(k,t)/t)
    error = prices-mids
    iv_error = fitted_iv-iv
    covered = (prices>=bids-1e-10)&(prices<=asks+1e-10)

    def metrics(mask):
        return dict(count=int(np.count_nonzero(mask)),price_rmse=float(np.sqrt(np.mean(error[mask]**2))),
                    price_max_abs=float(np.max(np.abs(error[mask]))),
                    iv_rmse=float(np.sqrt(np.mean(iv_error[mask]**2))),iv_max_abs=float(np.max(np.abs(iv_error[mask]))),
                    bid_ask_coverage=float(np.mean(covered[mask])),spread_rmse=float(np.sqrt(np.mean((error[mask]/weights[mask])**2))))

    overall = metrics(np.ones(len(t),dtype=bool))
    per_expiry = tuple(MappingProxyType(dict(model_time=expiry,**metrics(t==expiry))) for expiry in times)
    checks = fitted.arbitrage_diagnostics()
    gates.extend([QualityGate('optimizer_converged',bool(best.success),bool(best.success),None,str(best.message)),
                  QualityGate('analytic_arbitrage',bool(checks['analytic_sufficient']),True,None),
                  QualityGate('numerical_density',checks['min_density_factor']>=-1e-10,checks['min_density_factor'],0.),
                  QualityGate('numerical_calendar',checks['min_w_t']>=-1e-10,checks['min_w_t'],0.)])
    for label,limit,lower in (('price_rmse',config.max_price_rmse,False),('iv_rmse',config.max_iv_rmse,False),
                              ('bid_ask_coverage',config.min_bid_ask_coverage,True),('spread_rmse',config.max_spread_rmse,False)):
        # Every expiry must meet the quality threshold, not just a pooled average.
        value = min(m[label] for m in per_expiry) if lower else max(m[label] for m in per_expiry)
        gates.append(QualityGate(label,bool(value>=limit if lower else value<=limit),value,limit,'worst expiry'))
    optimizer = dict(success=bool(best.success),status=int(best.status),message=str(best.message),
                     nfev=int(best.nfev),cost=float(best.cost),optimality=float(best.optimality),
                     starts=[dict(success=bool(r.success),nfev=int(r.nfev),cost=float(r.cost)) for r in runs],
                     arbitrage=checks)
    fitted_quotes = tuple(MappingProxyType(dict(row_number=q.row_number,model_price=float(p),model_iv=float(v),
                                               price_error=float(e),iv_error=float(ve),inside_bid_ask=bool(ok)))
                          for q,p,v,e,ve,ok in zip(quotes,prices,fitted_iv,error,iv_error,covered))
    return CalibrationResult(fitted,validation,config,tuple(gates),MappingProxyType(overall),per_expiry,
                             MappingProxyType(optimizer),fitted_quotes)
