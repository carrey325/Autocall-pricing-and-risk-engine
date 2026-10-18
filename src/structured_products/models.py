"""Immutable single-asset diffusion models on a separate ACT/252 market clock."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Protocol

import numpy as np

from .marketdata import MarketCurves, _finite_tuple, _scalar, _time
from .volatility import CalibrationResult, SSVISurface


class PricingModel(Protocol):
    anchor_spot: float
    time_offset: float
    day_count: str

    def discount(self, t): ...
    def forward_growth(self, t0, t1): ...
    def local_vol(self, t, spot): ...
    def diagnostics(self): ...


class _CurveModel:
    """All public time arguments are relative to the current valuation date."""

    @property
    def day_count(self):
        return self.curves.day_count

    def _validate(self):
        offset = float(self.time_offset)
        anchor = self.curves.forwards[0] if self.anchor_spot is None else float(self.anchor_spot)
        if not np.isfinite(offset) or offset < 0 or not np.isfinite(anchor) or anchor <= 0:
            raise ValueError('model offset/anchor must be finite, nonnegative time and positive spot')
        object.__setattr__(self, 'time_offset', offset)
        object.__setattr__(self, 'anchor_spot', anchor)

    def discount(self, t):
        return self.curves.discount_between(self.time_offset, self.time_offset + _time(t))

    def forward_growth(self, t0, t1):
        return self.curves.forward_growth(self.time_offset + _time(t0), self.time_offset + _time(t1))

    def rolled(self, time_offset, current_spot):
        """Set absolute market time/conditional spot; retain absolute diffusion axis."""
        return replace(self, time_offset=time_offset, anchor_spot=current_spot)

    def rebase(self, spot):
        """Sticky log-forward-moneyness mark: scale the entire absolute price axis."""
        if not np.isfinite(spot) or spot <= 0:
            raise ValueError('rebase spot must be finite and positive')
        factor = spot/self.anchor_spot
        shifted = replace(self.curves, forwards=tuple(f*factor for f in self.curves.forwards))
        return replace(self, curves=shifted, anchor_spot=spot)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class _AcceptedCalibration:
    """Internal immutable proof retained by cheap roll/rebase operations."""

    json: str
    surface: SSVISurface
    support: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class LocalVolModel(_CurveModel):
    """Dupire diffusion from regularized SSVI; bare surfaces are theoretical.

    At zero total variance, analytic shape ratios give the exact right limit of
    Dupire's density factor. Positive finite spot is required; S=0 is a PDE
    boundary, never a local-vol evaluation point. No variance clipping is used.
    """

    surface: SSVISurface
    curves: MarketCurves
    time_offset: float = 0.
    anchor_spot: float | None = None
    calibration_json: str | None = None
    observed_support: tuple[tuple[float, float, float], ...] = ()
    _validated_calibration: _AcceptedCalibration | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self._validate()
        if self.surface.day_count != self.curves.day_count:
            raise ValueError('surface and curve day_count must match')
        support = tuple(tuple(float(v) for v in row) for row in self.observed_support)
        if any(len(row) != 3 or not np.all(np.isfinite(row)) or row[0] <= 0 or row[1] > row[2]
               for row in support):
            raise ValueError('invalid observed support')
        object.__setattr__(self, 'observed_support', support)
        evidence = self._validated_calibration
        if self.calibration_json is None:
            if evidence is not None:
                raise ValueError('validated calibration requires its matching evidence')
        else:
            canonical = json.dumps(json.loads(self.calibration_json), sort_keys=True,
                                   separators=(',', ':'), allow_nan=False)
            if evidence is None:
                # Public raw evidence is untrusted. Validate once at construction;
                # internal immutable proof then survives roll/rebase without fitting.
                fit = CalibrationResult.from_dict(json.loads(canonical))
                if not fit.accepted or fit.surface != self.surface:
                    raise ValueError('calibration does not validate this surface')
                evidence = self.from_calibration(fit)._validated_calibration
                canonical = evidence.json
            if (not isinstance(evidence, _AcceptedCalibration)
                    or evidence.json != canonical or evidence.surface != self.surface):
                raise ValueError('calibration evidence and fitted surface must match')
            object.__setattr__(self, 'calibration_json', evidence.json)
            object.__setattr__(self, '_validated_calibration', evidence)
            object.__setattr__(self, 'observed_support', evidence.support)

    @classmethod
    def theoretical(cls, surface, curves, **kwargs):
        if kwargs.get('calibration_json') is not None or kwargs.get('_validated_calibration') is not None:
            raise ValueError('theoretical models cannot carry accepted calibration evidence')
        return cls(surface, curves, **kwargs)

    @classmethod
    def from_calibration(cls, fit, **kwargs):
        if not isinstance(fit, CalibrationResult) or not fit.accepted:
            raise ValueError('local-vol model requires an accepted calibration')
        quotes = fit.validation.accepted
        support = tuple((t, min(np.log(q.strike/q.forward) for q in quotes if q.model_time == t),
                          max(np.log(q.strike/q.forward) for q in quotes if q.model_time == t))
                        for t in sorted({q.model_time for q in quotes}))
        canonical = json.dumps(fit.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return cls(fit.surface, fit.validation.curves, calibration_json=canonical,
            observed_support=support,
            _validated_calibration=_AcceptedCalibration(canonical,fit.surface,support), **kwargs)

    def is_deterministic(self, horizon):
        """Monotone theta certifies zero variance over the entire future interval."""
        end = self.time_offset + float(_time(horizon))
        return self.surface.atm_total_variance(end) == self.surface.atm_total_variance(self.time_offset)

    def local_vol(self, t, spot):
        absolute = self.time_offset + _time(t)
        spot = np.asarray(spot, dtype=float)
        if np.any(~np.isfinite(spot)) or np.any(spot <= 0):
            raise ValueError('local volatility requires finite positive spot')
        # Difference of logs is safe even when spot/F would overflow or underflow.
        k = np.log(spot) - np.log(self.curves.forward(absolute))
        d = self.surface.derivatives(k, absolute)
        theta = np.asarray(self.surface.atm_total_variance(absolute))
        phi = self.surface.eta/((theta+self.surface.theta_floor)**self.surface.gamma
                               *(1+theta)**(1-self.surface.gamma))
        x = phi*k
        root = np.sqrt((x+self.surface.rho)**2 + 1-self.surface.rho**2)
        shape = 1+self.surface.rho*x+root
        ratio = phi*(self.surface.rho+(x+self.surface.rho)/root)/shape
        # wk/w ratio has a finite analytic limit at w=0, unlike direct division.
        denominator = (1-k*ratio/2)**2 - (d.w*ratio**2 + .25*d.w_k**2)/4 + d.w_kk/2
        if (np.any(~np.isfinite(denominator)) or np.any(denominator <= 0)
                or np.any(~np.isfinite(d.w_t)) or np.any(np.asarray(d.w_t) < 0)
                or np.any(np.asarray(d.w) < 0)):
            raise ValueError('invalid Dupire variance or nonpositive density denominator')
        variance = d.w_t/denominator
        if np.any(~np.isfinite(variance)) or np.any(variance < 0):
            raise ValueError('invalid or nonfinite Dupire local variance')
        return _scalar(np.sqrt(variance))

    def to_dict(self):
        return dict(kind='ssvi_dupire', surface=self.surface.to_dict(), curves=self.curves.to_dict(),
            time_offset=self.time_offset, anchor_spot=self.anchor_spot,
            calibration=None if self.calibration_json is None else json.loads(self.calibration_json),
            observed_support=[list(row) for row in self.observed_support])

    @classmethod
    def from_dict(cls, data):
        if data['kind'] != 'ssvi_dupire':
            raise ValueError('expected ssvi_dupire model')
        calibration = data.get('calibration')
        support = data.get('observed_support', ())
        if calibration is not None:
            fit = CalibrationResult.from_dict(calibration)
            if not fit.accepted:
                raise ValueError('serialized calibration fails revalidation')
            if fit.surface.to_dict() != data['surface']:
                raise ValueError('serialized calibrated surface differs from revalidated fit')
            base = cls.from_calibration(fit, time_offset=data['time_offset'], anchor_spot=data['anchor_spot'])
            return replace(base, curves=MarketCurves.from_dict(data['curves']))
        return cls.theoretical(SSVISurface.from_dict(data['surface']), MarketCurves.from_dict(data['curves']),
            time_offset=data['time_offset'], anchor_spot=data['anchor_spot'], observed_support=support)

    def diagnostics(self):
        return dict(kind='ssvi_dupire', identity=self.identity, time_offset=self.time_offset,
            anchor_spot=self.anchor_spot, day_count=self.day_count,
            provenance='research/theoretical' if self.calibration_json is None else 'accepted calibration',
            surface=self.surface.to_dict(), observed_support=[list(row) for row in self.observed_support],
            extrapolation='SSVI wings at all finite log-moneyness; PCHIP short end from (0,0); terminal theta secant',
            observed_support_scope='quoted strikes at quoted tenors only; all other points are modeled',
            zero_time_policy='analytic right limit', invalid_variance_policy='raise; no clipping or floors')


@dataclass(frozen=True)
class TimeVolModel(_CurveModel):
    """Research model: piecewise-constant deterministic instantaneous volatility.

    Volatility i applies from times[i] inclusive until times[i+1]; final value
    extends indefinitely. Knots are absolute market times, independent of roll.
    """

    curves: MarketCurves
    times: tuple[float, ...]
    volatilities: tuple[float, ...]
    time_offset: float = 0.
    anchor_spot: float | None = None

    def __post_init__(self):
        self._validate()
        object.__setattr__(self, 'times', _finite_tuple(self.times))
        object.__setattr__(self, 'volatilities', _finite_tuple(self.volatilities))
        if (self.times[0] != 0 or len(self.times) != len(self.volatilities)
                or np.any(np.diff(self.times) <= 0) or min(self.volatilities) < 0):
            raise ValueError('time volatility needs increasing times from zero and nonnegative volatilities')

    def local_vol(self, t, spot):
        t, spot = np.broadcast_arrays(self.time_offset + _time(t), np.asarray(spot, dtype=float))
        if np.any(~np.isfinite(spot)) or np.any(spot <= 0):
            raise ValueError('local volatility requires finite positive spot')
        return _scalar(np.asarray(self.volatilities)[np.searchsorted(self.times, t, side='right')-1])

    def is_deterministic(self, horizon):
        """Check every active piece, including knots inside the valuation interval."""
        end = self.time_offset + float(_time(horizon))
        knots = np.asarray(self.times)
        active = np.r_[self.time_offset, knots[(knots>self.time_offset)&(knots<end)]]
        indices = np.searchsorted(knots, active, side='right')-1
        return bool(np.all(np.asarray(self.volatilities)[indices] == 0.))

    def to_dict(self):
        return dict(kind='time_volatility', curves=self.curves.to_dict(), times=list(self.times),
            volatilities=list(self.volatilities), time_offset=self.time_offset, anchor_spot=self.anchor_spot)

    @classmethod
    def from_dict(cls, data):
        if data['kind'] != 'time_volatility':
            raise ValueError('expected time_volatility model')
        return cls(MarketCurves.from_dict(data['curves']), data['times'], data['volatilities'],
                   data['time_offset'], data['anchor_spot'])

    def diagnostics(self):
        return dict(kind='time_volatility', identity=self.identity, time_offset=self.time_offset,
            anchor_spot=self.anchor_spot, day_count=self.day_count, provenance='research/theoretical')


def validate_model(model, product, market):
    if product.n_assets != 1:
        raise ValueError('local models support single-asset products only')
    if model.day_count != f'ACT/{product.day_count}':
        raise ValueError('product and model day_count must match')
    if not np.isclose(market.spots[0], model.anchor_spot, rtol=1e-12, atol=0):
        raise ValueError('model anchor must match market spot; use rebase or rolled explicitly')


def checked_local_vol(model, t, spot):
    """Validate even user-supplied protocol implementations at the engine boundary."""
    volatility = np.asarray(model.local_vol(t, spot), dtype=float)
    if np.any(~np.isfinite(volatility)) or np.any(volatility < 0):
        raise ValueError('model returned invalid local volatility')
    return volatility
