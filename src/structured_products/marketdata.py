"""Immutable ACT/252 market inputs and auditable European quote ingestion."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _finite_tuple(values):
    result = tuple(float(x) for x in values)
    if not result or not np.all(np.isfinite(result)):
        raise ValueError("nodes must be nonempty and finite")
    return result


def _time(value):
    t = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError("model time must be finite and nonnegative (ACT/252)")
    return t


def _scalar(value):
    return float(value) if np.ndim(value) == 0 else value


def _timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.utcoffset() is None:
        raise ValueError("timestamp must have an explicit timezone")
    return parsed


@dataclass(frozen=True)
class MarketCurves:
    """Log-linear positive F(0,t), D(0,t); terminal log slopes extrapolate."""

    times: tuple[float, ...]
    forwards: tuple[float, ...]
    discounts: tuple[float, ...]
    day_count: str = 'ACT/252'

    def __post_init__(self):
        for name in ('times', 'forwards', 'discounts'):
            object.__setattr__(self, name, _finite_tuple(getattr(self, name)))
        if (self.day_count != 'ACT/252' or len(self.times) < 2
                or len(self.times) != len(self.forwards) or len(self.times) != len(self.discounts)
                or self.times[0] != 0 or np.any(np.diff(self.times) <= 0)
                or min(self.forwards) <= 0 or min(self.discounts) <= 0 or self.discounts[0] != 1):
            raise ValueError("curves require ACT/252 increasing nodes from 0, positive F/D and D(0)=1")

    def _log_value(self, t, values):
        t = _time(t)
        logs = np.log(values)
        result = np.interp(t, self.times, logs)
        slope = (logs[-1]-logs[-2])/(self.times[-1]-self.times[-2])
        result = np.where(t > self.times[-1], logs[-1]+slope*(t-self.times[-1]), result)
        with np.errstate(over='ignore', under='ignore'):
            value = np.exp(result)
        if np.any(~np.isfinite(value)) or np.any(value <= 0):
            raise ValueError("curve extrapolation is outside finite positive range")
        return _scalar(value)

    def forward(self, t):
        return self._log_value(t, self.forwards)

    def discount(self, t):
        return self._log_value(t, self.discounts)

    def discount_between(self, start, end):
        if np.any(_time(end) < _time(start)):
            raise ValueError("end must be at or after start")
        return self.discount(end)/self.discount(start)

    def forward_growth(self, start, end):
        if np.any(_time(end) < _time(start)):
            raise ValueError("end must be at or after start")
        return self.forward(end)/self.forward(start)

    def _slope(self, t, values):
        t = _time(t)
        slopes = np.diff(np.log(values))/np.diff(self.times)
        i = np.minimum(np.searchsorted(self.times[1:], t, side='right'), len(slopes)-1)
        return _scalar(slopes[i])

    def instantaneous_rate(self, t):
        return -self._slope(t, self.discounts)

    def instantaneous_carry(self, t):
        return self._slope(t, self.forwards)

    def to_dict(self):
        return dict(times=list(self.times), forwards=list(self.forwards),
                    discounts=list(self.discounts), day_count=self.day_count)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass(frozen=True)
class TradingCalendar:
    """Explicit sessions; optional fractions locate intraday events in a session.

    Distinct dates are successive supplied trading sessions; no holiday inference.
    A fraction is in [0,1], measured from that session's open to close. Timestamp
    labels require fractions; they are labels, never inferred trading hours.
    """

    dates: tuple[str, ...]
    model_times: tuple[float, ...]
    session_fractions: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'dates', tuple(str(d) for d in self.dates))
        object.__setattr__(self, 'model_times', _finite_tuple(self.model_times))
        fractions = tuple(float(x) for x in self.session_fractions)
        explicit = bool(fractions)
        if not explicit:
            fractions = (0.,)*len(self.dates)
        object.__setattr__(self, 'session_fractions', fractions)
        if len(self.dates) != len(self.model_times) or len(fractions) != len(self.dates):
            raise ValueError("calendar columns must have equal lengths")
        sessions = []
        for label in self.dates:
            if 'T' in label:
                if not explicit:
                    raise ValueError("intraday timestamps require explicit session fractions")
                instant = _timestamp(label)
                sessions.append(instant.date())
            else:
                day = date.fromisoformat(label)
                sessions.append(day)
        if len(set(self.dates)) != len(self.dates) or any(b < a for a,b in zip(sessions,sessions[1:])):
            raise ValueError("calendar labels must be unique and ordered")
        if not np.all(np.isfinite(fractions)) or min(fractions) < 0 or max(fractions) > 1:
            raise ValueError("session fractions must be finite in [0,1]")
        indices = np.cumsum([0]+[int(b != a) for a,b in zip(sessions,sessions[1:])])
        expected = (indices+np.asarray(fractions)-fractions[0])/252
        if (self.model_times[0] != 0 or np.any(np.diff(self.model_times) <= 0)
                or not np.allclose(expected, self.model_times, rtol=0, atol=1e-12)):
            raise ValueError("calendar times must follow supplied ACT/252 sessions and fractions")

    def time(self, value):
        key = value.isoformat() if isinstance(value, (date,datetime)) else str(value)
        try:
            return self.model_times[self.dates.index(key)]
        except ValueError as exc:
            raise ValueError(f"expiry {key!r} is absent from explicit trading calendar") from exc

    def to_dict(self):
        return dict(dates=list(self.dates), model_times=list(self.model_times),
                    session_fractions=list(self.session_fractions))

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    spot: float
    currency: str
    price_unit: str
    timestamp: str
    settlement_convention: str
    provenance: tuple[tuple[str, str], ...]
    day_count: str = 'ACT/252'

    def __post_init__(self):
        object.__setattr__(self, 'spot', float(self.spot))
        provenance = self.provenance.items() if isinstance(self.provenance, Mapping) else self.provenance
        object.__setattr__(self, 'provenance', tuple((str(k),str(v)) for k,v in provenance))
        _timestamp(self.timestamp)
        if (not np.isfinite(self.spot) or self.spot <= 0 or not self.symbol
                or not self.currency or not self.price_unit or not self.provenance
                or any(not k or not v for k,v in self.provenance)
                or len(dict(self.provenance)) != len(self.provenance)
                or self.day_count != 'ACT/252'
                or self.settlement_convention not in ('PM','explicit_calendar')):
            raise ValueError("snapshot needs positive spot, units, provenance and explicit ACT/252 settlement")

    def to_dict(self):
        return dict(symbol=self.symbol, spot=self.spot, currency=self.currency,
                    price_unit=self.price_unit, timestamp=self.timestamp,
                    settlement_convention=self.settlement_convention,
                    provenance=dict(self.provenance), day_count=self.day_count)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass(frozen=True)
class QuotePolicy:
    max_relative_spread: float = .5
    max_absolute_spread: float = 5.
    relative_price_floor: float = .1
    parity_tolerance: float = 1e-8

    def __post_init__(self):
        values = (self.max_relative_spread,self.max_absolute_spread,self.relative_price_floor,self.parity_tolerance)
        if not np.all(np.isfinite(values)) or min(values[:3]) <= 0 or self.parity_tolerance < 0:
            raise ValueError("invalid quote quality policy")


@dataclass(frozen=True)
class ValidatedQuote:
    row_number: int
    expiry: str
    settlement: str
    model_time: float
    strike: float
    option_type: str
    bid: float
    ask: float
    forward: float
    discount: float
    implied_vol: float

    @property
    def mid(self):
        return (self.bid+self.ask)/2

    def to_dict(self):
        return dict(vars(self), mid=self.mid)


@dataclass(frozen=True)
class QuoteRecord:
    row_number: int
    raw: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]
    quote: ValidatedQuote | None = None

    @property
    def accepted(self):
        return not self.reasons and self.quote is not None

    def to_dict(self):
        return dict(row_number=self.row_number, raw=dict(self.raw), accepted=self.accepted,
                    reasons=list(self.reasons), quote=None if self.quote is None else self.quote.to_dict())


@dataclass(frozen=True)
class QuoteValidationResult:
    snapshot: MarketSnapshot
    curves: MarketCurves
    calendar: TradingCalendar
    policy: QuotePolicy
    records: tuple[QuoteRecord, ...]

    @property
    def accepted(self):
        return tuple(r.quote for r in self.records if r.accepted)

    def to_dict(self):
        return dict(snapshot=self.snapshot.to_dict(),curves=self.curves.to_dict(),
                    calendar=self.calendar.to_dict(),policy=dict(vars(self.policy)),
                    records=[r.to_dict() for r in self.records])


def validate_quotes(rows, snapshot, calendar, curves, policy=None):
    """Validate independently, then reject all duplicate/parity-inconsistent rows.

    Raw cells are preserved as text, including invalid numeric cells, so audit JSON
    is finite. Bounds/parity use executable bid/ask intervals; IV uses midpoints.
    """
    from .volatility import implied_volatility

    policy = policy or QuotePolicy()
    if not np.isclose(snapshot.spot,curves.forwards[0],rtol=1e-12,atol=0):
        raise ValueError("curve forward at zero must equal snapshot spot")
    anchor = calendar.dates[0]
    if (('T' in anchor and _timestamp(anchor) != _timestamp(snapshot.timestamp))
            or ('T' not in anchor and anchor != _timestamp(snapshot.timestamp).date().isoformat())):
        raise ValueError("calendar origin must match snapshot date/timestamp")
    raw_rows, reasons, candidates = [], [], {}
    required = ('symbol','quote_timestamp','expiry','settlement','strike','option_type','bid','ask','exercise_style')
    for number,row in enumerate(rows,1):
        raw = {str(k): '' if v is None else str(v) for k,v in row.items()}
        raw_rows.append(tuple(raw.items()))
        errors = []
        reasons.append(errors)
        if any(not raw.get(k,'').strip() for k in required):
            errors.append('missing_required')
            continue
        try:
            strike,bid,ask = (float(raw[k]) for k in ('strike','bid','ask'))
            if not np.all(np.isfinite((strike,bid,ask))) or strike <= 0 or bid < 0 or ask <= 0:
                raise ValueError()
            for field in ('volume','open_interest'):
                if raw.get(field,'').strip():
                    value = float(raw[field])
                    if not np.isfinite(value) or value < 0 or not value.is_integer():
                        raise ValueError()
        except (ValueError,TypeError):
            errors.append('invalid_numeric')
            continue
        try:
            if _timestamp(raw['quote_timestamp']) != _timestamp(snapshot.timestamp):
                errors.append('timestamp_mismatch')
        except ValueError:
            errors.append('invalid_timestamp')
        if raw['symbol'] != snapshot.symbol:
            errors.append('symbol_mismatch')
        if raw['exercise_style'].lower() != 'european':
            errors.append('non_european')
        kind = raw['option_type'].lower()
        if kind not in ('call','put'):
            errors.append('invalid_option_type')
        settlement = raw['settlement'].upper()
        if (settlement not in ('AM','PM')
                or (snapshot.settlement_convention=='PM' and settlement!='PM')
                or (settlement=='AM' and 'T' not in raw['expiry'])):
            errors.append('ambiguous_settlement')
        try:
            t = calendar.time(raw['expiry'])
            if t <= 0:
                errors.append('expired')
        except ValueError:
            errors.append('missing_calendar_expiry')
            continue
        if bid > ask:
            errors.append('crossed_quote')
        spread = ask-bid
        mid = (bid+ask)/2
        if spread > policy.max_absolute_spread or spread/max(mid,policy.relative_price_floor) > policy.max_relative_spread:
            errors.append('wide_quote')
        if errors:
            continue
        f,d = curves.forward(t),curves.discount(t)
        intrinsic = d*max((f-strike) if kind=='call' else (strike-f),0)
        upper = d*(f if kind=='call' else strike)
        if ask < intrinsic-policy.parity_tolerance or bid > upper+policy.parity_tolerance:
            errors.append('option_bounds')
            continue
        try:
            iv = implied_volatility(mid,f,strike,d,t,kind)
        except ValueError:
            errors.append('iv_not_identifiable')
            continue
        candidates[number] = ValidatedQuote(number,raw['expiry'],settlement,t,strike,kind,bid,ask,f,d,iv)
    groups = {}
    for number,q in candidates.items():
        groups.setdefault((q.expiry,q.settlement,q.strike,q.option_type),[]).append(number)
    for group in groups.values():
        if len(group)>1:
            for number in group:
                reasons[number-1].append('duplicate')
    pairs = {}
    for number,q in candidates.items():
        if not reasons[number-1]:
            pairs.setdefault((q.expiry,q.settlement,q.strike),{})[q.option_type] = q
    for pair in pairs.values():
        if len(pair)==2:
            c,p = pair['call'],pair['put']
            parity = c.discount*(c.forward-c.strike)
            if parity < c.bid-p.ask-policy.parity_tolerance or parity > c.ask-p.bid+policy.parity_tolerance:
                reasons[c.row_number-1].append('put_call_parity')
                reasons[p.row_number-1].append('put_call_parity')
    records = tuple(QuoteRecord(i,raw,tuple(reasons[i-1]),candidates.get(i)) for i,raw in enumerate(raw_rows,1))
    return QuoteValidationResult(snapshot,curves,calendar,policy,records)


@dataclass(frozen=True)
class MarketInput:
    snapshot: MarketSnapshot
    curves: MarketCurves
    calendar: TradingCalendar
    rows: tuple[Mapping[str,str], ...]


def load_market_data(directory):
    """Read snapshot.json, curves.csv, calendar.csv and quotes.csv without repair."""
    directory = Path(directory)
    def read_csv(name):
        with (directory/name).open(encoding='utf-8-sig',newline='') as handle:
            return tuple(dict(row) for row in csv.DictReader(handle))
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    snapshot = MarketSnapshot.from_dict(json.loads((directory/'snapshot.json').read_text(encoding='utf-8-sig'),parse_constant=reject_constant))
    nodes = read_csv('curves.csv')
    curves = MarketCurves(tuple(float(n['model_time']) for n in nodes),
                          tuple(float(n['forward']) for n in nodes),tuple(float(n['discount_factor']) for n in nodes))
    sessions = read_csv('calendar.csv')
    fractions = tuple(float(n['session_fraction']) for n in sessions) if sessions and 'session_fraction' in sessions[0] else ()
    calendar = TradingCalendar(tuple(n['date'] for n in sessions),tuple(float(n['model_time']) for n in sessions),fractions)
    return MarketInput(snapshot,curves,calendar,tuple(MappingProxyType(r) for r in read_csv('quotes.csv')))
