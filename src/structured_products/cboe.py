"""Offline Cboe EOD sample adapter; synchronous 15:45 fields only.

No downloads, vendor IV reuse, holiday inference, cross-root parity or quote repair.
The caller supplies an explicit exchange schedule with timezone-labelled opens and
closes and source evidence. AM is a conventional opening-fixing label; component
opens can be asynchronous, and separate settlement cash lag is not modeled.
"""
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .marketdata import MarketInput, MarketSnapshot, MarketCurves, TradingCalendar, QuotePolicy
from .workflow_io import write_csv, write_json


@dataclass(frozen=True)
class CboeAdapterResult:
    inputs: MarketInput
    parity: tuple[dict,...]
    provenance: dict


def adapt_cboe_eod(raw_csv,schedule,snapshot_timestamp,*,root='SPX',expiries=None,
                   normalize_spot=100.,strike_range=(.8,1.2),parity_range=(.9,1.1),
                   min_pairs=9,quote_policy=None,output_directory=None):
    if root not in ('SPX','SPXW'): raise ValueError('root must be SPX (AM) or SPXW (PM)')
    snap=datetime.fromisoformat(snapshot_timestamp)
    if snap.utcoffset() is None or (snap.hour,snap.minute,snap.second)!=(15,45,0):
        raise ValueError('snapshot requires explicit Eastern 15:45 label and timezone offset')
    if snap.utcoffset().total_seconds() not in (-4*3600,-5*3600):
        raise ValueError('snapshot offset must follow declared Eastern session')
    sessions=schedule['sessions']
    if not schedule.get('source'): raise ValueError('explicit exchange schedule requires source evidence')
    if not sessions or sessions[0]['date']!=snap.date().isoformat():
        raise ValueError('schedule must start at snapshot session')
    dates=[s['date'] for s in sessions]
    if dates!=sorted(set(dates)): raise ValueError('schedule dates must be unique and sorted')
    opening=datetime.fromisoformat(sessions[0]['open']); closing=datetime.fromisoformat(sessions[0]['close'])
    if snap.utcoffset()!=opening.utcoffset() or snap.utcoffset()!=closing.utcoffset():
        raise ValueError('snapshot offset must match its supplied Eastern cash session')
    fraction=(snap-opening).total_seconds()/(closing-opening).total_seconds()
    if not 0<=fraction<=1: raise ValueError('snapshot is outside its explicit cash session')
    with Path(raw_csv).open(encoding='utf-8-sig',newline='') as f: raw=list(csv.DictReader(f))
    selected=[(i,r) for i,r in enumerate(raw,1) if r['underlying_symbol']=='^SPX' and r['root']==root
              and (expiries is None or r['expiration'] in expiries)]
    if not selected: raise ValueError('no quotes for selected root/expiry')
    bids={float(r['underlying_bid_1545']) for _,r in selected}
    asks={float(r['underlying_ask_1545']) for _,r in selected}
    if (len(bids)!=1 or len(asks)!=1 or any(r['quote_date']!=snap.date().isoformat() for _,r in selected)
            or not 0<next(iter(bids))<=next(iter(asks))):
        raise ValueError('synchronous date/underlying bid/ask required across selected root')
    spot=(next(iter(bids))+next(iter(asks)))/2
    factor=normalize_spot/spot
    if not np.isfinite(factor) or factor<=0: raise ValueError('normalization must be finite positive')
    selected=[(i,r) for i,r in selected if strike_range[0]*spot<=float(r['strike'])<=strike_range[1]*spot]
    settlement='AM' if root=='SPX' else 'PM'
    labels={s['date']:s['open' if settlement=='AM' else 'close'] for s in sessions}
    times={s['date']:(i+(0 if settlement=='AM' else 1)-fraction)/252 for i,s in enumerate(sessions)}
    # Include every actual session, retaining one ACT/252 day for shortened sessions.
    endpoints=sessions if settlement=='PM' and fraction<1 else sessions[1:]
    calendar=TradingCalendar((snapshot_timestamp,)+tuple(labels[s['date']] for s in endpoints),
        (0.,)+tuple(times[s['date']] for s in endpoints),
        (fraction,)+(0. if settlement=='AM' else 1.,)*len(endpoints))
    groups=defaultdict(lambda:defaultdict(dict))
    for _,r in selected:
        expiry=r['expiration']; strike=float(r['strike']); kind=r['option_type']
        if kind in groups[expiry][strike]: raise ValueError('duplicate root/expiry/strike/option')
        groups[expiry][strike][kind]=r
    policy=quote_policy or QuotePolicy()
    def good(r):
        b,a=float(r['bid_1545']),float(r['ask_1545'])
        return (0<b<=a and float(r['bid_size_1545'])>0 and float(r['ask_size_1545'])>0
            and a-b<=policy.max_absolute_spread and (a-b)/max((a+b)/2,policy.relative_price_floor)<=policy.max_relative_spread)
    nodes=[(0.,normalize_spot,1.)]; parity=[]
    for expiry in sorted(groups):
        if expiry not in labels or times[expiry]<=0: raise ValueError('expiry absent from future explicit exchange sessions')
        pairs=[]
        for k,pair in sorted(groups[expiry].items()):
            if parity_range[0]*spot<=k<=parity_range[1]*spot and set(pair)=={'C','P'} and all(good(r) for r in pair.values()):
                c,p=pair['C'],pair['P']
                pairs.append((k,float(c['bid_1545'])-float(p['ask_1545']),float(c['ask_1545'])-float(p['bid_1545'])))
        if len(pairs)<min_pairs: raise ValueError(f'insufficient synchronous quality parity pairs for {expiry}: {len(pairs)} < {min_pairs}')
        x,low,high=np.asarray(pairs).T; y=(low+high)/2
        slope,intercept=np.polyfit(x-spot,y,1); d=-slope; a=intercept+d*spot; f=a/d
        prediction=a-d*x
        inside=(prediction>=low-policy.parity_tolerance)&(prediction<=high+policy.parity_tolerance)
        if not np.all(np.isfinite((d,f))) or min(d,f)<=0 or not all(inside):
            raise ValueError(f'parity F/D gate failed for {expiry}; no curve repair')
        parity.append(dict(expiry=expiry,expiry_label=labels[expiry],model_time=times[expiry],pairs=len(pairs),
            raw_forward=float(f),discount_factor=float(d),raw_rmse=float(np.sqrt(np.mean((prediction-y)**2))),
            raw_max_abs_residual=float(np.max(abs(prediction-y))),bid_ask_consistent_pairs=int(sum(inside)),
            all_bid_ask_consistent=bool(all(inside)),strikes=x.tolist()))
        nodes.append((times[expiry],float(f*factor),float(d)))
    curves=MarketCurves(*(tuple(row[i] for row in nodes) for i in range(3)))
    provenance=dict(kind='real',source='https://datashop.cboe.com/download/sample/217',root=root,
        raw_csv_sha256=hashlib.sha256(Path(raw_csv).read_bytes()).hexdigest(),raw_spot=str(spot),
        normalization_factor=str(factor),calendar_source=str(schedule['source']),
        clock='ACT/252 supplied cash sessions and explicit intraday fractions',
        AM_fixing='conventional opening label; component opens asynchronous; separate cash lag omitted',
        usage='Cboe demonstration sample; keep vendor raw and derived quotes local; redistribution not established')
    snapshot=MarketSnapshot(root,normalize_spot,'USD','index points normalized to snapshot spot',snapshot_timestamp,
        'explicit_calendar',provenance)
    rows=tuple(MappingProxyType(dict(symbol=root,quote_timestamp=snapshot_timestamp,expiry=labels[r['expiration']],
        settlement=settlement,strike=float(r['strike'])*factor,option_type={'C':'call','P':'put'}[r['option_type']],
        bid=float(r['bid_1545'])*factor,ask=float(r['ask_1545'])*factor,exercise_style='European',
        volume=r['trade_volume'],open_interest=r['open_interest'],source_row_number=i,source_root=root,
        source_bid_size=r['bid_size_1545'],source_ask_size=r['ask_size_1545'])) for i,r in selected)
    inputs=MarketInput(snapshot,curves,calendar,rows)
    if output_directory is not None:
        directory=Path(output_directory); directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/'snapshot.json',snapshot.to_dict())
        write_json(directory/'parity-curves.json',parity)
        write_json(directory/'session-schedule.json',schedule)
        write_csv(directory/'curves.csv',[dict(model_time=t,forward=f,discount_factor=d) for t,f,d in nodes])
        write_csv(directory/'calendar.csv',[dict(date=d,model_time=t,session_fraction=f)
            for d,t,f in zip(calendar.dates,calendar.model_times,calendar.session_fractions)])
        write_csv(directory/'quotes.csv',[dict(r) for r in rows])
    return CboeAdapterResult(inputs,tuple(parity),provenance)
