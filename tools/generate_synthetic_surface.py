"""Regenerate the explicitly synthetic European market example (no live data)."""

import csv
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from structured_products import MarketCurves, MarketSnapshot, SSVISurface, black_price


def generate(directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    days = [date(2026,1,2)]
    while len(days)<505:
        next_day = days[-1]+timedelta(days=1)
        while next_day.weekday()>=5:
            next_day += timedelta(days=1)
        days.append(next_day)
    tenors = (21,63,126,252,504)
    times = tuple(n/252 for n in tenors)
    c = MarketCurves((0.,)+times,tuple(100*np.exp(.015*t) for t in (0.,)+times),
                     tuple(np.exp(-.03*t) for t in (0.,)+times))
    s = SSVISurface(times,(.004,.011,.024,.052,.104),rho=-.45,eta=.65,gamma=.3)
    snapshot = MarketSnapshot('SYNTH',100,'USD','index_points','2026-01-02T21:00:00+00:00','PM',
                              (('kind','synthetic'),('source','tools/generate_synthetic_surface.py'),
                               ('calendar','synthetic weekdays; holidays deliberately not modeled'),
                               ('real_data_acceptance','pending')))
    def write_csv(name,fields,rows):
        with (directory/name).open('w',encoding='utf-8',newline='') as handle:
            writer = csv.writer(handle,lineterminator='\n')
            writer.writerow(fields)
            writer.writerows(rows)
    (directory/'snapshot.json').write_text(json.dumps(snapshot.to_dict(),indent=2)+'\n',encoding='utf-8')
    (directory/'generating_surface.json').write_text(json.dumps(s.to_dict(),indent=2)+'\n',encoding='utf-8')
    write_csv('curves.csv',('model_time','forward','discount_factor'),zip(c.times,c.forwards,c.discounts))
    write_csv('calendar.csv',('date','model_time'),((d.isoformat(),i/252) for i,d in enumerate(days)))
    rows = []
    for n,t in zip(tenors,times):
        f,d = c.forward(t),c.discount(t)
        for k in np.linspace(-.2,.2,9):
            strike = f*np.exp(k)
            for kind in ('call','put'):
                p = black_price(f,strike,d,s.total_variance(k,t),kind)
                half_spread = min(.02,.1*p)
                rows.append(('SYNTH',snapshot.timestamp,days[n].isoformat(),'PM',strike,kind,
                             p-half_spread,p+half_spread,'European',100,1000))
    write_csv('quotes.csv',('symbol','quote_timestamp','expiry','settlement','strike','option_type',
                            'bid','ask','exercise_style','volume','open_interest'),rows)


if __name__ == '__main__':
    generate(Path(__file__).resolve().parents[1]/'src'/'structured_products'/'data'/'market_surface')
