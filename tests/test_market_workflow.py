import csv
from dataclasses import replace
from importlib.resources import files
import json
import pytest
import structured_products as sp
from structured_products.cli import main


def test_calibration_cli_and_provenance_labels(tmp_path,capsys):
    source=files('structured_products').joinpath('data/market_surface')
    assert main(['calibrate','--market-input',str(source),'--output',str(tmp_path/'fit.json')])==0
    saved=json.loads((tmp_path/'fit.json').read_text())
    assert saved['accepted'] and saved['real_data_acceptance']=='synthetic_accepted; real_market_pending'
    assert (tmp_path/'fit.cleaning.csv').is_file()
    data=sp.load_market_data(source)
    snap=replace(data.snapshot,provenance=(('kind','real'),('source','test fixture, not market data')))
    rejected=sp.calibrate_ssvi(sp.validate_quotes(data.rows[:1],snap,data.calendar,data.curves))
    assert rejected.to_dict()['real_data_acceptance']=='rejected_quality'
    capsys.readouterr()


def test_model_price_fair_and_stress_terms_cli_file_roundtrips(tmp_path,capsys):
    c=sp.MarketCurves((0.,1.),(100.,101.),(1.,.97))
    m=sp.LocalVolModel.theoretical(sp.SSVISurface((.25,1.),(.01,.04),-.4,.4),c)
    path=tmp_path/'model.json'; path.write_text(json.dumps(m.to_dict()))
    args=['--example','classic_snowball','--model',str(path),'--grid-size','101']
    assert main(['price',*args,'--method','pde','--no-greeks'])==0
    target=json.loads(capsys.readouterr().out)['price']['present_value']
    assert main(['fair-coupon',*args,'--method','pde','--target-pv',str(target),'--lower','0','--upper','1'])==0
    assert abs(json.loads(capsys.readouterr().out)['fair_coupon']['residual'])<1e-5
    (tmp_path/'stress.json').write_text(json.dumps([{'name':'zero'}]))
    assert main(['stress',*args,'--scenarios',str(tmp_path/'stress.json'),'--output',str(tmp_path/'out.json'),
        '--csv-output',str(tmp_path/'out.csv')])==0
    assert json.loads((tmp_path/'out.json').read_text())['cells'][0]['holder_loss']==0
    capsys.readouterr()
    assert main(['compare-terms',*args,'--coupons','.1','--ki-ratios','.7','--ko-ratios','1',
        '--scenarios',str(tmp_path/'stress.json')])==0
    assert len(json.loads(capsys.readouterr().out)['cells'])==1
    assert main(['price',*args,'--method','pde'])==2
    assert 'no-greeks' in capsys.readouterr().err


def test_cboe_adapter_synchronous_am_pm_and_curve_checks(tmp_path):
    schedule={'source':'test synthetic explicit calendar','sessions':[
        {'date':'2023-08-25','open':'2023-08-25T09:30:00-04:00','close':'2023-08-25T16:00:00-04:00'},
        {'date':'2023-08-28','open':'2023-08-28T09:30:00-04:00','close':'2023-08-28T16:00:00-04:00'}]}
    rows=[]
    for root in ('SPX','SPXW'):
        for k in range(90,111,2):
            for kind in ('C','P'):
                price=sp.black_price(100.,k,.999,.002,'call' if kind=='C' else 'put')
                rows.append(dict(underlying_symbol='^SPX',root=root,expiration='2023-08-28',strike=k,
                    option_type=kind,quote_date='2023-08-25',underlying_bid_1545=99.99,underlying_ask_1545=100.01,
                    bid_1545=max(.001,price-.01),ask_1545=price+.01,bid_size_1545=1,ask_size_1545=1,
                    trade_volume=0,open_interest=0))
    raw=tmp_path/'raw.csv'
    with raw.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=sp.adapt_cboe_eod(raw,schedule,'2023-08-25T15:45:00-04:00',expiries=('2023-08-28',),
        min_pairs=9,output_directory=tmp_path/'canonical')
    assert result.inputs.snapshot.spot==100.
    assert all(r['settlement']=='AM' for r in result.inputs.rows)
    assert result.inputs.calendar.model_times[-1]==pytest.approx((1-25/26)/252)
    assert result.parity[0]['all_bid_ask_consistent']
    assert sp.load_market_data(tmp_path/'canonical').curves.discounts==result.inputs.curves.discounts
    (tmp_path/'schedule.json').write_text(json.dumps(schedule))
    assert main(['adapt-cboe','--raw-csv',str(raw),'--schedule',str(tmp_path/'schedule.json'),
        '--timestamp','2023-08-25T15:45:00-04:00','--root','SPX','--expiries','2023-08-28',
        '--output',str(tmp_path/'cli-canonical')])==0
    assert sp.load_market_data(tmp_path/'cli-canonical').curves.discounts==result.inputs.curves.discounts
    pm=sp.adapt_cboe_eod(raw,schedule,'2023-08-25T15:45:00-04:00',root='SPXW',expiries=('2023-08-28',),min_pairs=9)
    assert pm.inputs.rows[0]['settlement']=='PM'
    assert pm.inputs.calendar.model_times[-1]==pytest.approx((2-25/26)/252)
    same_day=[{**r,'expiration':'2023-08-25'} for r in rows if r['root']=='SPXW']
    same=tmp_path/'same.csv'
    with same.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(same_day[0]));writer.writeheader();writer.writerows(same_day)
    same_result=sp.adapt_cboe_eod(same,schedule,'2023-08-25T15:45:00-04:00',root='SPXW',min_pairs=9)
    assert same_result.inputs.calendar.time(same_result.inputs.rows[0]['expiry'])==pytest.approx((1-25/26)/252)
    # The15:45 vendor column is exchange-local; a winter -04 label would
    # otherwise look like14:45 in the supplied -05 session and silently pass.
    winter={'source':'synthetic winter fixture','sessions':[
        {'date':'2023-12-01','open':'2023-12-01T09:30:00-05:00','close':'2023-12-01T16:00:00-05:00'},
        {'date':'2023-12-04','open':'2023-12-04T09:30:00-05:00','close':'2023-12-04T16:00:00-05:00'}]}
    winter_rows=[{**r,'quote_date':'2023-12-01','expiration':'2023-12-04'} for r in rows]
    winter_csv=tmp_path/'winter.csv'
    with winter_csv.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(winter_rows[0]));writer.writeheader();writer.writerows(winter_rows)
    with pytest.raises(ValueError,match='offset'):
        sp.adapt_cboe_eod(winter_csv,winter,'2023-12-01T15:45:00-04:00',min_pairs=9)
    with pytest.raises(ValueError,match='root'): sp.adapt_cboe_eod(raw,schedule,'2023-08-25T15:45:00-04:00',root='bad')
    with pytest.raises(ValueError,match='pairs'): sp.adapt_cboe_eod(raw,schedule,'2023-08-25T15:45:00-04:00',min_pairs=99)
    rows[0]['underlying_bid_1545']=98.
    with raw.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match='synchronous'): sp.adapt_cboe_eod(raw,schedule,'2023-08-25T15:45:00-04:00')
