"""Shared strict finite loaders and audit writers for file-based workflows."""
import csv
import json
from pathlib import Path

from .models import LocalVolModel, TimeVolModel
from .marketdata import MarketCurves
from .volatility import CalibrationResult, SSVISurface


def read_json(path):
    def reject(value): raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rows=[{k:json.dumps(v,allow_nan=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()} for row in rows]
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def load_model(*,model=None,calibration=None,surface=None,curves=None):
    if sum(p is not None for p in (model,calibration,surface))!=1:
        raise ValueError('provide exactly one model/calibration/surface')
    if curves and not surface: raise ValueError('--curves is only used with --surface')
    if model:
        data=read_json(model)
        return (TimeVolModel if data['kind']=='time_volatility' else LocalVolModel).from_dict(data)
    if calibration: return LocalVolModel.from_calibration(CalibrationResult.read_json(calibration))
    if not curves: raise ValueError('--surface requires --curves')
    if Path(curves).suffix.lower()=='.csv':
        with Path(curves).open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        curve=MarketCurves(tuple(float(r['model_time']) for r in rows),tuple(float(r['forward']) for r in rows),
                           tuple(float(r['discount_factor']) for r in rows))
    else: curve=MarketCurves.from_dict(read_json(curves))
    return LocalVolModel.theoretical(SSVISurface.from_dict(read_json(surface)),curve)
