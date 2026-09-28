"""Rebuild the frozen final RF from official training data, without test input."""
import argparse,json,zipfile
from pathlib import Path
import joblib,numpy as np
from cuhkx.core import read_csv
from cuhkx.imu_emotion import feature_record,train_model,prior_fit
parser=argparse.ArgumentParser();parser.add_argument('training_qa',type=Path);parser.add_argument('sensor_zip',type=Path);parser.add_argument('output_dir',type=Path);args=parser.parse_args()
package=Path(__file__).resolve().parents[1]
cfg=json.loads((package/'configs/sensor_config.json').read_text())
rows=[r for r in read_csv(args.training_qa) if r['category']=='emotion']
features=[];fitting=[];columns=None
with zipfile.ZipFile(args.sensor_zip) as z:
 names=set(z.namelist())
 for row in rows:
  f,a=feature_record(z,names,row,'train',cfg)
  if columns is None:columns=list(f)
  assert list(f)==columns
  if a['usable']:features.append(list(f.values()));fitting.append(row)
out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
if (out/'final_model.joblib').exists():raise RuntimeError('Use a new output directory')
model=train_model(np.asarray(features,dtype=np.float32),fitting,6,cfg)
joblib.dump(model,out/'final_model.joblib',compress=3)
(out/'prior.json').write_text(json.dumps([dict(c) for c in prior_fit(fitting)],indent=2))
(out/'feature_columns.json').write_text(json.dumps(columns,indent=2))
(out/'sensor_config.json').write_text(json.dumps(cfg,indent=2))
print('Training records:',len(fitting))
