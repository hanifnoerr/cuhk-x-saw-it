"""Portable reconstruction of exp006 inference; new wrapper, original model rules."""
import argparse, collections, hashlib, json, os, sys, zipfile
from pathlib import Path, PurePosixPath
import joblib
import numpy as np
from cuhkx import gemini
from cuhkx.core import read_csv, write_csv, sha256, clip_path, validate_submission
from cuhkx.consistency import apply_rule
from cuhkx.imu_emotion import feature_record, prior_scores, option_scores, letters

PACKAGE = Path(__file__).resolve().parents[1]

def find_qa(root):
    choices=[root/'test_qa.csv',root/'Testing/test_qa.csv',root/'Large-Model-Track/Testing/test_qa.csv']
    found=[p for p in choices if p.is_file()]
    if len(found)!=1:raise ValueError('Supply the official Testing directory (or its track parent); expected exactly one test_qa.csv')
    return found[0]

def video_path(root, clip, work):
    relative=PurePosixPath(clip)/'IR/IR.mp4'
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe clip path')
    for folder in [root,root/'Testing',root/'Large-Model-Track/Testing']:
        p=folder/relative
        if p.is_file():return p.resolve()
    for folder in [root,root/'Testing',root/'Large-Model-Track/Testing']:
        archive=folder/'large_model_track_test.zip'
        if archive.is_file():
            with zipfile.ZipFile(archive) as z:
                matches=[n for n in z.namelist() if n==str(relative) or n.endswith('/'+str(relative))]
                if len(matches)!=1:raise ValueError('Missing or ambiguous IR clip: '+clip)
                dest=work/'videos'/relative
                dest.parent.mkdir(parents=True,exist_ok=True)
                import shutil
                with z.open(matches[0]) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
                return dest.resolve()
    raise FileNotFoundError('Missing IR video for '+clip)

class SensorDirectory:
    def __init__(self, root):
        self.files={}
        for p in root.rglob('*.csv'):
            relative=p.relative_to(root).as_posix()
            self.files['LMT_(IMU,Radar,Skeleton)/'+relative]=p
    def namelist(self):return list(self.files)
    def read(self,name):return self.files[name].read_bytes()
    def close(self):pass

def sensor_source(root):
    override=os.environ.get('CUHKX_SENSOR_PATH')
    candidates=[Path(override)] if override else [root/'LMT_(IMU,Radar,Skeleton).zip',root.parent/'LMT_(IMU,Radar,Skeleton).zip',root/'Large-Model-Track/LMT_(IMU,Radar,Skeleton).zip',root/'LMT_(IMU,Radar,Skeleton)']
    for p in candidates:
        if p.is_file():return zipfile.ZipFile(p)
        if p.is_dir():return SensorDirectory(p)
    return None

def infer(root, output):
    root=root.resolve();qa=find_qa(root);questions=read_csv(qa)
    if any('answer' in r for r in questions):raise ValueError('Inference expects label-free questions')
    required = ['final_model.joblib', 'prior.json', 'sensor_config.json', 'feature_columns.json']
    if any(not (PACKAGE/'artifacts'/name).is_file() for name in required):
        raise RuntimeError('Local model artifacts are missing. Run bash train.sh TRAINING_QA_CSV OFFICIAL_SENSOR_ZIP artifacts before inference.')
    work=Path(os.environ.get('CUHKX_RUN_DIR',str(output.parent/'cuhkx_run'))).resolve();work.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((PACKAGE/'configs/gemini_ir.json').read_text())
    gemini.settings=lambda:cfg
    gemini.directory=lambda:work
    def api_key():
        token=os.environ.get('GEMINI_API_KEY')
        if not token:raise RuntimeError('Set GEMINI_API_KEY separately. Uncached inference invokes the paid Gemini API.')
        return token
    gemini.key=api_key
    groups=collections.defaultdict(list)
    for r in questions:groups[clip_path(r)].append(r)
    predictions={}
    for clip,rows in sorted(groups.items()):
        video=video_path(qa.parent,clip,work)
        qs=[{k:r[k] for k in ['qa_id','category','question','A','B','C','D']} for r in rows]
        fingerprint=hashlib.sha256(json.dumps({'video':sha256(video),'questions':qs,'config':cfg,'prompt':gemini.PROMPT,'client':sha256(PACKAGE/'cuhkx/gemini.py')},sort_keys=True).encode()).hexdigest()
        record={'id':fingerprint,'path':str(video),'sha256':sha256(video),'bytes':video.stat().st_size,'questions':qs}
        response_path=work/'responses'/f'{fingerprint}.json';attempt=work/'attempts'/f'{fingerprint}.json'
        if response_path.exists():response=json.loads(response_path.read_text())
        else:
            api_key()
            if attempt.exists():raise RuntimeError('An earlier API attempt has no saved response; reconcile it before retrying: '+str(attempt))
            request=gemini.request_for(record,gemini.upload(record))
            gemini.save(work/'requests'/f'{fingerprint}.json',request)
            gemini.save(attempt,{'state':'sending'})
            response,_=gemini.http('/v1beta/models/'+cfg['model']+':generateContent','POST',request)
            gemini.save(response_path,response)
        try:parsed=gemini.parsed(record,response)
        except (ValueError,KeyError,TypeError) as error:
            repaired=work/'repair_responses'/f'{fingerprint}.json';repair_attempt=work/'repair_attempts'/f'{fingerprint}.json'
            if repaired.exists():response=json.loads(repaired.read_text())
            else:
                api_key()
                if repair_attempt.exists():raise RuntimeError('Uncertain repair attempt; reconcile before retrying')
                request=gemini.request_for(record,gemini.upload(record))
                request['contents'][0]['parts'].append({'text':'Previous response failed the output contract: '+str(error)+'. Re-examine the video and return every answer. For each sequence, include every nonempty option letter exactly once, even if an action is hard to observe. This is format feedback, not an answer label.'})
                gemini.save(work/'repair_requests'/f'{fingerprint}.json',request)
                gemini.save(repair_attempt,{'state':'sending'})
                response,_=gemini.http('/v1beta/models/'+cfg['model']+':generateContent','POST',request)
                gemini.save(repaired,response)
            parsed=gemini.parsed(record,response)
        gemini.save(work/'parsed'/f'{fingerprint}.json',parsed)
        predictions.update({r['qa_id']:r['prediction'] for r in parsed['answers']})
        print('Completed IR clip',clip,flush=True)
    mapped,changes=apply_rule(questions,predictions)
    model=joblib.load(PACKAGE/'artifacts/final_model.joblib')
    prior=tuple(collections.Counter(c) for c in json.loads((PACKAGE/'artifacts/prior.json').read_text()))
    sensor_cfg=json.loads((PACKAGE/'artifacts/sensor_config.json').read_text())
    columns=json.loads((PACKAGE/'artifacts/feature_columns.json').read_text())
    archive=sensor_source(root);audits=[];emotion=[r for r in questions if r['category']=='emotion']
    try:
        if archive and emotion:
            names=set(archive.namelist());features=[]
            for row in emotion:
                values,audit=feature_record(archive,names,row,'test',sensor_cfg)
                if list(values)!=columns:raise ValueError('Feature column drift')
                features.append([values[c] for c in columns]);audits.append(audit)
            x=np.asarray(features,dtype=np.float32)
            p=.5*option_scores(model,x,emotion,prior)+.5*prior_scores(emotion,prior)
            margin=np.sort(p,axis=1)[:,-1]-np.sort(p,axis=1)[:,-2];pred=letters(p)
            for i,row in enumerate(emotion):
                if audits[i]['usable'] and margin[i]>=.1:mapped[row['qa_id']]=str(pred[i])
        else:audits=[{'note':'Nonvisual input unavailable; retained IR predictions for every emotion question'}]
    finally:
        if archive:archive.close()
    sample_path=qa.parent/'sample_submission.csv'
    sample=read_csv(sample_path) if sample_path.exists() else [{'qa_id':q['qa_id'],'prediction':''} for q in questions]
    result=[{'qa_id':r['qa_id'],'prediction':mapped[r['qa_id']]} for r in sample]
    validate_submission(result,questions,sample);write_csv(output,result,['qa_id','prediction'])
    gemini.save(work/'sensor_audit.json',audits);gemini.save(work/'consistency_changes.json',changes)
    gemini.save(work/'delivery.json',{'rows':len(result),'sha256':sha256(output),'test_qa_sha256':sha256(qa),'wrapper':'reconstructed 2026-09-15; sequential API transport','exact_original_api_responses_recovered':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('data_dir',type=Path);parser.add_argument('output_csv',type=Path);args=parser.parse_args()
    infer(args.data_dir,args.output_csv.resolve())
