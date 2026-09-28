"""Frozen IMU preprocessing, classifier and option prior."""
import collections, hashlib, io
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
DEVICES = ("WTC", "WTLA", "WTRA", "WTLL", "WTRL")
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz", "acc_norm", "gyro_norm")

def norm(value):
    return ' '.join(value.lower().split())


def signal_features(x, hz):
    x = np.asarray(x, dtype=float)
    centered = x - x.mean()
    sd = x.std()
    q = np.quantile(x, [.1, .25, .5, .75, .9])
    f = {'mean': x.mean(), 'std': sd, 'p10': q[0], 'median': q[2],
         'p90': q[4], 'iqr': q[3]-q[1], 'range': np.ptp(x),
         'diff_rms': np.sqrt(np.mean(np.diff(x)**2)) if len(x)>1 else 0.0}
    power = np.abs(np.fft.rfft(centered * np.hanning(len(x))))**2
    freq = np.fft.rfftfreq(len(x), 1/hz)
    power[0] = 0
    total = power.sum()
    probs = power / max(total, 1e-12)
    f['log_power'] = np.log1p(total / len(x))
    f['centroid'] = np.sum(freq*probs)
    f['dominant_hz'] = freq[np.argmax(power)] if total>1e-12 else 0.0
    nonzero = probs[probs>0]
    f['spectral_entropy'] = -np.sum(nonzero*np.log(nonzero))/max(np.log(len(power)),1)
    for lo, hi in [(0, .5), (.5, 1), (1, 2), (2, 5.01)]:
        f[f'band_{lo}_{hi}'] = probs[(freq>=lo)&(freq<hi)].sum()
    for seconds in [.1, .5, 1.0]:
        lag = round(seconds*hz)
        f[f'acf_{seconds}'] = (np.mean(centered[:-lag]*centered[lag:])/(sd**2)
                                if len(x)>lag and sd>1e-10 else 0.0)
    for i, part in enumerate(np.array_split(x, 3)):
        f[f'third{i}_std'] = part.std() if len(part) else 0.0
        f[f'third{i}_mean'] = part.mean() if len(part) else 0.0
    return f


def parse_csv(data):
    frame = pd.read_csv(io.StringIO(data.decode('utf-8-sig')))
    columns = frame.columns.tolist()
    if len(columns)<8 or not ('设备' in columns[1] or 'device' in columns[1].lower()):
        raise ValueError('Unrecognized IMU device header')
    if not ('加速度' in columns[2] or 'acc' in columns[2].lower()):
        raise ValueError('Unrecognized acceleration header')
    if not ('角速度' in columns[5] or columns[5]=='AsX(°/s)' or any(s in columns[5].lower() for s in ['gyro','angular'])):
        raise ValueError('Unrecognized gyroscope header')
    result = collections.defaultdict(list)
    times = pd.to_datetime(frame.iloc[:, 0], errors='coerce', format='mixed')
    values = frame.iloc[:, 2:8].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    for t, device, v in zip(times, frame.iloc[:, 1], values):
        name = str(device).split('(')[0].strip().upper()
        if name in DEVICES and not pd.isna(t) and np.isfinite(v).all():
            result[name].append((t.value/1e9, *v))
    return result


def device_signal(rows, cfg):
    if len(rows)<cfg['minimum_device_samples']:
        return None, {'reason': 'too_few_samples', 'samples': len(rows)}
    a = np.asarray(rows, dtype=float)
    inversions = int(np.sum(np.diff(a[:, 0])<0))
    a = a[np.argsort(a[:, 0], kind='stable')]
    times, inverse = np.unique(a[:, 0], return_inverse=True)
    values = np.zeros((len(times), 6))
    np.add.at(values, inverse, a[:, 1:])
    values /= np.bincount(inverse)[:, None]
    times -= times[0]
    duration = float(times[-1])
    gap = float(np.max(np.diff(times))) if len(times)>1 else 0.0
    audit = {'samples': len(a), 'unique_times': len(times), 'duration': duration,
             'out_of_order': inversions, 'max_gap': gap}
    if len(times)<cfg['minimum_device_samples'] or duration<.7 or duration>300 or gap>cfg['max_gap_seconds']:
        return None, {**audit, 'reason': 'insufficient_or_gapped_recording'}
    grid = np.arange(0, duration+1e-6, 1/cfg['sample_hz'])
    resampled = np.column_stack([np.interp(grid, times, values[:, i]) for i in range(6)])
    x = np.column_stack([resampled, np.linalg.norm(resampled[:, :3], axis=1),
                         np.linalg.norm(resampled[:, 3:], axis=1)])
    # Hash physical readings and relative timing, not device addresses/calendar time.
    digest = hashlib.sha256(np.round(np.column_stack([times, values]), 4).astype('<f8').tobytes()).hexdigest()
    return (x, digest), audit


def feature_record(z, names, row, split, cfg):
    path = row['path'].replace('\\', '/')
    if split=='test':
        path = '/'.join(path.split('/')[:-2])
    prefix = 'LMT_(IMU,Radar,Skeleton)/' + ('Testing/' if split=='test' else 'Training/') + path + '/IMU/'
    readings = collections.defaultdict(list)
    hashes = {}
    for filename in ['up(LA+RA+C).csv', 'down(LL+RL).csv']:
        member = prefix+filename
        if member not in names:
            continue
        data = z.read(member)
        hashes[member] = hashlib.sha256(data).hexdigest()
        for device, values in parse_csv(data).items():
            readings[device].extend(values)
    f, audit, signal_hashes = {}, {}, []
    template = signal_features(np.zeros(10), cfg['sample_hz'])
    for device in DEVICES:
        signal, details = device_signal(readings[device], cfg)
        audit[device] = details
        f[device+'_present'] = float(signal is not None)
        f[device+'_duration'] = details.get('duration', np.nan)
        if signal is not None:
            x, digest = signal
            signal_hashes.append(device+':'+digest)
        for i, channel in enumerate(CHANNELS):
            features = signal_features(x[:, i], cfg['sample_hz']) if signal is not None else {k: np.nan for k in template}
            f.update({device+'_'+channel+'_'+k: float(v) for k,v in features.items()})
    usable = len(signal_hashes)>=2
    digest = hashlib.sha256('|'.join(signal_hashes).encode()).hexdigest() if usable else None
    return f, {'qa_id': row['qa_id'], 'split': split, 'usable': usable, 'files': hashes,
               'devices': audit, 'signal_sha256': digest}


def labels(rows):
    return np.array([norm(r[r['answer']]) for r in rows])


def prior_fit(rows):
    offered, chosen = collections.Counter(), collections.Counter()
    for r in rows:
        offered.update(norm(r[k]) for k in 'ABCD' if r[k])
        chosen[norm(r[r['answer']])] += 1
    return offered, chosen


def prior_scores(rows, prior):
    offered, chosen = prior
    scores = np.array([[(chosen[norm(r[k])]+2)/(offered[norm(r[k])]+8) if r[k] else 0 for k in 'ABCD'] for r in rows])
    return scores / scores.sum(axis=1, keepdims=True)


def train_model(x, rows, leaf, cfg):
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
        RandomForestClassifier(n_estimators=cfg['trees'], min_samples_leaf=leaf,
                               max_features=.5, n_jobs=cfg['workers'], random_state=cfg['seed']))
    return model.fit(x, labels(rows))


def option_scores(model, x, rows, prior):
    proba = model.predict_proba(x)
    classes = {c:i for i,c in enumerate(model[-1].classes_)}
    p = np.array([[proba[j, classes[norm(r[k])]] if norm(r[k]) in classes and r[k] else 0.0
                   for k in 'ABCD'] for j,r in enumerate(rows)])
    empty = p.sum(axis=1)==0
    p[empty] = prior_scores(rows, prior)[empty]
    return p / p.sum(axis=1, keepdims=True)


def letters(scores):
    return np.array(list('ABCD'))[scores.argmax(axis=1)]
