"""Gemini request and parsing code; credentials are supplied only via environment."""
import datetime, json, os, time
import urllib.error, urllib.parse, urllib.request
from pathlib import Path
from .core import ROOT, sha256, validate_answer
BASE = "https://generativelanguage.googleapis.com"
SETTINGS = ROOT / "configs/gemini_ir.json"

PROMPT = '''Answer every supplied multiple-choice question using the attached infrared video.
The same video is shared by all questions. Infrared is grayscale: do not infer RGB colors.
Observe the full timeline, body movements, hands and object interactions. Options are candidates,
not statements that an action happened. Prefer visible evidence over a stereotypical routine.
For single: choose the focal action answering that question, not necessarily the only action.
For multi: select exactly the actions observed among that question's options; one or more may
be correct. Return distinct letters in alphabetical order, without separators.
For combination: choose ONE option containing the correct combination of actions.
For sequence: return ALL available option letters in the order the actions occur in the video.
Do not sort sequence letters alphabetically or substitute the usual real-world action order.
For emotion: infer the manner/context expressed by movement from the available descriptions;
do not rely on facial visibility or invent personal traits.
For object_interaction: choose ONE option supported by the visible action and object use.
Letter mappings are specific to each question; translate action meaning back to each mapping.
Return a short list of timestamped observed events and one prediction for every qa_id.
No explanation outside JSON. When uncertain, give your best legal choice. Do not omit questions.
'''


def key():
    token = os.environ.get("GEMINI_API_KEY")
    if not token:
        raise RuntimeError("GEMINI_API_KEY must be supplied separately through the environment")
    return token


def directory():
    return ROOT / "run"


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def http(path, method='GET', body=None, headers=None, raw=False):
    token = key()
    url = path if path.startswith('https://') else BASE + path
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname != 'generativelanguage.googleapis.com':
        raise ValueError('Unexpected API host')
    hdr = {'x-goog-api-key': token}
    if headers:
        hdr.update(headers)
    if isinstance(body, dict):
        body = json.dumps(body).encode()
        hdr['Content-Type'] = 'application/json'
    request = urllib.request.Request(url, method=method, data=body, headers=hdr)
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            payload = response.read()
            data = payload if raw else (json.loads(payload) if payload else {})
            return data, dict(response.headers)
    except urllib.error.HTTPError as error:
        message = error.read().decode(errors='replace').replace(token, '[REDACTED]')
        raise RuntimeError(f'HTTP {error.code}: {message[:1500]}') from None
    except Exception as error:
        raise RuntimeError(f'{type(error).__name__}: {str(error).replace(token, "[REDACTED]")}') from None


def settings():
    return json.loads(SETTINGS.read_text())


def upload(record):
    out = directory()
    cache = out/'uploads'/f"{record['id']}.json"
    local = ROOT/record['path']
    if sha256(local)!=record['sha256']:
        raise ValueError('Video input hash changed')
    if cache.exists():
        info = json.loads(cache.read_text())
        expiry = datetime.datetime.fromisoformat(info['expirationTime'].replace('Z','+00:00'))
        if expiry <= datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=26):
            raise ValueError('Uploaded file too near expiry; refresh explicitly before a batch')
    else:
        _, headers = http('/upload/v1beta/files', 'POST', {'file':{'display_name':f"cuhkx-ir-{record['id']}"}},
                          {'X-Goog-Upload-Protocol':'resumable','X-Goog-Upload-Command':'start',
                           'X-Goog-Upload-Header-Content-Length':str(record['bytes']),
                           'X-Goog-Upload-Header-Content-Type':'video/mp4'})
        url = next(v for k,v in headers.items() if k.lower()=='x-goog-upload-url')
        response,_ = http(url,'POST',local.read_bytes(),{'Content-Type':'video/mp4','X-Goog-Upload-Offset':'0',
                                                       'X-Goog-Upload-Command':'upload, finalize'})
        info = response['file']
        save(cache,info)
    for _ in range(60):
        if info.get('state')=='ACTIVE':
            return info
        if info.get('state')=='FAILED':
            raise ValueError('Video processing failed')
        time.sleep(2)
        info,_ = http('/v1beta/'+info['name'])
        save(cache,info)
    raise TimeoutError('Video is still processing; resume later')


def request_for(record, info):
    cfg = settings()
    return {'contents':[{'role':'user','parts':[
        {'fileData':{'mimeType':'video/mp4','fileUri':info['uri']},'videoMetadata':{'fps':cfg['fps']}},
        {'text':PROMPT+'\nQuestions:\n'+json.dumps(record['questions'],ensure_ascii=False)}]}],
        'generationConfig':{'thinkingConfig':{'thinkingLevel':cfg['thinking_level']},
                            'maxOutputTokens':cfg['max_output_tokens'],'responseMimeType':'application/json',
                            'responseSchema':{'type':'OBJECT','properties':{
                                'events':{'type':'ARRAY','items':{'type':'STRING'}},
                                'answers':{'type':'ARRAY','items':{'type':'OBJECT','properties':{
                                    'qa_id':{'type':'STRING'},'prediction':{'type':'STRING'}},
                                    'required':['qa_id','prediction']}}},'required':['events','answers']}}}


def parsed(record,response):
    candidates = response.get('candidates',[])
    if len(candidates)!=1 or candidates[0].get('finishReason')!='STOP':
        raise ValueError('Missing or incomplete candidate: '+str([c.get('finishReason') for c in candidates]))
    text = ''.join(p.get('text','') for p in candidates[0]['content']['parts'] if not p.get('thought'))
    value = json.loads(text)
    answers = value['answers']
    ids = [r['qa_id'] for r in answers]
    expected = {q['qa_id']:q for q in record['questions']}
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('Missing, repeated or unexpected qa_id')
    for r in answers:
        validate_answer(expected[r['qa_id']],r['prediction'])
        if expected[r['qa_id']]['category']=='multi':
            r['prediction']=''.join(sorted(r['prediction']))
    return value
