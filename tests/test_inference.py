import importlib.util,json,os,sys,tempfile,unittest,zipfile,shutil
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cuhkx.core import write_csv,read_csv,validate_answer,validate_submission
spec=importlib.util.spec_from_file_location('portable_infer',ROOT/'code/infer.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class VerificationTests(unittest.TestCase):
 def test_unseen_input_and_missing_sensor_fallback(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);video=root/'new_clip/IR/IR.mp4';video.parent.mkdir(parents=True);video.write_bytes(b'synthetic fixture not a real video')
   common={'path':'new_clip/IR/IR.mp4','source':'HAU','question':'synthetic fixture'}
   qs=[dict(common,qa_id='new-combo',category='combination',A='walk, sit',B='stand',C='',D=''),dict(common,qa_id='new-multi',category='multi',A='walk',B='sit',C='stand',D=''),dict(common,qa_id='new-sequence',category='sequence',A='walk',B='sit',C='stand',D=''),dict(common,qa_id='new-emotion',category='emotion',A='calm',B='angry',C='',D='')]
   write_csv(root/'test_qa.csv',qs,list(qs[0]))
   answers=[{'qa_id':q['qa_id'],'prediction':v} for q,v in zip(qs,['A','A','BAC','B'])]
   result={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':json.dumps({'events':[],'answers':answers})}]}}]}
   # Synthetic placeholders exercise fallback without trained weights.
   assets=root/'package'
   for name in ['configs','cuhkx','artifacts']:(assets/name).mkdir(parents=True)
   shutil.copyfile(ROOT/'configs/gemini_ir.json',assets/'configs/gemini_ir.json')
   shutil.copyfile(ROOT/'cuhkx/gemini.py',assets/'cuhkx/gemini.py')
   runner.joblib.dump(None,assets/'artifacts/final_model.joblib')
   for name,value in [('prior.json',[{},{}]),('sensor_config.json',{}),('feature_columns.json',[])]:
    (assets/'artifacts'/name).write_text(json.dumps(value))
   with patch.object(runner,'PACKAGE',assets),patch.dict(os.environ,{'GEMINI_API_KEY':'synthetic-not-a-real-key','CUHKX_RUN_DIR':str(root/'cache')},clear=False),patch.object(runner.gemini,'upload',return_value={'uri':'https://example.invalid/synthetic'}),patch.object(runner.gemini,'http',return_value=(result,{})) as http:
    runner.infer(root,root/'result.csv');self.assertEqual(http.call_count,1)
    runner.infer(root,root/'result2.csv');self.assertEqual(http.call_count,1)
   output=read_csv(root/'result.csv');self.assertEqual([r['prediction'] for r in output],['A','AB','BAC','B'])
   self.assertEqual((root/'result.csv').read_bytes(),(root/'result2.csv').read_bytes())
 def test_sequence_contract(self):
  q={'qa_id':'x','category':'sequence','A':'a','B':'b','C':'c','D':''}
  validate_answer(q,'BAC')
  with self.assertRaises(ValueError):validate_answer(q,'AB')
  with self.assertRaises(ValueError):validate_answer(q,'ABCD')
 def test_reject_labelled_input(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);write_csv(root/'test_qa.csv',[{'qa_id':'x','answer':'A'}],['qa_id','answer'])
   with self.assertRaisesRegex(ValueError,'label-free'):runner.infer(root,root/'out.csv')
 def test_video_archive_refreshes_changed_input(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);archive=root/'large_model_track_test.zip';work=root/'cache'
   for content in [b'first',b'new data same path']:
    with zipfile.ZipFile(archive,'w') as z:z.writestr('new_clip/IR/IR.mp4',content)
    self.assertEqual(runner.video_path(root,'new_clip',work).read_bytes(),content)
 def test_output_order_contract(self):
  questions=[{'qa_id':'x','category':'single','A':'a','B':'b','C':'','D':''},{'qa_id':'y','category':'single','A':'a','B':'b','C':'','D':''}]
  sample=[{'qa_id':'x','prediction':''},{'qa_id':'y','prediction':''}]
  with self.assertRaises(ValueError):validate_submission([{'qa_id':'y','prediction':'A'},{'qa_id':'x','prediction':'A'}],questions,sample)

 def test_missing_weights_blocks_before_api(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   q={'qa_id':'new','path':'new_clip/IR/IR.mp4','category':'single','A':'walk','B':'sit','C':'','D':''}
   write_csv(root/'test_qa.csv',[q],list(q))
   with patch.object(runner,'PACKAGE',root/'no_model'),patch.object(runner.gemini,'http') as http:
    with self.assertRaisesRegex(RuntimeError,'Local model artifacts are missing'):
     runner.infer(root,root/'result.csv')
    http.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
