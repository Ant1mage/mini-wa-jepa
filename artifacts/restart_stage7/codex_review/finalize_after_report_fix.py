"""Finish only derived reports/exports after the preserved timing-assertion failure."""
import json,subprocess,sys,traceback
from pathlib import Path
from mini_wa_jepa.provenance import provenance,write_json
from training.flow_matching import file_sha256
r=Path('artifacts/restart_stage7');v=r/'codex_review';state=json.loads((r/'delivery_status.json').read_text())
state['previous_error']=state.pop('error',None)
def run(name,command):
 state['current']=name;write_json(r/'delivery_status.json',state)
 log=v/(name+'.log')
 with log.open('w') as stream:p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{name} failed: see {log}')
 return dict(command=command,returncode=p.returncode,log=str(log))
try:
 tests=json.loads((v/'tests_result.json').read_text())
 assert tests['passed'] and tests['source_tree_sha256']==provenance()['source_tree_sha256']
 for name in ['browser_checks','rrd_checks','marimo_checks']:
  assert json.loads((v/(name+'.json')).read_text())['passed']
 for item in json.loads((v/'rrd_checks.json').read_text())['checks']:
  assert file_sha256(item['path'])==item['sha256']
 run('report_corrected',[sys.executable,'-m','scripts.report_stage7'])
 exports=[]
 for name,source in [('chapter7','chapters/chapter7_candidate_planning/app.py'),('dashboard','scripts/dashboard_marimo.py')]:
  output=r/(name+'.html');assert not output.exists()
  proof=run('export_'+name,['.venv/bin/marimo','export','html',source,'-o',str(output)])
  exports.append(dict(path=str(output),bytes=output.stat().st_size,sha256=file_sha256(output),**proof))
 write_json(v/'html_exports.json',dict(passed=True,exports=exports))
 run('report_final',[sys.executable,'-m','scripts.report_stage7'])
 state.update(completed=True,current='delivery_complete');write_json(r/'delivery_status.json',state)
except BaseException:
 state['error']=traceback.format_exc();write_json(r/'delivery_status.json',state);raise
