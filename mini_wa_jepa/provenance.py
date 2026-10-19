"""Run provenance shared by data, training and evaluation."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import subprocess


def provenance(config=None):
    root=Path(__file__).resolve().parent.parent
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for directory in ('configs','controller','dataset','env','models','training','evaluation','scripts','mini_wa_jepa')
           for p in (root/directory).rglob('*.py') if '__pycache__' not in p.parts}
    return dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()),
        goal_sha256=hashlib.sha256((root/'GOAL.md').read_bytes()).hexdigest(),
        source_tree_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
        source_files=files,config=asdict(config) if is_dataclass(config) else config)


def write_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)
