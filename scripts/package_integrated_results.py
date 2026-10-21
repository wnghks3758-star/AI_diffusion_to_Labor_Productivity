"""Package gate-passing derived results and source; exclude raw data and caches."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dml_acr.adaptive_experiments import OLD_ROOTS


def package(destination='work/adaptive_deliverables'):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    current=Path('results/adaptive200');summary=json.loads((current/'summary.json').read_text())
    assert summary['stop_reason'] in ['target_found','budget_exhausted']
    for name in ['adaptive200_audit.json','adaptive200_history_audit.json','integrated_audit.json']:
        assert json.loads((Path('validation')/name).read_text())['passed']
    result=destination/'Integrated_Results_All_Searches.zip'
    with zipfile.ZipFile(result,'w',zipfile.ZIP_DEFLATED) as archive:
        for folder in [Path('results/integrated'),current]:
            for path in sorted(folder.rglob('*')):
                if not path.is_file() or 'checkpoints' in path.parts:continue
                # The integrated accepted/ tree already contains every local case.
                if folder==current and 'accepted' in path.parts:continue
                if path.suffix not in ['.json','.jsonl','.csv','.md','.png']:continue
                archive.write(path,path)
        for prior in OLD_ROOTS:
            for name in ['plan.json','summary.json','all_attempts.jsonl','all_specifications.csv']:
                path=Path(prior)/name
                if path.exists():archive.write(path,'history/'+Path(prior).name+'/'+name)
        for path in sorted(Path('docs').glob('*.md')):archive.write(path,path)
        for name in ['adaptive200_audit.json','adaptive200_history_audit.json','integrated_audit.json']:
            path=Path('validation')/name;archive.write(path,path)
        for path in Path('work/integrated_reporting/output').glob('*'):
            if path.suffix in ['.pdf','.docx'] or path.name=='report_validation.json':archive.write(path,'reports/'+path.name)
    source=destination/'Updated_Python_Source.zip'
    with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as archive:
        for directory in ['dml_acr','scripts','tests','docs','configs']:
            for path in sorted(Path(directory).rglob('*')):
                if path.is_file() and path.suffix in ['.py','.md','.json']:archive.write(path,path)
        for name in ['README.md','requirements.in','requirements.lock.txt','requirements-report.txt','setup.sh','start_jupyter.sh']:
            archive.write(name,name)
    manifest={}
    for path in [result,source]:
        with zipfile.ZipFile(path) as archive:
            assert not any(name.endswith(('.xlsx','.ipynb','.joblib','.pkl','.npy','.mmap')) for name in archive.namelist())
        manifest[path.name]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (destination/'packages_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(current/'checkpoints/latest_random_results.zip',destination/'progress_results.zip')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='work/adaptive_deliverables');args=parser.parse_args()
    print(json.dumps(package(args.output),ensure_ascii=False))
