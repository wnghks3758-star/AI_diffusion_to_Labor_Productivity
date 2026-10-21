"""python -m dml_acr run/plan/search/summarize."""
import argparse
import json

def main():
    parser=argparse.ArgumentParser(description='Continuous DiD / finite-difference ACR')
    commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run');run.add_argument('--config',required=True);run.add_argument('--output')
    run.add_argument('--set',action='append',default=[],metavar='KEY=JSON',help='Override one setting; repeat as needed')
    plan=commands.add_parser('plan');plan.add_argument('--output',default='results/screen32')
    search=commands.add_parser('search');search.add_argument('--output',default='results/screen32')
    search.add_argument('--limit',type=int);search.add_argument('--retry-failed',action='store_true')
    summary=commands.add_parser('summarize');summary.add_argument('--output',default='results/screen32')
    args=parser.parse_args()
    if args.command=='run':
        from .pipeline import run_analysis
        from .config import AnalysisConfig
        values=AnalysisConfig.load(args.config).to_dict()
        for override in args.set:
            key,separator,value=override.partition('=')
            if not separator:parser.error('--set must be KEY=JSON')
            try:values[key]=json.loads(value)
            except json.JSONDecodeError:values[key]=value
        _,status=run_analysis(AnalysisConfig.from_dict(values),args.output);print(json.dumps(status,ensure_ascii=False))
    else:
        from .experiments import make_screen_plan,run_plan,summarize
        if args.command=='plan':print(json.dumps(make_screen_plan(args.output),ensure_ascii=False))
        elif args.command=='search':print(json.dumps(run_plan(args.output,args.limit,args.retry_failed),ensure_ascii=False))
        else:print(json.dumps(summarize(args.output)[0],ensure_ascii=False))

if __name__=='__main__':main()
