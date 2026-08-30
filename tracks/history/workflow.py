"""History workflow: structured proposals -> validation -> model candidates -> top-1."""
import argparse,json,os,sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'src'))
from data import load,encode
from evaluate import evaluate
from baseline import FM
from train import history,train_candidate
from llm_feature_proposal import parse_llm_response,proposal_prompt

def feature_index(spec):
    # Computation adapter: structured specs must name a supported aggregation.
    agg=str(spec['aggregation']).lower(); formula=str(spec['formula']).lower()
    if 'positive_rate' in agg or 'long_view rate' in formula: return 3
    if 'interaction_count' in agg or 'count' in agg: return 1
    if 'ratio' in agg: return 2
    if 'recent' in agg or 'latest' in formula: return 4
    if 'duration' in agg or 'bucket' in formula: return 5
    if 'exists' in agg or 'binary' in agg: return 0
    raise ValueError('unsupported feature computation: '+spec['aggregation'])

def main(a):
    with open(a.proposals) as f: raw=f.read()
    proposals=parse_llm_response(raw)
    s=load(a.data_dir); e,dim=encode(s); X,y,_=e['train']; Xv,yv,uv=e['valid']; H=history(s['train'],s['train'],True); Hv=history(s['train'],s['valid']); results=[]
    for p in proposals:
        i=feature_index(p['spec']); m,v,ep=train_candidate(X,y,Xv,yv,uv,H[:,i:i+1],Hv[:,i:i+1],dim,a); results.append({'feature_id':p['feature_id'],'display_name':p['display_name'],'spec':p['spec'],'valid_GAUC':float(v['GAUC']),'valid_nDCG@5':float(v['nDCG@5']),'valid_primary':float(v['primary']),'best_epoch':int(ep),'learned_linear_weight':float(m.Wh[0])})
    results.sort(key=lambda x:x['valid_primary'],reverse=True); top=results[0]; out={'workflow':['structured LLM proposal','schema/leakage validation','canonical identifier','feature computation adapter','independent candidate training','validation-primary top-1 selection'],'seed':0,'test_used':False,'proposals_received':len(proposals),'candidates':results,'selected_top1':top,'baseline_valid_primary':.6015,'gain_vs_baseline':float(top['valid_primary']-.6015)}
    d=os.path.join(ROOT,'results_by_track/history/metrics'); os.makedirs(d,exist_ok=True)
    # Preserve the exact one-time LLM response and the prompt contract used to
    # obtain it. These are audit artifacts, not a claim of a live API call.
    with open(os.path.join(d,'raw_llm_response.json'),'w') as f: f.write(raw)
    with open(os.path.join(d,'llm_prompt.txt'),'w') as f: f.write(proposal_prompt())
    with open(os.path.join(d,'workflow_seed0.json'),'w') as f: json.dump(out,f,indent=2)
    with open(os.path.join(ROOT,'results_by_track/history/manifest.json'),'w') as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data_dir',required=True);p.add_argument('--proposals',required=True);p.add_argument('--k',type=int,default=16);p.add_argument('--lr',type=float,default=.001);p.add_argument('--l2',type=float,default=1e-6);p.add_argument('--batch_size',type=int,default=8192);p.add_argument('--epochs',type=int,default=15);main(p.parse_args())
