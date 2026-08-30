"""History workflow: structured proposals -> validation -> model candidates -> top-1."""
import argparse,json,os,sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'src'))
from data import load,encode
from evaluate import evaluate
from baseline import FM
from train import train_candidate
from llm_feature_proposal import parse_llm_response,proposal_prompt

MIN_VALID_COVERAGE=0.05

def compute_feature(train, rows, spec, prefix=False):
    """Compute one validated proposal without using a shared six-column proxy.

    The loader exposes date (not raw time_ms), so date plus source-row order is
    the available chronological order. Validation uses the complete training
    history; training uses only the prefix before each row.
    """
    import re
    q=np.quantile([x[5] for x in train],np.linspace(0,1,11)[1:-1])
    global_rate=sum(x[6] for x in train)/max(len(train),1); out=np.zeros((len(rows),1),np.float32)
    agg=str(spec['aggregation']).lower(); formula=str(spec['formula']).lower(); cols=set(spec['source_columns'])
    if 'interaction_count' in agg or 'count(' in formula or 'number of prior' in formula: kind='count'
    elif 'duration' in agg or 'duration' in formula or 'bucket' in formula: kind='duration'
    elif 'video_id' in cols: kind='video'
    elif 'tab' in cols: kind='tab'
    elif 'recency' in agg or 'recent' in str(spec['window']).lower(): kind='recent_author'
    elif 'author_id' in cols and 'smoothed_positive_rate' in agg: kind='author'
    else: raise ValueError('validated proposal has no implemented computation path: '+str(spec))
    counts={}; positives={}; recent={}; total=0
    def key(x):
        u,video,author,tab=x[1],x[2],x[3],x[4]
        if kind=='video': return (u,video)
        if kind=='tab': return (u,tab)
        if kind=='duration': return (u,int(np.searchsorted(q,x[5])))
        return (u,author)
    def value(x):
        nonlocal total
        if kind=='count': return np.log1p(counts.get((x[1],x[3]),0))
        if kind=='recent_author':
            vals=recent.get((x[1],x[3]),[])[-20:][::-1]
            if not vals: return global_rate
            weights=np.asarray([1.0/(i+1) for i in range(len(vals))]); return float(np.dot(vals,weights)/weights.sum())
        k=key(x); n=counts.get(k,0); p=positives.get(k,0)
        return (p+20*global_rate)/(n+20)
    history_rows=train if not prefix else []
    for i,x in enumerate(rows):
        if not prefix:
            # State is built from the entire official training split before
            # validation/test rows are scored.
            pass
        out[i,0]=value(x)
        if prefix or i==len(rows)-1:
            y=float(x[6]); k=key(x); counts[k]=counts.get(k,0)+1; positives[k]=positives.get(k,0)+y
            recent.setdefault((x[1],x[3]),[]).append(y); total+=1
    if not prefix:
        # Recompute validation values from all train history, then preserve
        # the row-wise values above only when rows are the training prefix.
        counts={}; positives={}; recent={}
        for x in train:
            y=float(x[6]); k=key(x); counts[k]=counts.get(k,0)+1; positives[k]=positives.get(k,0)+y; recent.setdefault((x[1],x[3]),[]).append(y)
        for i,x in enumerate(rows): out[i,0]=value(x)
    return out

def history_coverage(train, rows, spec, prefix=False):
    """Return the fraction of rows with real prior matching history."""
    agg=str(spec['aggregation']).lower(); cols=set(spec['source_columns'])
    if 'interaction_count' in agg: kind='author'
    elif 'duration' in agg or 'bucket' in agg or 'duration' in str(spec['formula']).lower(): kind='duration'
    elif 'video_id' in cols: kind='video'
    elif 'tab' in cols: kind='tab'
    else: kind='author'
    q=np.quantile([x[5] for x in train],np.linspace(0,1,11)[1:-1])
    def key(x):
        if kind=='video': return (x[1],x[2])
        if kind=='tab': return (x[1],x[4])
        if kind=='duration': return (x[1],int(np.searchsorted(q,x[5])))
        return (x[1],x[3])
    if not prefix:
        seen={key(x) for x in train}
        return sum(key(x) in seen for x in rows)/max(len(rows),1)
    seen=set(); covered=0
    for x in rows:
        covered += key(x) in seen
        seen.add(key(x))
    return covered/max(len(rows),1)

def main(a):
    with open(a.proposals) as f: raw=f.read()
    proposals=parse_llm_response(raw)
    s=load(a.data_dir); e,dim=encode(s); X,y,_=e['train']; Xv,yv,uv=e['valid']; results=[]; rejected=[]; models={}
    for p in proposals:
        valid_coverage=history_coverage(s['train'],s['valid'],p['spec'])
        if valid_coverage < MIN_VALID_COVERAGE:
            rejected.append({'feature_id':p['feature_id'],'display_name':p['display_name'],'spec':p['spec'],'valid_history_coverage':float(valid_coverage),'reason':'below minimum validation coverage','minimum_validation_coverage':MIN_VALID_COVERAGE})
            continue
        H=compute_feature(s['train'],s['train'],p['spec'],True); Hv=compute_feature(s['train'],s['valid'],p['spec'],False); m,v,ep=train_candidate(X,y,Xv,yv,uv,H,Hv,dim,a); models[p['feature_id']]=m; results.append({'feature_id':p['feature_id'],'display_name':p['display_name'],'spec':p['spec'],'valid_GAUC':float(v['GAUC']),'valid_nDCG@5':float(v['nDCG@5']),'valid_primary':float(v['primary']),'best_epoch':int(ep),'learned_linear_weight':float(m.Wh[0]),'train_history_coverage':float(history_coverage(s['train'],s['train'],p['spec'],True)),'valid_history_coverage':float(history_coverage(s['train'],s['valid'],p['spec']))})
    if not results: raise ValueError('no proposals meet minimum validation coverage')
    results.sort(key=lambda x:x['valid_primary'],reverse=True); top=results[0]; out={'workflow':['structured LLM proposal','schema/leakage validation','canonical identifier','coverage eligibility filter','feature computation adapter','independent candidate training','coverage audit','validation-primary top-1 selection'],'seed':0,'test_used':False,'proposals_received':len(proposals),'candidates':results,'rejected_candidates':rejected,'minimum_validation_coverage':MIN_VALID_COVERAGE,'selected_top1':top,'baseline_valid_primary':.6015,'gain_vs_baseline':float(top['valid_primary']-.6015),'selection_note':'Candidates below 5% validation coverage are rejected before training; top-1 is then selected by validation primary.'}
    d=os.path.join(ROOT,'results_by_track/history/metrics'); os.makedirs(d,exist_ok=True)
    checkpoint='results_by_track/history/checkpoints/history_top1_seed0.npz'; os.makedirs(os.path.dirname(os.path.join(ROOT,checkpoint)),exist_ok=True)
    selected_model=models[top['feature_id']]
    np.savez(os.path.join(ROOT,checkpoint),V=selected_model.V,W=selected_model.W,Vh=selected_model.Vh,Wh=selected_model.Wh,b=selected_model.b,selected_feature=top['display_name'],feature_id=top['feature_id'],seed=0,best_epoch=top['best_epoch'])
    out['checkpoint_path']=checkpoint
    # Preserve the exact one-time LLM response and the prompt contract used to
    # obtain it. These are audit artifacts, not a claim of a live API call.
    with open(os.path.join(d,'raw_llm_response.json'),'w') as f: f.write(raw)
    with open(os.path.join(d,'llm_prompt.txt'),'w') as f: f.write(proposal_prompt())
    with open(os.path.join(d,'workflow_seed0.json'),'w') as f: json.dump(out,f,indent=2)
    with open(os.path.join(ROOT,'results_by_track/history/manifest.json'),'w') as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data_dir',required=True);p.add_argument('--proposals',required=True);p.add_argument('--k',type=int,default=16);p.add_argument('--lr',type=float,default=.001);p.add_argument('--l2',type=float,default=1e-6);p.add_argument('--batch_size',type=int,default=8192);p.add_argument('--epochs',type=int,default=15);main(p.parse_args())
