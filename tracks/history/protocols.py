"""Leakage-safer seed-0 feature-selection protocols."""
import argparse,json,os,sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0,os.path.join(ROOT,'src'))
from data import load,encode
from evaluate import evaluate
from baseline import FM
from train import Top1HistoryFM,history_matrix,HISTORY_FIELDS

def fit_base(train_rows,hold_rows,dim,epochs=7):
    # Reuse shared categorical encoding supplied by official loader; caller provides arrays.
    pass

def select_internal(s,e,a):
    X,y,_=e['train']; n=int(len(y)*.8); Xt,yt=X[:n],y[:n]; Xh,yh,uh=e['train'][0][n:],e['train'][1][n:],e['train'][2][n:]
    m=FM(a.dim,k=16,lr=.001,l2=1e-6,seed=0); rng=np.random.default_rng(0)
    for _ in range(a.select_epochs):
        idx=rng.permutation(n)
        for j in range(0,n,8192): m.step(Xt[idx[j:j+8192]],yt[idx[j:j+8192]])
    H=history_matrix(s['train'][:n],s['train'][n:],False); base=m.predict(Xh); b=evaluate(uh,yh,base)['primary']; scores=[]
    for i,f in enumerate(HISTORY_FIELDS):
        v=evaluate(uh,yh,base+a.selection_weight*H[:,i]); scores.append({'feature':f,'primary':float(v['primary']),'gain':float(v['primary']-b)})
    return max(scores,key=lambda x:x['primary'])['feature'],scores

def select_cv(s,e,a):
    rows=s['train']; X,y,u=e['train']; u=np.asarray(u); n=len(rows); fold=np.arange(n)%3; scores={f:[] for f in HISTORY_FIELDS}
    for k in range(3):
        tr=fold!=k; va=~tr; m=FM(a.dim,k=16,lr=.001,l2=1e-6,seed=0); rng=np.random.default_rng(0); ids=np.where(tr)[0]
        for _ in range(a.select_epochs):
            order=ids[rng.permutation(len(ids))]
            for j in range(0,len(order),8192): m.step(X[order[j:j+8192]],y[order[j:j+8192]])
        train_rows=[rows[i] for i in ids]; val_rows=[rows[i] for i in np.where(va)[0]]; H=history_matrix(train_rows,val_rows,False); base=m.predict(X[va]); b=evaluate(u[va],y[va],base)['primary']
        for i,f in enumerate(HISTORY_FIELDS): scores[f].append(float(evaluate(u[va],y[va],base+a.selection_weight*H[:,i])['primary']-b))
    summary=[{'feature':f,'mean_gain':float(np.mean(v)),'std_gain':float(np.std(v))} for f,v in scores.items()]; return max(summary,key=lambda x:x['mean_gain'])['feature'],summary

def final_train(s,e,a,selected):
    X,y,_=e['train']; Xv,yv,uv=e['valid']; i=HISTORY_FIELDS.index(selected); H=history_matrix(s['train'],s['train'],True)[:,i:i+1]; Hv=history_matrix(s['train'],s['valid'],False)[:,i:i+1]; m=Top1HistoryFM(a.dim,16,a.lr,a.l2,0); rng=np.random.default_rng(0); best=(-1,None,0)
    for ep in range(1,a.epochs+1):
        ids=rng.permutation(len(y))
        for j in range(0,len(ids),a.batch_size): m.step(X[ids[j:j+a.batch_size]],H[ids[j:j+a.batch_size]],y[ids[j:j+a.batch_size]])
        v=evaluate(uv,yv,m.predict(Xv,Hv))
        if v['primary']>best[0]: best=(v['primary'],(m.V.copy(),m.W.copy(),m.Vh.copy(),m.Wh.copy(),m.b),ep)
    m.V,m.W,m.Vh,m.Wh,m.b=best[1]; return evaluate(uv,yv,m.predict(Xv,Hv)),best[2]

def main(a):
    s=load(a.data_dir); e,dim=encode(s); a.dim=dim; out=[]
    for protocol in ('internal_holdout','train_cv'):
        selected,selection=select_internal(s,e,a) if protocol=='internal_holdout' else select_cv(s,e,a); v,ep=final_train(s,e,a,selected); out.append({'protocol':protocol,'selected_top1':selected,'selection':selection,'valid_GAUC':float(v['GAUC']),'valid_nDCG@5':float(v['nDCG@5']),'valid_primary':float(v['primary']),'baseline_valid_primary':.6015,'gain_vs_baseline':float(v['primary']-.6015),'best_epoch':ep,'seed':0,'test_used':False})
    d=os.path.join(ROOT,'results_by_track/history/metrics'); os.makedirs(d,exist_ok=True)
    with open(os.path.join(d,'safer_protocols_seed0.json'),'w') as f: json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data_dir',required=True);p.add_argument('--selection_weight',type=float,default=.25);p.add_argument('--select_epochs',type=int,default=5);p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch_size',type=int,default=8192);p.add_argument('--lr',type=float,default=.001);p.add_argument('--l2',type=float,default=1e-6);main(p.parse_args())
