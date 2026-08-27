"""Shared validation-only trainer for the four reviewed headroom objectives.

The trainer owns splitting, FM updates, early stopping, and evaluation. Modules
only provide a logit-gradient or leakage-safe feature transform.
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
sys.path.insert(0, KIT); sys.path.insert(0, os.path.dirname(__file__))
from baseline import FM, sigmoid
from data import encode
from evaluate import evaluate
from modules.loss_adapter import listwise_logit_gradient, normalize_watch_time
from rich_data import load_rich, encode_rich

DATA = os.path.join(KIT, "KuaiRand-Pure", "data")

class GradientFM(FM):
    def apply_logit_gradient(self, X, grad):
        grad = np.asarray(grad, dtype=np.float32); b=max(len(grad),1)
        z,E,S=self.logits(X); g=grad/b
        gV=np.zeros_like(self.V); gW=np.zeros_like(self.W)
        np.add.at(gW,X,g[:,None]); np.add.at(gV,X,g[:,None,None]*(S[:,None,:]-E))
        gV += self.l2*self.V; gW += self.l2*self.W
        self.t += 1; b1,b2,eps=.9,.999,1e-8
        for P,G,M,V in ((self.V,gV,self.mV,self.vV),(self.W,gW,self.mW,self.vW)):
            M*=b1; M+=(1-b1)*G; V*=b2; V+=(1-b2)*(G*G)
            P-=self.lr*(M/(1-b1**self.t))/(np.sqrt(V/(1-b2**self.t))+eps)
        self.b -= self.lr*float(np.sum(g))

def history_rows(rows):
    """Return rows with train-only prior counts; current label is added last."""
    user,tab,author={}, {}, {}; out=[]
    for row in sorted(rows,key=lambda r:(int(r[0]),str(r[1]))):
        date,uid,vid,aid,t,dur,y=row[:7]; kt=(uid,t); ka=(uid,aid)
        out.append(tuple(row)+(user.get(uid,0),tab.get(kt,0),author.get(ka,0)))
        user[uid]=user.get(uid,0)+int(y); tab[kt]=tab.get(kt,0)+int(y); author[ka]=author.get(ka,0)+int(y)
    return out

def train(mode="listwise", limit=None, epochs=8, seed=0):
    train_rows,valid_rows=load_rich(DATA)
    if limit: train_rows=train_rows[:limit]; valid_rows=valid_rows[:max(1000,limit//4)]
    et,ev,dim=encode_rich(train_rows,valid_rows,include_history=True)
    Xtr,ytr,train_users,aux_tr,play_tr,dur_tr=et; Xva,yva,users,_,_,_=ev
    m=GradientFM(dim,k=16,lr=.001,seed=seed); rng=np.random.default_rng(seed)
    best=-1.; state=None; bad=0
    for ep in range(1,epochs+1):
        idx=rng.permutation(len(ytr)); losses=[]
        for start in range(0,len(idx),8192):
            ix=idx[start:start+8192]; scores=m.predict(Xtr[ix]); grad=sigmoid(scores)-ytr[ix]
            if mode=="listwise":
                # Vectorized softmax by user within this batch. Sorting once avoids
                # the previous Python nested scan over every exposure.
                gu=np.asarray([train_users[j] for j in ix],dtype=object)
                order=np.argsort(gu,kind="stable"); sorted_u=gu[order]
                starts=np.r_[0,1+np.flatnonzero(sorted_u[1:]!=sorted_u[:-1])]
                ends=np.r_[starts[1:],len(order)]
                ss=scores[order]; yy=ytr[ix][order]
                mx=np.maximum.reduceat(ss,starts)
                ee=np.exp(ss-np.repeat(mx,ends-starts))
                denom=np.add.reduceat(ee,starts)
                grad_sorted=ee/np.repeat(denom,ends-starts)
                sums=np.add.reduceat(yy,starts)
                grad_sorted-=yy/np.repeat(np.maximum(sums,1.0),ends-starts)
                grad=np.empty_like(grad_sorted); grad[order]=grad_sorted
            elif mode=="multitask":
                # Shared FM head with real auxiliary feedback labels. The weighted
                # auxiliary residual backpropagates through the shared representation.
                weights={"is_click":.10,"is_like":.10,"is_follow":.05,"is_comment":.05,"is_forward":.05}
                aux_target=sum(w*aux_tr[k][ix] for k,w in weights.items())/sum(weights.values())
                grad=0.7*grad+0.3*(sigmoid(scores)-aux_target)
            elif mode=="cwm":
                dur=dur_tr[ix]; target=normalize_watch_time(play_tr[ix],dur); pred=sigmoid(scores)
                cens=ytr[ix]<1; grad=0.8*(pred-ytr[ix])+0.2*np.where(cens,np.minimum(0,pred-target),pred-target)
            losses.append(float(np.mean(grad*grad))); m.apply_logit_gradient(Xtr[ix],grad)
        va=evaluate(users,yva,m.predict(Xva)); print(f"{mode} epoch {ep:02d} loss {np.mean(losses):.6f} validation {va}",flush=True)
        if va["primary"]>best+1e-5: best=va["primary"]; state=(m.V.copy(),m.W.copy(),m.b); bad=0
        else: bad+=1
        if bad>=3: break
    m.V,m.W,m.b=state; va=evaluate(users,yva,m.predict(Xva))
    return {"experiment":mode+"_fm","status":"success","split":"validation_only","test_access":False,
            "metrics":{k:float(v) for k,v in va.items()},"epochs":ep}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=["listwise","history","multitask","cwm"],required=True); ap.add_argument("--limit",type=int); ap.add_argument("--epochs",type=int,default=8); a=ap.parse_args()
    print(json.dumps(train(a.mode,a.limit,a.epochs),indent=2))
