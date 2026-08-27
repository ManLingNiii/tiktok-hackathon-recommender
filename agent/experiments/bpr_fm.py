"""Official-data BPR FM experiment; validation-only development entrypoint."""
import json, os, sys, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
KIT = os.path.join(ROOT, "kuairand-starter-kit")
sys.path.insert(0, KIT)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from baseline import FM, sigmoid
from evaluate import evaluate
from validation_only import load_train_valid
from modules.loss_adapter import bpr_loss

DATA = os.path.join(KIT, "KuaiRand-Pure", "data")

class BPRFM(FM):
    def step_bpr(self, Xp, Xn):
        zp, Ep, Sp = self.logits(Xp); zn, En, Sn = self.logits(Xn)
        b = len(zp)
        g = (sigmoid(zn - zp) / max(b, 1)).astype(np.float32)
        gVp = np.zeros_like(self.V); gVn = np.zeros_like(self.V)
        gWp = np.zeros_like(self.W); gWn = np.zeros_like(self.W)
        np.add.at(gWp, Xp, -g[:, None]); np.add.at(gWn, Xn, g[:, None])
        np.add.at(gVp, Xp, -g[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gVn, Xn,  g[:, None, None] * (Sn[:, None, :] - En))
        gV, gW = gVp + gVn + self.l2 * self.V, gWp + gWn + self.l2 * self.W
        self.t += 1; b1, b2, eps = .9, .999, 1e-8
        for P, G, M, VV in ((self.V,gV,self.mV,self.vV),(self.W,gW,self.mW,self.vW)):
            M *= b1; M += (1-b1)*G; VV *= b2; VV += (1-b2)*(G*G)
            P -= self.lr*(M/(1-b1**self.t))/(np.sqrt(VV/(1-b2**self.t))+eps)
        self.b -= self.lr * (g.sum() - g.sum())
        return bpr_loss(zp, zn)

def run(epochs=12, bs=8192, patience=3, seed=0):
    splits = load_train_valid(DATA)
    sys.path.insert(0, KIT)
    from data import encode
    enc, dim = encode({**splits, "test": []})
    Xtr, ytr, _ = enc["train"]; Xva, yva, users = enc["valid"]
    model = BPRFM(dim, k=16, lr=.001, seed=seed); rng=np.random.default_rng(seed)
    best=-1.; best_state=None; bad=0
    for ep in range(1, epochs+1):
        pos=np.flatnonzero(ytr>0); neg=np.flatnonzero(ytr<=0)
        n=min(len(pos),len(neg)); rng.shuffle(pos); rng.shuffle(neg); losses=[]
        for i in range(0,n,bs):
            end = min(i + bs, n)
            losses.append(model.step_bpr(Xtr[pos[i:end]], Xtr[neg[i:end]]))
        va=evaluate(users,yva,model.predict(Xva)); print(f"epoch {ep:02d} | bpr_loss {np.mean(losses):.5f} | validation {va}",flush=True)
        if va['primary'] > best+1e-5: best=va['primary']; best_state=(model.V.copy(),model.W.copy(),model.b); bad=0
        else: bad+=1
        if bad>=patience: break
    model.V,model.W,model.b=best_state; va=evaluate(users,yva,model.predict(Xva))
    result={"experiment":"bpr_fm","status":"success","split":"validation_only","test_access":False,
            "metrics":{k:float(v) for k,v in va.items()},"baseline_primary":0.6015,
            "delta_primary":float(va['primary']-.6015),"epochs":ep}
    os.makedirs(os.path.join(ROOT,"runs"),exist_ok=True)
    with open(os.path.join(ROOT,"runs","bpr_fm_result.json"),"w",encoding="utf-8") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2)); return result

if __name__ == "__main__": run()
