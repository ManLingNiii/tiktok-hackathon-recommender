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
from modules.listwise_loss import complete_user_batches, grouped_listwise_gradient
from rich_data import load_rich, encode_rich
from modules.headroom_modules import MultiTaskModule
from modules.composition import predict_composition, save_best_composition_manifest
from modules.context_composition import (fit_context_composition, save_context_checkpoint,
                                          fit_composition_layer,
                                          save_composition_layer_checkpoint)
try:
    from checkpoint_manager import load_best_parameters, save_if_best
    from validation_guard import evaluate_confirmation
except ImportError:
    from agent.checkpoint_manager import load_best_parameters, save_if_best
    from agent.validation_guard import evaluate_confirmation

try:
    from dataset_config import data_dir, dataset_name, runs_dir, outputs_dir
except ImportError:
    from agent.dataset_config import data_dir, dataset_name, runs_dir, outputs_dir
DATA = data_dir()

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

    def step_bpr(self, Xp, Xn):
        """Apply the reviewed BPR update used by the standalone BPR family."""
        zp, Ep, Sp = self.logits(Xp); zn, En, Sn = self.logits(Xn)
        batch = max(len(zp), 1)
        g = (sigmoid(zn - zp) / batch).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, Xp, -g[:, None]); np.add.at(gW, Xn, g[:, None])
        np.add.at(gV, Xp, -g[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, g[:, None, None] * (Sn[:, None, :] - En))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1; b1, b2, eps = .9, .999, 1e-8
        for P, G, M, VV in ((self.V, gV, self.mV, self.vV),
                            (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1-b1) * G; VV *= b2; VV += (1-b2) * (G*G)
            P -= self.lr * (M/(1-b1**self.t)) / (np.sqrt(VV/(1-b2**self.t))+eps)
        # The score difference has no bias gradient.
        return float(np.mean(-np.log(np.maximum(sigmoid(zp-zn), 1e-8))))

class DualHeadFM(GradientFM):
    """Shared FM interaction factors plus an isolated auxiliary head."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auxW = np.zeros_like(self.W)
    def aux_predict(self, X):
        z, _, _ = self.logits(X)
        return z - self.W[X].sum(1) + self.auxW[X].sum(1)
    def apply_aux_gradient(self, X, grad):
        np.add.at(self.auxW, X, -self.lr * np.asarray(grad, dtype=np.float32)[:, None])

def history_rows(rows):
    """Return rows with train-only prior counts; current label is added last."""
    user,tab,author={}, {}, {}; out=[]
    for row in sorted(rows,key=lambda r:(int(r[0]),str(r[1]))):
        date,uid,vid,aid,t,dur,y=row[:7]; kt=(uid,t); ka=(uid,aid)
        out.append(tuple(row)+(user.get(uid,0),tab.get(kt,0),author.get(ka,0)))
        user[uid]=user.get(uid,0)+int(y); tab[kt]=tab.get(kt,0)+int(y); author[ka]=author.get(ka,0)+int(y)
    return out

def _listwise_batches(users, batch_size, rng):
    """Build batches that never split a user's exposure group."""
    users = np.asarray(users, dtype=object)
    order = np.argsort(users, kind="stable")
    starts = np.r_[0, 1 + np.flatnonzero(users[order][1:] != users[order][:-1])]
    ends = np.r_[starts[1:], len(order)]
    groups = [order[s:e] for s, e in zip(starts, ends)]
    # Shuffle whole groups, never individual rows, so each softmax sees the
    # same user's complete exposure list.
    current = []
    current_size = 0
    for group_index in rng.permutation(len(groups)):
        group = groups[group_index]
        if current and current_size + len(group) > batch_size:
            yield np.concatenate(current)
            current = []
            current_size = 0
        if len(group) <= batch_size:
            current.append(group)
            current_size += len(group)
        else:
            if current:
                yield np.concatenate(current)
                current = []
                current_size = 0
            for start in range(0, len(group), batch_size):
                yield group[start:start + batch_size]
    if current:
        yield np.concatenate(current)


def train(mode="listwise", limit=None, epochs=8, seed=0, batch_size=8192, patience=None):
    cfg = json.loads(os.environ.get("AGENT_MODEL_CONFIG", "{}"))
    if mode == "composition" and not cfg:
        manifest_path = os.path.join(ROOT, "submission_ready", "composition_manifest.json")
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)
        cfg = dict(manifest.get("composition", {}))
        cfg["name"] = "manifest_composition"
    k = int(cfg.get("k", 16)); lr = float(cfg.get("lr", .001)); l2 = float(cfg.get("l2", 1e-6))
    epochs = int(cfg.get("epochs", epochs)); batch_size = int(cfg.get("batch_size", batch_size))
    patience = int(cfg.get("patience", 3 if patience is None else patience))
    train_rows,valid_rows=load_rich(DATA)
    if limit: train_rows=train_rows[:limit]; valid_rows=valid_rows[:max(1000,limit//4)]
    if mode == "composition":
        # All five offline family checkpoints are frozen.  Only this final
        # composition layer is trainable for every task candidate.
        comp_data, comp_state, scores = fit_composition_layer(
            ROOT, cfg, rich_rows=(train_rows, valid_rows))
        context_users = comp_data["valid_users"]
        context_labels = comp_data["valid_labels"]
        va = evaluate(context_users, context_labels, scores)
        confirmation = evaluate_confirmation(context_users, context_labels, scores)
        checkpoint = save_composition_layer_checkpoint(ROOT, comp_state, cfg, va)
        return {"experiment":"composition_fm", "dataset":dataset_name(), "status":"success",
                "split":"validation_only", "test_access":False,
                "metrics":{k:float(v) for k,v in va.items()},
                "confirmation_metrics":{k:float(v) for k,v in confirmation.items()},
                "config":cfg, "checkpoint":checkpoint,
                "checkpoint_saved":True, "family_best_metrics":{},
                "composition_model":comp_state["model"],
                "composition_loss":comp_state["loss"],
                "feature_schema":list(comp_state["feature_names"]),
                "raw_weights":[float(x) for x in comp_state.get("raw_weights", [])],
                "normalized_weights":[float(x) for x in comp_state.get("normalized_weights", [])],
                "prediction_input_weights":[float(x) for x in comp_state.get("prediction_input_weights", [])],
                "optimizer":comp_state.get("optimizer", "adam"),
                "loss":float(comp_state.get("train_monitor_loss", 0.0)),
                "target":"long_view",
                "prediction_analysis":comp_state.get("prediction_analysis", {}),
                "feature_variance":comp_state.get("feature_variance", {}),
                "training_history": comp_state.get("training_history", []),
                "schema_analysis":{"source":"train_only",
                                    "feature_names":list(comp_state["feature_names"])},
                "composition_manifest": {"saved": False, "deferred_until": "multi_seed_confirmation"}}
    # History is a separate reviewed ablation; keep multitask comparable to
    # the official FM/BPR feature space instead of mixing two hypotheses.
    history_cap = int(cfg.get("history_cap", 20))
    history_transform = cfg.get("history_transform", "clip")
    et,ev,dim=encode_rich(train_rows,valid_rows,include_history=(mode=="history" or cfg.get("include_history", False)),
                           history_cap=history_cap, history_transform=history_transform)
    Xtr,ytr,train_users,aux_tr,play_tr,dur_tr=et; Xva,yva,users,_,_,_=ev
    if mode == "composition":
        raise RuntimeError("unreachable composition branch")
    seed = int(cfg.get("seed", seed))
    model_cls = DualHeadFM if cfg.get("architecture") == "dual_head" and mode == "multitask" else GradientFM
    m=model_cls(dim,k=k,lr=lr,l2=l2,seed=seed); rng=np.random.default_rng(seed)
    def ranking_scores(X):
        base = m.predict(X)
        mix = float(cfg.get("inference_aux_mix", 0.0))
        if mix and isinstance(m, DualHeadFM):
            aux = sigmoid(m.aux_predict(X))
            return base + mix * (aux - np.mean(aux))
        return base
    warmstart_loaded = False
    warmstart_note = None
    if os.environ.get("AGENT_USE_PRETRAINED", "0") == "1":
        pretrained = cfg.get("pretrained")
        if mode == "listwise" and not pretrained:
            default_warmstart = os.path.join(outputs_dir(), "listwise_external_warmstart.npz")
            warmstart = cfg.get("pretrained") or default_warmstart
            if not os.path.isabs(warmstart):
                warmstart = os.path.join(ROOT, warmstart)
            if os.path.exists(warmstart):
                with np.load(warmstart, allow_pickle=False) as saved:
                    if saved["V"].shape == m.V.shape and saved["W"].shape == m.W.shape:
                        m.V[...] = saved["V"]; m.W[...] = saved["W"]; m.b = np.float32(saved["b"])
                        warmstart_loaded = True
                        pretrained = False
                    else:
                        # A generated config may intentionally change k.  Do
                        # not make a valid candidate fatal just because an
                        # optional warm-start has a different shape.
                        warmstart_note = "skipped_incompatible_listwise_warmstart"
        if pretrained and mode != "listwise":
            warmstart = pretrained
            if not os.path.isabs(warmstart):
                warmstart = os.path.join(ROOT, warmstart)
            if os.path.exists(warmstart):
                pretrained = warmstart
            else:
                warmstart_note = "skipped_missing_configured_warmstart"
                pretrained = False
        if pretrained is not False and not pretrained:
            pretrained = load_best_parameters(mode + "_fm")
        if pretrained and pretrained["V"].shape == m.V.shape and pretrained["W"].shape == m.W.shape:
            m.V[...] = pretrained["V"]; m.W[...] = pretrained["W"]; m.b = np.float32(pretrained["b"])
    multitask = MultiTaskModule(auxiliary_weights={"is_click": 1.0}, primary_weight=0.95)
    best=-1.; state=None; bad=0
    if warmstart_loaded:
        # A warm-start is a valid candidate in its own right. Fine-tuning may
        # not replace it with a lower validation score.
        initial_metrics = evaluate(users, yva, ranking_scores(Xva))
        best = float(initial_metrics["primary"])
        state = (m.V.copy(), m.W.copy(), m.b)
    for ep in range(1, epochs+1):
        losses=[]
        if mode == "multitask" and cfg.get("primary_objective") == "bpr":
            pos=np.flatnonzero(ytr>0); neg=np.flatnonzero(ytr<=0)
            rng.shuffle(pos); rng.shuffle(neg); n=min(len(pos),len(neg))
            for start in range(0, n, batch_size):
                end=min(start+batch_size, n)
                losses.append(m.step_bpr(Xtr[pos[start:end]], Xtr[neg[start:end]]))
                if isinstance(m, DualHeadFM) and float(cfg.get("auxiliary_weight", 0.0)) > 0:
                    ax=np.concatenate((pos[start:end], neg[start:end]))
                    aux_scores=m.aux_predict(Xtr[ax])
                    m.apply_aux_gradient(Xtr[ax], float(cfg["auxiliary_weight"]) *
                                         (sigmoid(aux_scores)-aux_tr["is_click"][ax]))
        else:
            batches = (complete_user_batches(train_users,batch_size,rng) if mode=="listwise" else
                       (rng.permutation(len(ytr))[start:start+8192]
                        for start in range(0,len(ytr),batch_size)))
            for ix in batches:
                scores=m.predict(Xtr[ix]); grad=sigmoid(scores)-ytr[ix]
                if mode=="listwise":
                    users_batch=np.asarray([train_users[j] for j in ix],dtype=object)
                    grad=grouped_listwise_gradient(
                        users_batch, ytr[ix], scores,
                        temperature=float(cfg.get("listwise_temperature",1.0)),
                    )
                elif mode=="multitask":
                    bpr=np.zeros_like(grad)
                    gu=np.asarray([train_users[j] for j in ix],dtype=object)
                    order=np.argsort(gu,kind="stable"); su=gu[order]
                    starts=np.r_[0,1+np.flatnonzero(su[1:]!=su[:-1])]
                    gid=np.cumsum(np.isin(np.arange(len(order)),starts))-1; ng=int(gid[-1])+1
                    fp=np.full(ng,len(order),np.int32); fn=fp.copy()
                    np.minimum.at(fp,gid,np.where(ytr[ix][order]>0,np.arange(len(order)),len(order)))
                    np.minimum.at(fn,gid,np.where(ytr[ix][order]<=0,np.arange(len(order)),len(order)))
                    ok=(fp<len(order))&(fn<len(order)); p=order[fp[ok]]; nidx=order[fn[ok]]
                    q=sigmoid(scores[p]-scores[nidx])-1.; bpr[p]=q; bpr[nidx]=-q
                    pw=float(cfg.get("pointwise_weight",.70)); rw=float(cfg.get("pairwise_weight",.25)); aw=float(cfg.get("auxiliary_weight",.05))
                    primary=pw*grad+rw*bpr
                    aux_scores=sigmoid(m.aux_predict(Xtr[ix]) if isinstance(m,DualHeadFM) else scores)
                    aux_tasks=cfg.get("auxiliary_tasks", ["is_click"])
                    aux_weights=cfg.get("auxiliary_task_weights", {})
                    total_aux_weight=sum(float(aux_weights.get(name, 1.0)) for name in aux_tasks) or 1.0
                    aux_terms=[]
                    for name in aux_tasks:
                        residual=aux_scores-aux_tr[name][ix]
                        if cfg.get("auxiliary_normalization") == "prevalence_rms":
                            prevalence=float(np.mean(aux_tr[name]))
                            scale=max(np.sqrt(prevalence*(1.0-prevalence)), 1e-3)
                            residual=residual/scale
                        aux_terms.append(float(aux_weights.get(name, 1.0)) / total_aux_weight * residual)
                    aux=np.sum(aux_terms, axis=0)
                    if cfg.get("auxiliary_normalization") == "prevalence_rms":
                        aux=aux/(np.sqrt(np.mean(aux*aux))+1e-6)
                    if cfg.get("schedule")=="primary_first" and ep<=int(cfg.get("warmup_epochs",4)): aw=0.
                    if cfg.get("gradient_policy")=="project_aux":
                        primary_norm=float(np.dot(primary,primary))+1e-12; aux=aux-(float(np.dot(aux,primary))/primary_norm)*primary
                    grad=primary
                    if isinstance(m,DualHeadFM): m.apply_aux_gradient(Xtr[ix],aw*aux)
                    else: grad=primary+aw*aux
                elif mode=="cwm":
                    pred=sigmoid(scores); target=normalize_watch_time(play_tr[ix],dur_tr[ix])
                    grad=float(cfg.get("cwm_pointwise_weight",.80))*(pred-ytr[ix])+float(cfg.get("cwm_weight",.20))*np.where(ytr[ix]<1,np.minimum(0,pred-target+float(cfg.get("cwm_margin",0.))),pred-target)
                losses.append(float(np.mean(grad*grad))); m.apply_logit_gradient(Xtr[ix],grad)
        va=evaluate(users,yva,ranking_scores(Xva)); print(f"{mode} epoch {ep:02d} loss {np.mean(losses):.6f} validation {va}",flush=True)
        if va["primary"]>best+1e-5:
            best=va["primary"]; state=(m.V.copy(),m.W.copy(),m.b,
                                        getattr(m,"auxW",None).copy() if isinstance(m,DualHeadFM) else None); bad=0
        else: bad+=1
        if bad>=patience: break
    m.V,m.W,m.b,aux_state=state
    if isinstance(m,DualHeadFM) and aux_state is not None: m.auxW=aux_state
    scores=ranking_scores(Xva); va=evaluate(users,yva,scores)
    confirmation=evaluate_confirmation(users,yva,scores)
    experiment = mode + "_fm"
    checkpoint = save_if_best(
        experiment,
        {"V": m.V, "W": m.W, "b": np.asarray(m.b),
         "auxW": getattr(m, "auxW", np.zeros_like(m.W)), "config": json.dumps(cfg)},
        va,
        config=cfg,
        source="agent/formal_trainer.py",
    )
    # Optional independent artifact for an explicitly registered initial
    # checkpoint. Keep it separate from the family canonical-best policy.
    independent_name = cfg.get("independent_checkpoint")
    if independent_name:
        candidate = os.path.abspath(os.path.join(outputs_dir(), str(independent_name)))
        outputs_root = os.path.abspath(outputs_dir())
        if os.path.commonpath([candidate, outputs_root]) != outputs_root:
            raise ValueError("independent checkpoint must stay under outputs_dir")
        os.makedirs(outputs_root, exist_ok=True)
        np.savez(candidate, V=m.V, W=m.W, b=np.asarray(m.b),
                 auxW=getattr(m, "auxW", np.zeros_like(m.W)),
                 config=json.dumps(cfg))
    return {"experiment":mode+"_fm","dataset":dataset_name(),"status":"success","split":"validation_only","test_access":False,
            "metrics":{k:float(v) for k,v in va.items()},"epochs":ep,"config":cfg,
            "confirmation_metrics":{k:float(v) for k,v in confirmation.items()},
            "recovery_events":([warmstart_note] if warmstart_note else []),
            "checkpoint":checkpoint["checkpoint"],
            "checkpoint_saved":checkpoint["checkpoint_saved"],
            "family_best_metrics":checkpoint.get("metrics", {})}


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["listwise","history","multitask","cwm","composition"],required=True)
    ap.add_argument("--limit",type=int); ap.add_argument("--epochs",type=int,default=8)
    ap.add_argument("--patience",type=int); ap.add_argument("--batch-size",type=int,default=8192)
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()
    print(json.dumps(train(a.mode,a.limit,a.epochs,a.seed,a.batch_size,a.patience),indent=2))
