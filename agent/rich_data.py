"""Train/validation-only rich loader with leakage-safe history domains."""
import csv, os
import numpy as np

FIELDS = ["user_id","video_id","author_id","tab","dur_bucket",
          "user_hist","user_tab_hist","user_author_hist"]
AUX = ["is_click","is_like","is_follow","is_comment","is_forward"]

def load_rich(data_dir):
    authors={}
    with open(os.path.join(data_dir,"video_features_basic_pure.csv"),encoding="utf-8") as f:
        for r in csv.DictReader(f): authors[r["video_id"]]=r["author_id"]
    tr,va=[],[]
    for fn in ("log_standard_4_08_to_4_21_pure.csv","log_standard_4_22_to_5_08_pure.csv"):
        with open(os.path.join(data_dir,fn),encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d=int(r["date"]); base=(d,r["user_id"],r["video_id"],authors.get(r["video_id"],"UNK"),r["tab"],float(r["duration_ms"]))
                item={"base":base,"y":int(r["long_view"]!="0"),"aux":{k:float(r[k] or 0) for k in AUX},"play":float(r["play_time_ms"] or 0),"duration":float(r["duration_ms"] or 1)}
                if 20220408<=d<=20220421: tr.append(item)
                elif 20220422<=d<=20220428: va.append(item)
    user={}; tab={}; author={}
    for x in sorted(tr,key=lambda z:(z["base"][0],z["base"][1])):
        d,u,v,a,t,_=x["base"]; x["hist"]=(user.get(u,0),tab.get((u,t),0),author.get((u,a),0))
        user[u]=user.get(u,0)+x["y"]; tab[(u,t)]=tab.get((u,t),0)+x["y"]; author[(u,a)]=author.get((u,a),0)+x["y"]
    # Validation may use only history accumulated through the end of train.
    for x in va:
        _,u,_,a,t,_=x["base"]; x["hist"]=(user.get(u,0),tab.get((u,t),0),author.get((u,a),0))
    return tr,va

def encode_rich(train, valid, include_history=True):
    edges=np.quantile(np.asarray([x["base"][5] for x in train]),np.linspace(0,1,11)[1:-1])
    def raw(x):
        b=x["base"]; vals=[b[1],b[2],b[3],b[4],str(int(np.searchsorted(edges,b[5])))]
        if include_history: vals += [str(min(x["hist"][i],20)) for i in range(3)]
        return vals
    voc=[{} for _ in raw(train[0])]
    for x in train:
        for i,v in enumerate(raw(x)):
            if v not in voc[i]: voc[i][v]=len(voc[i])
    dims=[len(v)+1 for v in voc]; offs=np.cumsum([0]+dims[:-1]).astype(np.int32)
    def enc(rows):
        X=np.empty((len(rows),len(voc)),np.int32); y=np.empty(len(rows),np.float32); users=[]
        aux={k:np.empty(len(rows),np.float32) for k in AUX}; play=np.empty(len(rows),np.float32); dur=np.empty(len(rows),np.float32)
        for n,x in enumerate(rows):
            for i,v in enumerate(raw(x)): X[n,i]=voc[i].get(v,len(voc[i]))+offs[i]
            y[n]=x["y"]; users.append(x["base"][1]); play[n]=x["play"]; dur[n]=x["duration"]
            for k in AUX: aux[k][n]=x["aux"][k]
        return X,y,users,aux,play,dur
    return enc(train),enc(valid),int(sum(dims))
