"""Rigorous top-1 History-FM selection and final training (seed=0)."""
import argparse,json,os,sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0,os.path.join(ROOT,'src'))
from data import load,encode
from evaluate import evaluate
from baseline import FM,sigmoid

FIELDS=['same_author_as_candidate','author_interaction_count','author_interaction_ratio','author_affinity_rate','recent_engagement_rate','dur_bucket_affinity']
def history(train,rows,prefix=False):
 q=np.quantile([x[5] for x in train],np.linspace(0,1,11)[1:-1]); g=sum(x[6] for x in train)/len(train); c={};t={};r={}; out=np.zeros((len(rows),6),np.float32)
 if not prefix:
  for x in train:
   u,a,d=x[1],x[3],int(np.searchsorted(q,x[5])); y=float(x[6]); c[u,a]=c.get((u,a),0)+1;t[u]=t.get(u,0)+1;c[u,a,'p']=c.get((u,a,'p'),0)+y;c[u,d,'p']=c.get((u,d,'p'),0)+y;c[u,d,'n']=c.get((u,d,'n'),0)+1;r.setdefault(u,[]).append(y)
 for i,x in enumerate(rows):
  u,a,d=x[1],x[3],int(np.searchsorted(q,x[5])); n=t.get(u,0); z=c.get((u,a),0); ap=c.get((u,a,'p'),0); dp=c.get((u,d,'p'),0); dn=c.get((u,d,'n'),0); out[i]=[z>0,np.log1p(z),z/max(n,1),(ap+20*g)/(z+20),np.mean(r.get(u,[])[-20:]) if r.get(u) else g,(dp+20*g)/(dn+20)]
  if prefix:
   y=float(x[6]);c[u,a]=z+1;t[u]=n+1;c[u,a,'p']=ap+y;c[u,d,'p']=dp+y;c[u,d,'n']=dn+1;r.setdefault(u,[]).append(y)
 return out

class Model:
 def __init__(self,dim,k,lr,l2):
  q=np.random.default_rng(0); self.V=q.normal(0,.01,(dim,k)).astype(np.float32);self.W=np.zeros(dim,np.float32);self.Vh=q.normal(0,.01,(1,k)).astype(np.float32);self.Wh=np.zeros(1,np.float32);self.b=np.float32(0);self.lr=lr;self.l2=l2;self.n=0;self.m=[np.zeros_like(x) for x in (self.V,self.W,self.Vh,self.Wh)];self.v=[np.zeros_like(x) for x in (self.V,self.W,self.Vh,self.Wh)]
 def score(self,X,H):
  E=self.V[X];S=E.sum(1);Eh=self.Vh[None]*H[:,:,None];Sh=Eh.sum(1);return self.b+self.W[X].sum(1)+.5*((S*S).sum(1)-(E*E).sum((1,2)))+H@self.Wh+(S*Sh).sum(1),E,S,Eh,Sh
 def step(self,X,H,y):
  z,E,S,Eh,Sh=self.score(X,H);g=((sigmoid(z)-y)/len(y)).astype(np.float32);a=np.zeros_like(self.V);b=np.zeros_like(self.W);c=np.sum(g[:,None,None]*H[:,:,None]*(S[:,None,:]+Sh[:,None,:]-Eh),0);d=np.sum(g[:,None]*H,0);np.add.at(b,X,g[:,None]);np.add.at(a,X,g[:,None,None]*(S[:,None,:]+Sh[:,None,:]-E));self.n+=1
  for p,x,m,v in ((self.V,a,self.m[0],self.v[0]),(self.W,b,self.m[1],self.v[1]),(self.Vh,c,self.m[2],self.v[2]),(self.Wh,d,self.m[3],self.v[3])):
   x+=self.l2*p;m[:]=.9*m+.1*x;v[:]=.999*v+.001*x*x;p-=self.lr*(m/(1-.9**self.n))/(np.sqrt(v/(1-.999**self.n))+1e-8)
  self.b-=self.lr*g.sum()
 def predict(self,X,H):return self.score(X,H)[0]

def train_candidate(X,y,Xv,yv,uv,H,Hv,dim,a):
 m=Model(dim, a.k,a.lr,a.l2);rng=np.random.default_rng(0);best=(-1,None,0)
 for ep in range(1,a.epochs+1):
  ix=rng.permutation(len(y))
  for j in range(0,len(ix),a.batch_size):m.step(X[ix[j:j+a.batch_size]],H[ix[j:j+a.batch_size]],y[ix[j:j+a.batch_size]])
  v=evaluate(uv,yv,m.predict(Xv,Hv))
  if v['primary']>best[0]+1e-5:best=(v['primary'],(m.V.copy(),m.W.copy(),m.Vh.copy(),m.Wh.copy(),m.b),ep)
 m.V,m.W,m.Vh,m.Wh,m.b=best[1];return m,evaluate(uv,yv,m.predict(Xv,Hv)),best[2]

def main(a):
 s=load(a.data_dir);e,dim=encode(s);X,y,_=e['train'];Xv,yv,uv=e['valid'];H=history(s['train'],s['train'],True);Hv=history(s['train'],s['valid']);rows=[];models=[]
 for i,name in enumerate(FIELDS):
  m,v,ep=train_candidate(X,y,Xv,yv,uv,H[:,i:i+1],Hv[:,i:i+1],dim,a);rows.append({'feature':name,'valid_GAUC':float(v['GAUC']),'valid_nDCG@5':float(v['nDCG@5']),'valid_primary':float(v['primary']),'best_epoch':int(ep),'learned_linear_weight':float(m.Wh[0])});models.append(m)
 rows.sort(key=lambda x:x['valid_primary'],reverse=True);selected=rows[0]['feature'];m=models[FIELDS.index(selected)];rel='results_by_track/history/checkpoints/history_top1_seed0.npz';os.makedirs(os.path.join(ROOT,'results_by_track/history/checkpoints'),exist_ok=True);np.savez(os.path.join(ROOT,rel),V=m.V,W=m.W,Vh=m.Vh,Wh=m.Wh,b=m.b,selected_feature=selected,seed=0,best_epoch=rows[0]['best_epoch'])
 r={'track':'history','method':'end_to_end_learned_top1_history_fm','selected_top1':selected,'candidate_results':rows,'history_fields_used':[selected],'hyperparameters':{'k':a.k,'lr':a.lr,'l2':a.l2,'batch_size':a.batch_size,'epochs':a.epochs,'seed':0},'checkpoint_path':rel,'best_epoch':rows[0]['best_epoch'],'valid_GAUC':rows[0]['valid_GAUC'],'valid_nDCG@5':rows[0]['valid_nDCG@5'],'valid_primary':rows[0]['valid_primary'],'baseline_valid_primary':.6015,'gain_vs_baseline':rows[0]['valid_primary']-.6015,'selection':'six candidates independently trained end-to-end; top-1 selected by validation primary; selected feature weight learned by gradient descent','test_used':False}
 os.makedirs(os.path.join(ROOT,'results_by_track/history/metrics'),exist_ok=True)
 with open(os.path.join(ROOT,'results_by_track/history/metrics/history_top1_seed0.json'),'w') as f:json.dump(r,f,indent=2)
 with open(os.path.join(ROOT,'results_by_track/history/manifest.json'),'w') as f:json.dump({'model':r,'selected_feature':selected,'feature_selection':'six independent end-to-end candidate models'},f,indent=2)
 print(json.dumps(r,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data_dir',required=True);p.add_argument('--k',type=int,default=16);p.add_argument('--lr',type=float,default=.001);p.add_argument('--l2',type=float,default=1e-6);p.add_argument('--batch_size',type=int,default=8192);p.add_argument('--epochs',type=int,default=15);main(p.parse_args())
