"""Rank registered experiment configs for the autonomous agent.

This is UCB1 for experiment selection, not a replacement for GAUC/nDCG@5.
It reads only validation records and fail-closed ignores test records.
"""
import argparse, json, math, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPACE = os.path.join(ROOT, "agent", "configs", "search_space.json")
LEADERBOARD = os.path.join(ROOT, "runs", "pure", "validation_leaderboard.json")


def _records(path):
    if not os.path.exists(path): return []
    try:
        value=json.load(open(path,encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return []
    if isinstance(value,dict): value=value.get("experiments",[])
    return [x for x in value if x.get("status")=="success" and x.get("split")=="validation_only"
            and not x.get("test_access") and x.get("metrics",{}).get("primary") is not None]


def rank(candidates, records=None, exploration=0.15):
    records=records if records is not None else _records(LEADERBOARD)
    tried={x.get("config",{}).get("name") for x in records}
    tried.discard(None)
    by={}
    for r in records:
        name=r.get("config",{}).get("name") or r.get("experiment")
        by.setdefault(name,[]).append(float(r["metrics"]["primary"]))
    observed=[v for vals in by.values() for v in vals]
    baseline=max(observed) if observed else 0.0
    total=max(1,len(observed))
    ranked=[]
    for c in candidates:
        name=c.get("name"); vals=by.get(name,[])
        mean=sum(vals)/len(vals) if vals else baseline
        bonus=exploration*math.sqrt(math.log(total+1)/(len(vals)+1))
        ranked.append({"name":name,"family":(c.get("families") or ["generic"])[0],
                       "ucb":mean+bonus,"mean_primary":mean,"trials":len(vals),
                       "untried":name not in tried})
    return sorted(ranked,key=lambda x:(x["untried"],x["ucb"]),reverse=True)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--top",type=int,default=10); a=ap.parse_args()
    choices=json.load(open(SPACE,encoding="utf-8"))["candidates"]
    print(json.dumps(rank(choices)[:a.top],ensure_ascii=False,indent=2))
