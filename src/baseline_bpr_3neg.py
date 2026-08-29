"""KuaiRand-Pure baselines。
  --model pop   : item popularity（官方 baseline，纯统计，不训练）
  --model fm    : Factorization Machine（起步模型，学生从这里往上改）
  --model random: 随机打分（下界，用来自检评测代码没坏）
只依赖 numpy。用法见 README.md
"""
import argparse, collections, time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity（官方 baseline） ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
# 影片本身好不好
# +這個使用者喜不喜歡
# +這個使用者是否喜歡這類影片
# +這些 feature 彼此的組合

class FM:
    # 把每個feature轉成向量
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    # 總分 =bias+ 每個 feature 的單獨影響+ feature 兩兩互動的影響
    def logits(self, X):
        # 每個 feature 自己的影響 ex.user_id 的影響+video_id 的影響
        E = self.V[X]    # 每筆資料的 feature embedding                              # (B,F,k)
        S = E.sum(1)
        #捕捉個人化偏好                      # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    # 比較正樣本和負樣本
    # 排序誰在前面
    def step(self, X_pos, X_neg):
        B = len(X_pos)

        # 正例和負例各自計算 FM 分數
        z_pos, E_pos, S_pos = self.logits(X_pos)
        z_neg, E_neg, S_neg = self.logits(X_neg)

        # BPR: 希望 z_pos - z_neg 越大越好
        diff = z_pos - z_neg

        # 對 z_pos 的梯度是負的，對 z_neg 的梯度是正的
        g = (sigmoid(-diff) / B).astype(np.float32)
        g_pos = -g
        g_neg = g

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)

        # 正例部分的梯度
        np.add.at(gW, X_pos, g_pos[:, None])
        np.add.at(gV, X_pos,
                g_pos[:, None, None] * (S_pos[:, None, :] - E_pos))

        # 負例部分的梯度
        np.add.at(gW, X_neg, g_neg[:, None])
        np.add.at(gV, X_neg,
                g_neg[:, None, None] * (S_neg[:, None, :] - E_neg))

        # L2 regularization
        gV += self.l2 * self.V
        gW += self.l2 * self.W

        # Adam 更新
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8

        for P, G, M, Vv in (
            (self.V, gV, self.mV, self.vV),
            (self.W, gW, self.mW, self.vW)
        ):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)

            P -= self.lr * (
                M / (1 - b1 ** self.t)
            ) / (
                np.sqrt(Vv / (1 - b2 ** self.t)) + eps
            )

        # BPR 中 bias 對正例和負例會互相抵消，所以不更新 bias
        return float(np.mean(np.logaddexp(0.0, -diff)))

    def predict(self, X, bs=200_000):
        return np.concatenate([
            self.logits(X[i:i + bs])[0]
            for i in range(0, len(X), bs)
        ])

# 原始訓練資料
#     ↓
# 依 user 分組
#     ↓
# 建立正負樣本 pair
#     ↓
# 打亂 pair
#     ↓
# X_pos, X_neg
#     ↓
# m.step(X_pos, X_neg)
#     ↓
# 學習排序
def run_fm(splits, k=16, lr=0.001, epochs=40,
           bs=8192, patience=4, seed=0, verbose=True):

    enc, dim = encode(splits)
    # 需要user資料
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']

    # 依照 user 分別整理正例和負例
    # 正例：long_view = 1
    # 負例：long_view = 0
    pos_by_user = collections.defaultdict(list)
    neg_by_user = collections.defaultdict(list)

    for i, (user, label) in enumerate(zip(utr, ytr)):
        if label == 1:
            pos_by_user[user].append(i)
        else:
            neg_by_user[user].append(i)

    # 建立 BPR 訓練配對：
    # 同一個 user 的一筆正例 + 一筆負例
    pair_pos = []
    pair_neg = []

    rng = np.random.default_rng(seed)
    # 如果某個 user 沒有正例或負例，就無法組成 BPR pair，因此跳過
    for user in pos_by_user:
        if user not in neg_by_user:
            continue

        positives = pos_by_user[user]
        negatives = neg_by_user[user]

        for pos_i in positives:
            neg_i = rng.choice(negatives)
            pair_pos.append(pos_i)
            pair_neg.append(neg_i)

    pair_pos = np.asarray(pair_pos, dtype=np.int32)
    pair_neg = np.asarray(pair_neg, dtype=np.int32)

    if verbose:
        print(f"BPR pairs: {len(pair_pos):,d}")

    # 建立 FM 模型
    m = FM(dim, k=k, lr=lr, seed=seed)

    best = -1
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        # 每個 epoch 重新打亂正負例配對
        order = rng.permutation(len(pair_pos))
        t0 = time.time()
        losses = []

        for start in range(0, len(order), bs):
            batch = order[start:start + bs]

            X_pos = Xtr[pair_pos[batch]]
            X_neg = Xtr[pair_neg[batch]]

            loss = m.step(X_pos, X_neg)
            losses.append(loss)

        # 只使用 validation 選擇最佳模型
        va = evaluate(uva, yva, m.predict(Xva))

        if verbose:
            print(
                f"  epoch {ep:2d} | loss {np.mean(losses):.4f} "
                f"| valid GAUC {va['GAUC']:.4f} "
                f"nDCG@5 {va['nDCG@5']:.4f} "
                f"primary {va['primary']:.4f} "
                f"| {time.time() - t0:.1f}s"
            )

        if va['primary'] > best + 1e-5:
            best = va['primary']
            bad = 0
            best_state = (
                m.V.copy(),
                m.W.copy(),
                np.float32(m.b)
            )
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    # 恢復 validation 表現最好的模型
    m.V, m.W, m.b = best_state

    return {
        'valid': evaluate(uva, yva, m.predict(Xva)),
        'test': evaluate(ute, yte, m.predict(Xte))
    }

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure 解压后的 data 目录')
    ap.add_argument('--model', default='fm', choices=['pop', 'fm', 'random'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = {'pop': run_pop, 'random': lambda s: run_random(s, a.seed),
           'fm': lambda s: run_fm(s, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)}[a.model](splits)
    print(f"\n=== {a.model} (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
