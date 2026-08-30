"""WIP BPR + pointwise logistic hybrid loss experiment."""
import argparse
import time

import numpy as np

from baseline_bpr import FM
from baseline_bpr_3neg import build_bpr_pairs
from data import load, encode, FIELDS
from evaluate import evaluate


class HybridFM(FM):
    """Existing FM with a separate, single-update hybrid training step."""

    def step_hybrid(self, X_pos, X_neg, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        B = len(X_pos)
        if B == 0:
            raise ValueError("hybrid batches must contain at least one pair")

        z_pos, E_pos, S_pos = self.logits(X_pos)
        z_neg, E_neg, S_neg = self.logits(X_neg)
        diff = z_pos - z_neg

        # Pairwise BPR gradient, matching the existing FM.step implementation.
        bpr_g = (sigmoid(-diff) / B).astype(np.float32)
        g_pos = (1.0 - alpha) * -bpr_g
        g_neg = (1.0 - alpha) * bpr_g

        # Pointwise logistic gradient over positive=1 and negative=0 rows.
        X_point = np.concatenate((X_pos, X_neg), axis=0)
        y_point = np.concatenate((np.ones(B), np.zeros(B))).astype(np.float32)
        z_point, E_point, S_point = self.logits(X_point)
        point_g = (alpha * (sigmoid(z_point) - y_point) / (2 * B)).astype(np.float32)

        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X_pos, g_pos[:, None])
        np.add.at(gV, X_pos, g_pos[:, None, None] * (S_pos[:, None, :] - E_pos))
        np.add.at(gW, X_neg, g_neg[:, None])
        np.add.at(gV, X_neg, g_neg[:, None, None] * (S_neg[:, None, :] - E_neg))
        np.add.at(gW, X_point, point_g[:, None])
        np.add.at(gV, X_point, point_g[:, None, None] * (S_point[:, None, :] - E_point))

        # Add L2 once, after combining both objective gradients.
        gV += self.l2 * self.V
        gW += self.l2 * self.W

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                            (self.W, gW, self.mW, self.vW)):
            M *= b1
            M += (1 - b1) * G
            Vv *= b2
            Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * point_g.sum()

        bpr_loss = np.mean(np.logaddexp(0.0, -diff))
        point_loss = np.mean(np.logaddexp(0.0, z_point) - y_point * z_point)
        return float(alpha * point_loss + (1.0 - alpha) * bpr_loss)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def run_fm(splits, k=16, lr=0.001, l2=1e-6, epochs=40,
           bs=8192, patience=4, seed=0, verbose=True, n_neg=1,
           alpha=0.5):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    rng = np.random.default_rng(seed)
    pair_pos, pair_neg = build_bpr_pairs(utr, ytr, n_neg=n_neg, rng=rng)
    model = HybridFM(dim, k=k, lr=lr, l2=l2, seed=seed)
    if verbose:
        print(f"seed={seed} | alpha={alpha} | n_neg={n_neg} | k={k} | lr={lr} | "
              f"l2={l2} | batch_size={bs} | epochs={epochs} | patience={patience}")
        print(f"loss=alpha*pointwise+(1-alpha)*BPR | pairs: {len(pair_pos):,d}")

    best = -1.0
    best_state = None
    bad = 0
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(pair_pos))
        t0 = time.time()
        losses = []
        for start in range(0, len(order), bs):
            batch = order[start:start + bs]
            losses.append(model.step_hybrid(Xtr[pair_pos[batch]], Xtr[pair_neg[batch]], alpha))
        va = evaluate(uva, yva, model.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} "
                  f"| valid GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f} "
                  f"primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.V, model.W, model.b = best_state
    return {'valid': evaluate(uva, yva, model.predict(Xva)),
            'test': evaluate(ute, yte, model.predict(Xte))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--model', default='fm', choices=['fm'])
    ap.add_argument('--n_neg', type=int, choices=[1, 3], default=1)
    ap.add_argument('--alpha', type=float, choices=[0.0, 0.25, 0.5, 0.75, 1.0], default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--l2', type=float, default=1e-6)
    ap.add_argument('--batch_size', type=int, default=8192)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=4)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({name: len(rows) for name, rows in splits.items()}, f"fields={FIELDS}")
    res = run_fm(splits, k=a.k, lr=a.lr, l2=a.l2, epochs=a.epochs,
                 bs=a.batch_size, patience=a.patience, seed=a.seed,
                 n_neg=a.n_neg, alpha=a.alpha)
    for split in ('valid', 'test'):
        r = res[split]
        print(f"  {split:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
