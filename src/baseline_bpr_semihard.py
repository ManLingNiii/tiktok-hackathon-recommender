"""WIP BPR-FM with static semi-hard same-user negative sampling."""
import argparse
import collections
import math
import time

import numpy as np

from baseline_bpr import FM
from data import load, encode, FIELDS
from evaluate import evaluate


def _group_train_rows(users, labels):
    pos_by_user = collections.defaultdict(list)
    neg_by_user = collections.defaultdict(list)
    for i, (user, label) in enumerate(zip(users, labels)):
        (pos_by_user if label == 1 else neg_by_user)[user].append(i)
    return pos_by_user, neg_by_user


def build_random_pairs(users, labels, n_neg=1, seed=0, rng=None):
    """Build flattened random same-user pairs from the supplied train rows."""
    if n_neg not in (1, 3):
        raise ValueError("n_neg must be one of 1 or 3")
    if rng is None:
        rng = np.random.default_rng(seed)
    pos_by_user, neg_by_user = _group_train_rows(users, labels)
    pair_pos, pair_neg = [], []
    for user, positives in pos_by_user.items():
        negatives = neg_by_user.get(user)
        if not negatives:
            continue
        for pos_i in positives:
            sampled = (rng.choice(negatives) if n_neg == 1 else
                       rng.choice(negatives, size=n_neg,
                                  replace=(len(negatives) < n_neg)))
            if n_neg == 1:
                sampled = (sampled,)
            for neg_i in sampled:
                pair_pos.append(pos_i)
                pair_neg.append(neg_i)
    return np.asarray(pair_pos, dtype=np.int32), np.asarray(pair_neg, dtype=np.int32)


def build_semihard_pairs(X, users, labels, model, n_neg=1,
                         semi_hard_fraction=0.3, seed=0, rng=None):
    """Choose random negatives from the highest-scoring fraction, not only the maximum.

    Scores are produced from the supplied warm-up model. All candidates are
    negative train rows for the same user as the positive row.
    """
    if n_neg not in (1, 3):
        raise ValueError("n_neg must be one of 1 or 3")
    if not 0 < semi_hard_fraction <= 1:
        raise ValueError("semi_hard_fraction must be in (0, 1]")
    if rng is None:
        rng = np.random.default_rng(seed)
    pos_by_user, neg_by_user = _group_train_rows(users, labels)
    negative_scores = model.predict(X)
    pair_pos, pair_neg = [], []
    for user, positives in pos_by_user.items():
        negatives = neg_by_user.get(user)
        if not negatives:
            continue
        ranked = sorted(negatives, key=lambda i: negative_scores[i], reverse=True)
        candidate_count = max(1, math.ceil(len(ranked) * semi_hard_fraction))
        candidates = ranked[:candidate_count]
        for pos_i in positives:
            sampled = (rng.choice(candidates) if n_neg == 1 else
                       rng.choice(candidates, size=n_neg,
                                  replace=(len(candidates) < n_neg)))
            if n_neg == 1:
                sampled = (sampled,)
            for neg_i in sampled:
                pair_pos.append(pos_i)
                pair_neg.append(neg_i)
    return np.asarray(pair_pos, dtype=np.int32), np.asarray(pair_neg, dtype=np.int32)


def _train_pairs(model, X, pair_pos, pair_neg, epochs, bs, rng):
    losses = []
    for _ in range(epochs):
        order = rng.permutation(len(pair_pos))
        for start in range(0, len(order), bs):
            batch = order[start:start + bs]
            losses.append(model.step(X[pair_pos[batch]], X[pair_neg[batch]]))
    return losses


def run_fm(splits, k=16, lr=0.001, l2=1e-6, epochs=40,
           bs=8192, patience=4, seed=0, verbose=True, n_neg=1,
           warmup_epochs=1, semi_hard_fraction=0.3):
    enc, dim = encode(splits)
    tr, va, te = (enc[name] for name in ('train', 'valid', 'test'))
    model = FM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    Xtr, ytr, utr = tr
    Xva, yva, uva = va
    Xte, yte, ute = te
    random_pos, random_neg = build_random_pairs(
        utr, ytr, n_neg=n_neg, rng=rng
    )
    if verbose:
        print(f"seed={seed} | n_neg={n_neg} | warmup_epochs={warmup_epochs} | "
              f"semi_hard_fraction={semi_hard_fraction} | k={k} | lr={lr} | "
              f"l2={l2} | batch_size={bs} | epochs={epochs} | patience={patience}")
        print(f"negative_sampling=random_same_user_warmup | pairs: {len(random_pos):,d}")

    warmup = min(max(warmup_epochs, 0), epochs)
    if len(random_pos) and warmup:
        _train_pairs(model, Xtr, random_pos, random_neg, warmup, bs, rng)

    pair_pos, pair_neg = build_semihard_pairs(
        Xtr, utr, ytr, model, n_neg=n_neg,
        semi_hard_fraction=semi_hard_fraction, rng=rng
    )
    if verbose:
        print(f"negative_sampling=semi_hard_same_user_static | pairs: {len(pair_pos):,d}")

    best = -1
    best_state = None
    bad = 0
    for ep in range(warmup + 1, epochs + 1):
        t0 = time.time()
        order = rng.permutation(len(pair_pos))
        losses = []
        for start in range(0, len(order), bs):
            batch = order[start:start + bs]
            losses.append(model.step(Xtr[pair_pos[batch]], Xtr[pair_neg[batch]]))
        metrics = evaluate(uva, yva, model.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} "
                  f"| valid GAUC {metrics['GAUC']:.4f} "
                  f"nDCG@5 {metrics['nDCG@5']:.4f} "
                  f"primary {metrics['primary']:.4f} | {time.time()-t0:.1f}s")
        if metrics['primary'] > best + 1e-5:
            best, bad = metrics['primary'], 0
            best_state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.V, model.W, model.b = best_state
    return {
        'valid': evaluate(uva, yva, model.predict(Xva)),
        'test': evaluate(ute, yte, model.predict(Xte))
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--model', default='fm', choices=['fm'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--l2', type=float, default=1e-6)
    ap.add_argument('--batch_size', type=int, default=8192)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_neg', type=int, choices=[1, 3], default=1)
    ap.add_argument('--warmup_epochs', type=int, default=1)
    ap.add_argument('--semi_hard_fraction', type=float, default=0.3)
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({name: len(rows) for name, rows in splits.items()}, f"fields={FIELDS}")
    res = run_fm(splits, k=a.k, lr=a.lr, l2=a.l2, epochs=a.epochs,
                 bs=a.batch_size, patience=a.patience, seed=a.seed,
                 n_neg=a.n_neg, warmup_epochs=a.warmup_epochs,
                 semi_hard_fraction=a.semi_hard_fraction)
    for split in ('valid', 'test'):
        r = res[split]
        print(f"  {split:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
