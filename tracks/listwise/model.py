"""Listwise FM model for ranking logged impressions within each user.

This module deliberately depends only on NumPy and the shared feature encoding.
It does not import or reuse any of the BPR, history, or multi-task tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ExposureGroup:
    """Row indices for one user's complete exposure list in one split."""

    user_id: str
    row_indices: np.ndarray
    positives: int


def group_user_exposures(user_ids: Sequence[str], labels: np.ndarray) -> list[ExposureGroup]:
    """Group rows by user without changing row order inside a user's list.

    A group is the complete set of logged impressions for one user in the given
    split.  Groups never cross users and no sessionization or negative sampling
    is applied.
    """

    if len(user_ids) != len(labels):
        raise ValueError("user_ids and labels must have identical lengths")
    rows_by_user: dict[str, list[int]] = {}
    for row, user_id in enumerate(user_ids):
        rows_by_user.setdefault(user_id, []).append(row)
    groups = [
        ExposureGroup(
            user_id=user_id,
            row_indices=np.asarray(rows, dtype=np.int64),
            positives=int(labels[rows].sum()),
        )
        for user_id, rows in rows_by_user.items()
    ]
    if sum(len(group.row_indices) for group in groups) != len(labels):
        raise AssertionError("exposure grouping lost or duplicated rows")
    return groups


def listwise_softmax_gradient(
    logits: np.ndarray,
    labels: np.ndarray,
    group_sizes: Iterable[int],
    score_temperature: float = 1.0,
    target_temperature: float = 0.1,
    group_weights: Iterable[float] | None = None,
    anchor_logits: np.ndarray | None = None,
    anchor_mix: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Return weighted soft-target ListNet loss and dL/dlogit.

    For each user, p=softmax(score/score_temperature) and
    q=softmax(label/target_temperature). Callers pass only discriminative groups
    because constant-label users cannot affect the official ranking metrics.
    """

    if score_temperature <= 0 or target_temperature <= 0:
        raise ValueError("score and target temperatures must be positive")
    if not 0.0 <= anchor_mix <= 1.0:
        raise ValueError("anchor_mix must be between 0 and 1")
    group_sizes = list(group_sizes)
    if sum(group_sizes) != len(logits) or len(labels) != len(logits):
        raise ValueError("group sizes must cover logits and labels exactly")
    if not group_sizes:
        raise ValueError("at least one exposure group is required")
    if anchor_logits is not None and len(anchor_logits) != len(logits):
        raise ValueError("anchor logits must cover the same rows as logits")
    if group_weights is None:
        weights = np.ones(len(group_sizes), dtype=np.float64)
    else:
        weights = np.asarray(list(group_weights), dtype=np.float64)
        if len(weights) != len(group_sizes) or np.any(weights <= 0):
            raise ValueError("group weights must be positive and match group sizes")
    weight_sum = float(weights.sum())

    grad = np.empty_like(logits, dtype=np.float32)
    total_loss = 0.0
    start = 0
    for group_index, size in enumerate(group_sizes):
        stop = start + size
        y = labels[start:stop]
        positives = float(y.sum())
        if positives <= 0 or positives >= size:
            raise ValueError("listwise loss requires discriminative user groups")
        scaled = logits[start:stop] / score_temperature
        shifted = scaled - scaled.max()
        exp_scores = np.exp(shifted)
        probabilities = exp_scores / exp_scores.sum()
        target_scaled = y / target_temperature
        target_exp = np.exp(target_scaled - target_scaled.max())
        label_target = target_exp / target_exp.sum()
        if anchor_logits is None or anchor_mix == 1.0:
            target = label_target
        else:
            anchor_scaled = anchor_logits[start:stop] / score_temperature
            anchor_exp = np.exp(anchor_scaled - anchor_scaled.max())
            anchor_target = anchor_exp / anchor_exp.sum()
            target = (1.0 - anchor_mix) * anchor_target + anchor_mix * label_target
        normalized_weight = float(weights[group_index] / weight_sum)
        total_loss -= normalized_weight * float(
            np.dot(target, np.log(probabilities + 1e-12))
        )
        grad[start:stop] = (
            normalized_weight * (probabilities - target) / score_temperature
        )
        start = stop

    return total_loss, grad


def approx_ndcg_gradient(
    logits: np.ndarray,
    labels: np.ndarray,
    group_sizes: Iterable[int],
    rank_temperature: float = 0.5,
    cutoff_temperature: float = 0.5,
    group_weights: Iterable[float] | None = None,
    k: int = 5,
) -> tuple[float, np.ndarray]:
    """Return differentiable ApproxNDCG@k loss and dL/dlogit.

    Every user's full slate is processed as one score matrix. The implementation
    does not construct sampled positive/negative pairs or use a BPR objective.
    """

    if rank_temperature <= 0 or cutoff_temperature <= 0:
        raise ValueError("ApproxNDCG temperatures must be positive")
    group_sizes = list(group_sizes)
    if sum(group_sizes) != len(logits) or len(labels) != len(logits):
        raise ValueError("group sizes must cover logits and labels exactly")
    if group_weights is None:
        weights = np.ones(len(group_sizes), dtype=np.float64)
    else:
        weights = np.asarray(list(group_weights), dtype=np.float64)
        if len(weights) != len(group_sizes) or np.any(weights <= 0):
            raise ValueError("group weights must be positive and match group sizes")
    weight_sum = float(weights.sum())
    gradient = np.zeros_like(logits, dtype=np.float32)
    total_loss = 0.0
    start = 0
    log2 = np.log(2.0)

    for group_index, size in enumerate(group_sizes):
        stop = start + size
        scores = logits[start:stop].astype(np.float64)
        relevance = labels[start:stop].astype(np.float64)
        positives = int(relevance.sum())
        if positives <= 0 or positives >= size:
            raise ValueError("ApproxNDCG requires discriminative user groups")

        # comparison[i,j] estimates whether item j ranks above item i.
        differences = (scores[None, :] - scores[:, None]) / rank_temperature
        comparison = 1.0 / (1.0 + np.exp(-np.clip(differences, -30, 30)))
        np.fill_diagonal(comparison, 0.0)
        soft_rank = 1.0 + comparison.sum(axis=1)
        gate = 1.0 / (
            1.0 + np.exp(-np.clip((k + 0.5 - soft_rank) / cutoff_temperature, -30, 30))
        )
        rank_log = np.log1p(soft_rank)
        discount = log2 / rank_log
        ideal_positions = np.arange(1, min(positives, k) + 1, dtype=np.float64)
        idcg = float(np.sum(1.0 / np.log2(ideal_positions + 1.0)))
        dcg = float(np.sum(relevance * gate * discount))
        normalized_weight = float(weights[group_index] / weight_sum)
        total_loss += normalized_weight * (1.0 - dcg / idcg)

        gate_derivative = -gate * (1.0 - gate) / cutoff_temperature
        discount_derivative = -log2 / ((1.0 + soft_rank) * rank_log**2)
        loss_by_rank = -normalized_weight * relevance * (
            gate_derivative * discount + gate * discount_derivative
        ) / idcg
        comparison_derivative = (
            comparison * (1.0 - comparison) / rank_temperature
        )
        np.fill_diagonal(comparison_derivative, 0.0)
        group_gradient = (
            loss_by_rank @ comparison_derivative
            - loss_by_rank * comparison_derivative.sum(axis=1)
        )
        gradient[start:stop] = group_gradient.astype(np.float32)
        start = stop

    return total_loss, gradient


def position_discounted_listnet_gradient(
    logits: np.ndarray,
    labels: np.ndarray,
    group_sizes: Iterable[int],
    sort_temperature: float = 0.5,
    group_weights: Iterable[float] | None = None,
    k: int = 5,
) -> tuple[float, np.ndarray]:
    """Return NeuralSort-style position-discounted ListNet loss and gradient.

    The first min(k, positives) soft permutation rows target the user's positive
    set and are weighted by DCG discounts. Each user list is optimized jointly;
    no BPR training pairs are created.
    """

    if sort_temperature <= 0:
        raise ValueError("sort temperature must be positive")
    group_sizes = list(group_sizes)
    if sum(group_sizes) != len(logits) or len(labels) != len(logits):
        raise ValueError("group sizes must cover logits and labels exactly")
    if group_weights is None:
        weights = np.ones(len(group_sizes), dtype=np.float64)
    else:
        weights = np.asarray(list(group_weights), dtype=np.float64)
        if len(weights) != len(group_sizes) or np.any(weights <= 0):
            raise ValueError("group weights must be positive and match group sizes")
    weight_sum = float(weights.sum())
    gradient = np.zeros_like(logits, dtype=np.float32)
    total_loss = 0.0
    start = 0

    for group_index, size in enumerate(group_sizes):
        stop = start + size
        scores = logits[start:stop].astype(np.float64)
        relevance = labels[start:stop].astype(np.float64)
        positives = int(relevance.sum())
        if positives <= 0 or positives >= size:
            raise ValueError("position-discounted ListNet requires discriminative groups")
        target = relevance / positives
        pair_sign = np.sign(scores[:, None] - scores[None, :])
        absolute_sum = np.abs(scores[:, None] - scores[None, :]).sum(axis=1)
        sign_row_sum = pair_sign.sum(axis=1)
        positions = np.arange(1, min(k, positives) + 1, dtype=np.float64)
        discounts = 1.0 / np.log2(positions + 1.0)
        discounts /= discounts.sum()
        user_weight = float(weights[group_index] / weight_sum)
        group_gradient = np.zeros(size, dtype=np.float64)

        for position, discount in zip(positions.astype(int), discounts):
            coefficient = size + 1 - 2 * position
            soft_logits = (coefficient * scores - absolute_sum) / sort_temperature
            shifted = soft_logits - soft_logits.max()
            probabilities = np.exp(shifted)
            probabilities /= probabilities.sum()
            row_weight = user_weight * float(discount)
            total_loss -= row_weight * float(
                np.dot(target, np.log(probabilities + 1e-12))
            )
            logit_gradient = row_weight * (probabilities - target)
            group_gradient += (
                coefficient * logit_gradient
                - logit_gradient * sign_row_sum
                + logit_gradient @ pair_sign
            ) / sort_temperature

        gradient[start:stop] = group_gradient.astype(np.float32)
        start = stop

    return total_loss, gradient


class ListwiseFM:
    """Factorization Machine trained with full-user listwise softmax loss."""

    def __init__(
        self,
        dim: int,
        k: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        score_temperature: float = 1.0,
        target_temperature: float = 0.1,
        anchor_mix: float = 1.0,
        ndcg_weight: float = 0.0,
        rank_temperature: float = 0.5,
        cutoff_temperature: float = 0.5,
        position_weight: float = 0.0,
        sort_temperature: float = 0.5,
        optimizer: str = "adamw",
        warmup_steps: int = 0,
        update_embeddings: bool = True,
        seed: int = 0,
    ) -> None:
        if optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be 'adamw' or 'sgd'")
        if not 0.0 <= ndcg_weight <= 1.0:
            raise ValueError("ndcg_weight must be between 0 and 1")
        if not 0.0 <= position_weight <= 1.0:
            raise ValueError("position_weight must be between 0 and 1")
        if ndcg_weight + position_weight > 1.0:
            raise ValueError("Listwise objective weights cannot sum above 1")
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr = lr
        self.weight_decay = weight_decay
        self.score_temperature = score_temperature
        self.target_temperature = target_temperature
        self.anchor_mix = anchor_mix
        self.ndcg_weight = ndcg_weight
        self.rank_temperature = rank_temperature
        self.cutoff_temperature = cutoff_temperature
        self.position_weight = position_weight
        self.sort_temperature = sort_temperature
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.update_embeddings = update_embeddings
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        embeddings = self.V[X]
        summed = embeddings.sum(axis=1)
        interactions = 0.5 * (
            (summed**2).sum(axis=1) - (embeddings**2).sum(axis=(1, 2))
        )
        scores = self.b + self.W[X].sum(axis=1) + interactions
        return scores, embeddings, summed

    def step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_sizes: Sequence[int],
        group_weights: Sequence[float] | None = None,
        anchor_logits: np.ndarray | None = None,
    ) -> float:
        scores, embeddings, summed = self.logits(X)
        listnet_loss, listnet_gradient = listwise_softmax_gradient(
            scores,
            y,
            group_sizes,
            score_temperature=self.score_temperature,
            target_temperature=self.target_temperature,
            group_weights=group_weights,
            anchor_logits=anchor_logits,
            anchor_mix=self.anchor_mix,
        )
        score_grad = (1.0 - self.ndcg_weight - self.position_weight) * listnet_gradient
        loss = (1.0 - self.ndcg_weight - self.position_weight) * listnet_loss
        if self.ndcg_weight:
            ndcg_loss, ndcg_gradient = approx_ndcg_gradient(
                scores,
                y,
                group_sizes,
                rank_temperature=self.rank_temperature,
                cutoff_temperature=self.cutoff_temperature,
                group_weights=group_weights,
            )
            loss += self.ndcg_weight * ndcg_loss
            score_grad += self.ndcg_weight * ndcg_gradient
        if self.position_weight:
            position_loss, position_gradient = position_discounted_listnet_gradient(
                scores,
                y,
                group_sizes,
                sort_temperature=self.sort_temperature,
                group_weights=group_weights,
            )
            loss += self.position_weight * position_loss
            score_grad += self.position_weight * position_gradient
        grad_v = np.zeros_like(self.V)
        grad_w = np.zeros_like(self.W)
        np.add.at(grad_w, X, score_grad[:, None])
        np.add.at(
            grad_v,
            X,
            score_grad[:, None, None] * (summed[:, None, :] - embeddings),
        )
        self.t += 1
        learning_rate = self.lr
        if self.warmup_steps:
            learning_rate *= min(1.0, self.t / self.warmup_steps)
        if self.optimizer == "sgd":
            if self.update_embeddings:
                self.V *= 1.0 - learning_rate * self.weight_decay
                self.V -= learning_rate * grad_v
            self.W *= 1.0 - learning_rate * self.weight_decay
            self.W -= learning_rate * grad_w
            return loss

        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        for parameter, gradient, first, second in (
            (self.V, grad_v, self.mV, self.vV),
            (self.W, grad_w, self.mW, self.vW),
        ):
            if parameter is self.V and not self.update_embeddings:
                continue
            first *= beta1
            first += (1 - beta1) * gradient
            second *= beta2
            second += (1 - beta2) * gradient * gradient
            first_hat = first / (1 - beta1**self.t)
            second_hat = second / (1 - beta2**self.t)
            parameter *= 1.0 - learning_rate * self.weight_decay
            parameter -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        # A per-list softmax is shift-invariant, so the global bias has zero gradient.
        return loss

    def predict(self, X: np.ndarray, batch_size: int = 200_000) -> np.ndarray:
        return np.concatenate(
            [
                self.logits(X[start : start + batch_size])[0]
                for start in range(0, len(X), batch_size)
            ]
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"V": self.V.copy(), "W": self.W.copy(), "b": np.asarray(self.b)}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.V = state["V"].copy()
        self.W = state["W"].copy()
        self.b = np.float32(state["b"])
