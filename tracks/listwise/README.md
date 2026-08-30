# Listwise Track

[English](#english) | [繁體中文](#繁體中文)

## English

This track changes only the FM training objective. It does not import, modify,
or run BPR, History, or Multi-task code.

### Final selection at a glance

- Checkpoint: `results_by_track/listwise/checkpoints/hpo_v3_01.npz`
- Seed: `0`
- Valid GAUC: `0.667148`
- Valid nDCG@5: `0.535823`
- Valid primary: `0.601486`
- Gain over reproduced baseline: `+0.000016`
- Selection rule: highest validation primary only; deterministic checkpoint-path
  order breaks an exact tie
- Test metrics used: no

## Exposure grouping

The canonical exposure group is **all logged impressions belonging to one
`user_id` within the training split**. Groups never cross users and there is no
sessionization. For the final HPO, each training slate keeps every positive and
the highest baseline-scored negatives from that same user, capped at 32, 64, or
128 where possible. This is still a listwise slate, never a positive/negative
pair. Validation always uses the evaluator's complete user exposure group.

Only discriminative training lists (`0 < positives < impressions`) contribute
to the loss. All-negative and all-positive users have constant labels, so their
within-user order cannot change GAUC or nDCG@5.

## Loss

For user list `u`, predicted distribution and relevance target are:

```text
p_ui = softmax(score_ui / score_temperature)
q_ui = softmax(long_view_ui / target_temperature)
L_u  = -sum_i q_ui log(p_ui)
```

The final search uses `target_temperature=0.35, 0.5, 0.75`. User weights blend
the two components of the official primary metric:

```text
w_u = 0.5 + 0.5 * min(positive_count_u / mean_positive_count, 5)
L   = weighted_mean_u(L_u, w_u)
```

AdamW/SGD, warmup, embedding freezing, and fine-grained validation intervals
were compared. The final candidates use AdamW with validation every 10 or 20
user batches; no pairs are constructed.

The follow-up anchored search also tested a teacher-preserving target:

```text
q_anchor = (1 - alpha) * softmax(baseline_score)
         + alpha * softmax(long_view / target_temperature)
```

It compared linear residual updates (frozen baseline embeddings), full FM
updates, and 25/50/75% hard-negative mixtures. The best anchored run used
`alpha=0.1`, a 50/50 hard/random slate, and full FM updates. It stabilized
fine-tuning but did not surpass the best unanchored soft-target run.

The nDCG-focused search adds a differentiable full-list ApproxNDCG@5 term:

```text
soft_rank_i = 1 + sum_j sigmoid((score_j - score_i) / rank_temperature)
soft_DCG@5  = sum_i long_view_i * top5_gate(soft_rank_i) / log2(1 + soft_rank_i)
L           = (1 - lambda) * ListNet + lambda * (1 - soft_DCG@5 / IDCG@5)
```

Two same-user sampling strategies were compared without constructing BPR pairs:

- S-04 keeps every positive, 24 negatives nearest the baseline top-5 cutoff,
  and random tail negatives up to cap 64.
- S-06 creates two cap-32 lists per user: a top-5-boundary slate and a disjoint
  random-negative slate.

For each strategy, `lambda=0.25, 0.5, 0.75, 1.0` was evaluated. S-04 with
`lambda=0.25` was strongest; increasing the ApproxNDCG weight did not improve
validation primary. S-06 was also slightly weaker than S-04.

The matching position-discounted ListNet search uses a NeuralSort-style soft
permutation for the first five positions:

```text
P_k = softmax(((n + 1 - 2k) * score_i - sum_j |score_i - score_j|) / sort_temp)
L_PD = -sum_k DCG_discount_k * CE(uniform_positive_target, P_k)
L = (1 - position_weight) * ListNet + position_weight * L_PD
```

The same `position_weight=0.25, 0.5, 0.75, 1.0` grid was run for both S-04 and
S-06. It did not beat the final cross-stage Top 1:

| Objective | Sampling | Best valid GAUC | Best valid nDCG@5 | Best valid primary |
|---|---|---:|---:|---:|
| ApproxNDCG@5 | S-04 top-5 boundary | 0.667144 | 0.535820 | 0.601482 |
| Position-discounted ListNet | S-04 top-5 boundary | 0.667134 | 0.535807 | 0.601470 |
| ApproxNDCG@5 | S-06 multi-slate | 0.667145 | 0.535811 | 0.601478 |
| Position-discounted ListNet | S-06 multi-slate | 0.667139 | 0.535809 | 0.601474 |

Within this fixed grid, differentiable ApproxNDCG@5 is the stronger top-5-aware
surrogate. Position weights produced nearly identical AdamW trajectories because
they mostly rescaled rather than redirected the early gradient.

## Fixed workflow

The entry point first reproduces the shared official FM at seed 0 using its
pointwise loss, saving the validation-best baseline state. Every HPO candidate
starts from that identical baseline state and is then updated only by the
configured pure Listwise loss. HPO is fixed to seed 0 and evaluates validation
only. `top1.json` is selected solely by `valid_primary`; the training entry point
never scores test.

```bash
PYTHONPATH=src /opt/anaconda3/envs/tiktok/bin/python tracks/listwise/train.py \
  --data_dir ./KuaiRand-Pure/data --seed 0
```

Outputs are written under `results_by_track/listwise/` as per-run logs, metrics,
checkpoints, and the validation-ranked Top 1 manifest.

## Completed seed-0 results

The reproduced shared baseline selected epoch 7 with validation GAUC `0.667133`,
nDCG@5 `0.535806`, and primary `0.601470`.

| Selection | Checkpoint | Hard cap | Learning rate | Best step | Valid GAUC | Valid nDCG@5 | Valid primary | Gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Top 1 | `hpo_v3_01.npz` | 64 | 0.000001 | 30 | 0.667148 | 0.535823 | 0.601486 | +0.000016 |

The best Listwise result narrowly exceeds the reproduced baseline. The gain is
real under the recorded validation run but too small to claim a material
improvement. No test metrics were computed or used to tune or rank candidates.

---

## 繁體中文

此 track 只修改 FM 的訓練目標，不會匯入、修改或執行 BPR、History 或
Multi-task 的程式碼。

### 最終選模摘要

- Checkpoint：`results_by_track/listwise/checkpoints/hpo_v3_01.npz`
- Seed：`0`
- Validation GAUC：`0.667148`
- Validation nDCG@5：`0.535823`
- Validation primary：`0.601486`
- 相對重現 baseline 的 gain：`+0.000016`
- 選模規則：只取 validation primary 最高者；若數值完全相同，以 checkpoint path
  的固定順序打破平手
- 是否使用 test metrics：否

### Exposure grouping

標準 exposure group 定義為：**training split 中屬於同一個 `user_id` 的全部
logged impressions**。任何 group 都不會跨 user，也不做 sessionization。

HPO 的 training slate 會保留同一 user 的全部正例，再加入該 user 中 baseline
分數較高的負例；依實驗設定盡量限制在 32、64 或 128 筆。這仍然是一個完整的
Listwise slate，不會建立 positive/negative pair。Validation 永遠使用官方 evaluator
定義的完整 user exposure group。

只有具有排序資訊的 training lists（`0 < positives < impressions`）會參與 loss。
全負或全正 user 的 label 固定，其 user 內排序不會改變 GAUC 或 nDCG@5。

### Loss

對 user list `u`，預測分布與 relevance target 為：

```text
p_ui = softmax(score_ui / score_temperature)
q_ui = softmax(long_view_ui / target_temperature)
L_u  = -sum_i q_ui log(p_ui)
```

實驗比較了 `target_temperature=0.35, 0.5, 0.75`。User weighting 用以下方式近似
官方 primary 的兩個組成部分：

```text
w_u = 0.5 + 0.5 * min(positive_count_u / mean_positive_count, 5)
L   = weighted_mean_u(L_u, w_u)
```

我們比較了 AdamW／SGD、warmup、凍結 embedding，以及更細的 validation
interval。正式候選使用 AdamW，每 10 或 20 個 user batches 驗證一次；整個流程
都不會建立 BPR pairs。

後續 anchored search 也測試了保留 baseline 排序資訊的 teacher target：

```text
q_anchor = (1 - alpha) * softmax(baseline_score)
         + alpha * softmax(long_view / target_temperature)
```

此階段比較 linear residual updates（凍結 baseline embeddings）、完整 FM 更新，
以及 25／50／75% hard-negative mixtures。最佳 anchored run 使用 `alpha=0.1`、
50/50 hard/random slate，以及完整 FM 更新。它能穩定 fine-tuning，但沒有超越最佳
unanchored soft-target run。

### Differentiable ApproxNDCG@5

nDCG-oriented search 加入針對完整 user list 的 differentiable ApproxNDCG@5：

```text
soft_rank_i = 1 + sum_j sigmoid((score_j - score_i) / rank_temperature)
soft_DCG@5  = sum_i long_view_i * top5_gate(soft_rank_i) / log2(1 + soft_rank_i)
L           = (1 - lambda) * ListNet + lambda * (1 - soft_DCG@5 / IDCG@5)
```

在不建立 BPR pairs 的前提下，比較兩種 same-user sampling：

- S-04：保留全部正例、距離 baseline top-5 cutoff 最近的 24 個負例，以及 random
  tail negatives，總上限為 64。
- S-06：每位 user 建立兩個 cap-32 lists；一個是 top-5-boundary slate，另一個是
  negatives 不重疊的 random slate。

兩種 sampling 都測試 `lambda=0.25, 0.5, 0.75, 1.0`。S-04 搭配
`lambda=0.25` 最好；提高 ApproxNDCG 權重並沒有提升 validation primary。S-06
則略低於 S-04。

### Position-discounted ListNet

對應的 Position-discounted ListNet 使用 NeuralSort-style soft permutation，直接對
前五個位置加上 DCG discount：

```text
P_k = softmax(((n + 1 - 2k) * score_i - sum_j |score_i - score_j|) / sort_temp)
L_PD = -sum_k DCG_discount_k * CE(uniform_positive_target, P_k)
L = (1 - position_weight) * ListNet + position_weight * L_PD
```

S-04 與 S-06 都測試 `position_weight=0.25, 0.5, 0.75, 1.0`，但沒有進入跨階段
最終 Top 1：

| Objective | Sampling | Best valid GAUC | Best valid nDCG@5 | Best valid primary |
|---|---|---:|---:|---:|
| ApproxNDCG@5 | S-04 top-5 boundary | 0.667144 | 0.535820 | 0.601482 |
| Position-discounted ListNet | S-04 top-5 boundary | 0.667134 | 0.535807 | 0.601470 |
| ApproxNDCG@5 | S-06 multi-slate | 0.667145 | 0.535811 | 0.601478 |
| Position-discounted ListNet | S-06 multi-slate | 0.667139 | 0.535809 | 0.601474 |

在此固定搜尋範圍內，Differentiable ApproxNDCG@5 是較好的 top-5-aware surrogate。
不同 position weights 的 AdamW 軌跡幾乎相同，因為這些權重主要改變 early
gradient 的尺度，而沒有明顯改變方向。

### 固定 workflow

訓練入口會先以 pointwise loss 重現 shared official FM seed-0 baseline，並儲存
validation-best baseline state。所有 HPO candidates 都從同一個 baseline state
開始，再使用設定好的純 Listwise loss 更新。

- Dataset：KuaiRand-Pure
- Split：官方 train／validation／test split，不重新切分
- Seed：固定 `0`
- HPO 與 early stopping：只看 validation
- Top 1 選擇鍵：只使用 `valid_primary`
- Test：不計算、不用於調參或 checkpoint 排序

```bash
PYTHONPATH=src /opt/anaconda3/envs/tiktok/bin/python tracks/listwise/train.py \
  --data_dir ./KuaiRand-Pure/data --seed 0
```

輸出位於 `results_by_track/listwise/`，包含每次執行的 logs、metrics、保留的
checkpoint，以及依 validation primary 選出的 `top1.json`。

### Seed-0 完成結果

重現的 shared baseline 在 epoch 7 最佳：validation GAUC `0.667133`、nDCG@5
`0.535806`、primary `0.601470`。

| Selection | Checkpoint | Hard cap | Learning rate | Best step | Valid GAUC | Valid nDCG@5 | Valid primary | Gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Top 1 | `hpo_v3_01.npz` | 64 | 0.000001 | 30 | 0.667148 | 0.535823 | 0.601486 | +0.000016 |

最佳 Listwise 結果微幅超越重現的 baseline。此 gain 在目前 validation run 中可重現，
但差距太小，不能宣稱為顯著改善。整個 HPO、early stopping 與 Top 1 選擇流程都沒有
計算或使用 test metrics。
