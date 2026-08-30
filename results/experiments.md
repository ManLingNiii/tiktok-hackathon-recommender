# Experiments

## Experiment 0: Random sanity check

<!--
用隨機分數測試資料讀取和評估程式是否正常。
這不是正式模型。
-->

- File: `baseline.py`
- Model: Random
- Seed: 0
- Valid primary: 0.4827
- Test primary: 0.4757
- Status: Passed

## Experiment 1: Official FM baseline

<!--
官方提供的起始模型。
之後所有改進都要和這個模型比較。
-->

### Configuration

- File: `baseline.py`
- Model: Factorization Machine
- Loss: Pointwise log loss
- Features: `user_id`, `video_id`, `author_id`, `tab`, `dur_bucket`
- Embedding dimension: `k=16`
- Learning rate: `0.001`
- Batch size: `8192`
- Maximum epochs: `40`
- Early stopping patience: `4`

### Results

<!--
使用不同 seed 重複執行，確認 baseline 結果是否穩定。
-->

| Seed | Valid GAUC | Valid nDCG@5 | Valid primary | Test GAUC | Test nDCG@5 | Test primary |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6671 | 0.5358 | 0.6015 | — | — | 0.5953 |
| 4 | 0.6679 | 0.5361 | 0.6020 | 0.6596 | 0.5267 | 0.5931 |

- Status: Passed

## Experiment 2: FM with BPR loss

<!--
只把 pointwise loss 改成 BPR loss，
測試更符合排序目標的訓練方式是否有效。
其他設定先保持和 FM baseline 一樣。
-->

### Hypothesis

Pairwise/BPR loss may improve ranking metrics because the official evaluation focuses on ranking videos within each user.

### Configuration

- File: `baseline_bpr.py`
- Model: Factorization Machine
- Loss: BPR / pairwise loss
- Negative sampling: random same-user negative from train split only
- `n_neg`: 1 for the original `baseline_bpr.py`; configurable as 1, 3, or 5 in `baseline_bpr_3neg.py`
- Features: Same as the official FM baseline
- Embedding dimension: `k=16`
- Learning rate: `0.001`
- Batch size: `8192`
- Maximum epochs: `40`
- Early stopping patience: `4`
- Seed: 4

### Required record fields for future runs

| track | experiment id | method | negative_sampling | n_neg | seed | k | lr | l2 | batch_size | epochs | patience | valid GAUC | valid nDCG@5 | valid primary | training time | checkpoint | git commit | notes |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| bpr | `<id>` | bpr | random_same_user | 3 | 0 | 16 | 0.001 | `<value>` | 8192 | 40 | 4 | `<fill>` | `<fill>` | `<fill>` | `<fill>` | `<path>` | `<commit>` | WIP |

## Experiment 4: Static semi-hard negative sampling (WIP)

No result is recorded yet. For future runs, use `method=bpr_fm`, `negative_sampling=semi_hard_same_user_static`, `n_neg`, `warmup_epochs`, and `semi_hard_fraction`, together with the shared hyperparameters and validation metrics. The first comparison should keep seed `0` and all settings fixed except sampling method.

## Experiment 5: BPR + pointwise hybrid loss (WIP)

No result is recorded yet. Future runs should record `method`, `loss_type`, `alpha`, `negative_sampling`, `n_neg`, `seed`, `k`, `lr`, `l2`, `batch_size`, `epochs`, `patience`, valid/test GAUC, valid/test nDCG@5, valid/test primary, checkpoint, and notes. The first comparison uses `n_neg=1`, `seed=0`, and alpha `0.25`, `0.5`, `0.75`; alpha `0` is the pure-BPR reference. Do not call seed-0 results statistically significant.

### Experiment 5 result: alpha comparison, seed 0

All runs used `n_neg=1`, `seed=0`, `k=16`, `lr=0.001`, `l2=1e-6`, `batch_size=8192`, `epochs=40`, and `patience=4`. The pure-BPR reference is valid primary `0.6021` and test primary `0.5955`. Each run produced `382,579` pairs.

| Method | alpha | Best epoch | Valid GAUC | Valid nDCG@5 | Valid primary | Gain vs pure BPR | Test GAUC | Test nDCG@5 | Test primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pure BPR | 0.00 | 5 | 0.6677 | 0.5366 | 0.6021 | 0.0000 | 0.6617 | 0.5293 | 0.5955 |
| Hybrid | 0.25 | 5 | 0.6674 | 0.5361 | 0.6018 | -0.0003 | 0.6617 | 0.5290 | 0.5953 |
| Hybrid | 0.50 | 5 | 0.6674 | 0.5359 | 0.6016 | -0.0005 | 0.6617 | 0.5287 | 0.5952 |
| Hybrid | 0.75 | 5 | 0.6669 | 0.5358 | 0.6013 | -0.0008 | 0.6620 | 0.5290 | 0.5955 |

- Classification: no meaningful improvement; all hybrid settings are small validation regressions.
- Validation GAUC and nDCG@5 both decrease for all hybrid settings relative to pure BPR.
- Test metrics move in the same general direction, with alpha `0.75` returning to the same rounded test primary as pure BPR.
- This is seed-0 evidence only and is not statistically significant.
- No checkpoint or total training time was emitted by the current script; both remain unavailable.
- Machine-readable record: `results_by_track/bpr/metrics/hybrid_alpha_comparison_seed0.json`

### Experiment 4 result: random vs static semi-hard, seed 0

All settings were fixed at `model=FM`, `n_neg=1`, `k=16`, `lr=0.001`, `l2=1e-6`, `batch_size=8192`, `epochs=40`, `patience=4`, and `seed=0`. The semi-hard run used `warmup_epochs=1` and `semi_hard_fraction=0.3`. Both methods produced `382,579` pairs.

| Method | Valid GAUC | Valid nDCG@5 | Valid primary | Gain vs random | Test GAUC | Test nDCG@5 | Test primary |
|---|---:|---:|---:|---:|---:|---:|---:|
| random same-user | 0.6677 | 0.5366 | 0.6021 | 0.0000 | 0.6617 | 0.5293 | 0.5955 |
| semi-hard same-user | 0.5660 | 0.4957 | 0.5308 | -0.0713 | 0.5510 | 0.4783 | 0.5147 |

- Classification: regression in this seed-0 comparison.
- Semi-hard lowered both GAUC and nDCG@5; validation and test moved in the same negative direction.
- It did not meet the `+0.002` improvement criterion and should not be promoted as a Top 3 candidate.
- No checkpoint or total training time was emitted by the current script, so both are recorded as unavailable rather than fabricated.
- Machine-readable record: `results_by_track/bpr/metrics/random_vs_semihard_seed0.json`

### Results

<!--
TBD 代表實驗還沒完成，完成後填入結果。
-->

| Seed | Valid GAUC | Valid nDCG@5 | Valid primary | Test GAUC | Test nDCG@5 | Test primary |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6677 | 0.5366 | 0.6021 | 0.6617 | 0.5293 | 0.5955 |
| 4 | 0.6673 | 0.5354 | 0.6013 | 0.6608 | 0.5285 | 0.5947 |

- Conclusion: BPR did not consistently outperform the official FM baseline across seeds 0 and 4.
- Status: WIP / historical result; not a finalized BPR result

## Experiment 3: Multiple random same-user negatives (seed 0)

Only `n_neg` varied. All runs used KuaiRand-Pure, the shared v1 split, `k=16`, `lr=0.001`, `l2=1e-6`, batch size `8192`, maximum epochs `40`, patience `4`, and seed `0`. Negative samples came only from train rows belonging to the same user. Test metrics below are reported only and were not used for selection. Git commit: `6df800fed1e97b3dc4e878c4d91240c0df171bbf`.

| Method | n_neg | Seed | Best epoch | Pairs | Valid GAUC | Valid nDCG@5 | Valid primary | Gain vs n_neg=1 | Test primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BPR | 1 | 0 | 5 | 382,579 | 0.6677 | 0.5366 | 0.6021 | 0.0000 | 0.5955 |
| BPR | 3 | 0 | 2 | 1,147,737 | 0.6684 | 0.5364 | 0.6024 | +0.0003 | 0.5969 |
| BPR | 5 | 0 | 2 | 1,912,895 | 0.6684 | 0.5354 | 0.6019 | -0.0002 | 0.5961 |

- Status: Preliminary seed-0 evidence only; not statistically significant and not finalized.
- `n_neg=3` is a marginal candidate improvement on validation primary, but does not reach the approximate `+0.002` threshold.
- `n_neg=5` does not improve validation primary and has lower validation nDCG@5.
- No checkpoint was generated by the current script; no checkpoint path is claimed.
- Detailed machine-readable results: `results_by_track/bpr/metrics/n_neg_comparison_seed0.json`

## Shared experiment record template

For each run, record:

- Track
- Experiment ID
- Method
- Hyperparameters
- Seed
- Best epoch
- Valid GAUC
- Valid nDCG@5
- Valid primary
- Checkpoint
- Git commit
- Notes

## Experiment 4: History aggregate pointwise model (seed 0)

使用七個 aggregate history features、linear logistic pointwise model、binary logistic loss。共同 split 與 seed 固定不變；test 僅報告，不用於選擇模型。

| Method | Best epoch | Valid GAUC | Valid nDCG@5 | Valid primary | Gain vs BPR n_neg=1 | Test GAUC | Test nDCG@5 | Test primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| History linear logistic | 1 | 0.5155245 | 0.47333515 | 0.49442983 | -0.10767017 | 0.50598377 | 0.45503333 | 0.48050857 |

- Seed: `0`
- Features: `user_history_count`, `user_history_long_view_rate`, `user_history_click_rate`, `candidate_video_seen_count`, `candidate_author_seen_count`, `candidate_author_long_view_rate`, `time_since_user_last_action`
- Training settings: `lr=0.001`, `l2=1e-6`, `batch_size=8192`, `epochs=40`, `patience=4`
- Training time: not recorded end-to-end; the script only emitted per-epoch timing
- Checkpoint: `results_by_track/history/checkpoints/history__seed0__best.npz`
- Git commit: `bb90e964d9d1109e0064e2491f2df5aeba196823`
- Status: preliminary seed-0 result; regression versus both BPR references; not statistically significant
- Machine-readable result: `results_by_track/history/metrics/history_aggregate_seed0.json`

## Experiment 6: History aggregate scaling correction (seed 0)

本次只修正數值尺度：count features 使用 `log1p`；time gap 先由毫秒轉為小時，再使用 `log1p(max(gap, 0))`。沒有新增 feature、修改模型、split 或 evaluation。Validation primary 用於選擇最佳 epoch，test 僅作最終報告。

| Method | Best epoch | Valid GAUC | Valid nDCG@5 | Valid primary | Gain vs original history | Gain vs BPR n_neg=1 | Test GAUC | Test nDCG@5 | Test primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| History linear logistic (scaled) | 2 | 0.5132645 | 0.47218773 | 0.4927261 | -0.00170373 | -0.1093739 | 0.50598687 | 0.45431086 | 0.48014885 |

- Seed: `0`
- Training settings: `lr=0.001`, `l2=1e-6`, `batch_size=8192`, `epochs=40`, `patience=4`
- Features: same seven aggregate history features, fixed feature order
- Negative time gaps: train `0`, valid `0`, test `1`; negative values were clamped to zero without deleting rows
- Training time: not recorded end-to-end by the script
- Checkpoint: `results_by_track/history/checkpoints/history__seed0__best.npz`
- Git commit: `bb90e964d9d1109e0064e2491f2df5aeba196823`
- Classification: no meaningful improvement versus the original history run; still a regression versus BPR. This is seed-0 evidence only.
- Machine-readable result: `results_by_track/history/metrics/history_aggregate_scaled_seed0.json`

## Experiment 7: History FM pointwise model (seed 0)

本次只將 scaled history 的 linear logistic model 替換為 history 專用 FM；7 個 numeric history features、5 個 categorical fields、dataset、split、evaluation 與訓練設定固定不變。Validation primary 用於選擇最佳 epoch，test 僅作報告。

| Track | Model | Loss | Best epoch | Valid GAUC | Valid nDCG@5 | Valid primary | Test GAUC | Test nDCG@5 | Test primary |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| history | FM | pointwise binary logistic | 3 | 0.66578263 | 0.53521466 | 0.6004987 | 0.65968186 | 0.5285109 | 0.5940964 |

- Seed: `0`; `k=16`; `lr=0.001`; `l2=1e-6`; `batch_size=8192`; `epochs=40`; `patience=4`
- Numeric features: the existing seven scaled history features, unchanged and in the original order
- Categorical fields: `user_id`, `video_id`, `author_id`, `tab`, `duration_bucket`
- Gain vs scaled linear history: `+0.1077726` validation primary
- Gain vs Pure BPR n_neg=1: `-0.0016013` validation primary
- Gain vs Random BPR n_neg=3: `-0.0019013` validation primary
- Training time: not recorded end-to-end by the current script
- Checkpoint: `results_by_track/history/checkpoints/history_fm__seed0__best.npz`
- Git commit: `bb90e964d9d1109e0064e2491f2df5aeba196823`
- Classification: weak candidate improvement versus scaled linear history, but no meaningful improvement versus BPR. Seed-0 evidence only; not statistically significant.
- Machine-readable result: `results_by_track/history/metrics/history_fm_seed0.json`

## Experiment 8: History FM Group A ablation (seed 0)

H0 與 H1 使用完全相同的 dataset、split、FM、pointwise binary logistic loss、seed 與訓練設定；唯一差異是 H1 額外加入 `user_behavior` group。結果是 preliminary ablation，不代表統計顯著。

| Config | Feature groups | Best epoch | Valid GAUC | Valid nDCG@5 | Valid primary | Test GAUC | Test nDCG@5 | Test primary |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | `base` | 3 | 0.66578263 | 0.53521466 | 0.60049870 | 0.65968186 | 0.52851090 | 0.59409640 |
| H1 | `base,user_behavior` | 3 | 0.66576385 | 0.53516700 | 0.60046540 | 0.65978634 | 0.52855220 | 0.59416926 |

- H1 gain vs H0 validation primary: `-0.00003330`
- H1 validation GAUC and nDCG@5 both slightly decreased.
- Test GAUC, nDCG@5 and primary slightly increased, so validation/test directions are not fully一致。
- Both runs had no NaN/Inf in feature diagnostics and used validation primary for checkpoint selection; test was not used for epoch selection.
- Training time: not recorded end-to-end by the current script
- H0 checkpoint: `results_by_track/history/checkpoints/history_fm__base__seed0__best.npz`
- H1 checkpoint: `results_by_track/history/checkpoints/history_fm__base_user_behavior__seed0__best.npz`
- Classification: no meaningful improvement; H1 should not replace H0 based on this single seed.
- Machine-readable results: `results_by_track/history/metrics/history_fm_ablation_base_user_behavior_seed0.json`

## Experiment 9: History FM video_history ablation (seed 0)

本次只將 H0 的 feature group 從 `base` 改為 `base,video_history`；model、pointwise binary logistic loss、seed、split、evaluation 與訓練設定固定不變。結果是 preliminary ablation，不代表統計顯著。

| Config | Feature groups | Best epoch | Valid GAUC | Valid nDCG@5 | Valid primary | Test GAUC | Test nDCG@5 | Test primary |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | `base` | 3 | 0.66578263 | 0.53521466 | 0.60049870 | 0.65968186 | 0.52851090 | 0.59409640 |
| H2 | `base,video_history` | 3 | 0.66555333 | 0.53509360 | 0.60032344 | 0.65962636 | 0.52809130 | 0.59385884 |

- H2 gain vs H0 validation primary: `-0.00017526`
- Validation GAUC and nDCG@5 both decreased; test GAUC, nDCG@5 and primary also decreased.
- No NaN/Inf was reported in feature diagnostics, and test was not used for epoch selection.
- Training time: not recorded end-to-end by the current script
- Checkpoint: `results_by_track/history/checkpoints/history_fm__base_video_history__seed0__best.npz`
- Git commit: `bb90e964d9d1109e0064e2491f2df5aeba196823`
- Classification: no meaningful improvement; video_history should not replace H0 based on this seed-0 ablation.
- Machine-readable result: `results_by_track/history/metrics/history_fm_video_history_seed0.json`

## Experiment 9: History FM video_history ablation (seed 0)

本次只將 H0 的 feature group 從 `base` 改為 `base,video_history`；model、pointwise binary logistic loss、seed、split、evaluation 與訓練設定固定不變。結果是 preliminary ablation，不代表統計顯著。

| Config | Feature groups | Best epoch | Valid GAUC | Valid nDCG@5 | Valid primary | Test GAUC | Test nDCG@5 | Test primary |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | `base` | 3 | 0.66578263 | 0.53521466 | 0.60049870 | 0.65968186 | 0.52851090 | 0.59409640 |
| H2 | `base,video_history` | 3 | 0.66555333 | 0.53509360 | 0.60032344 | 0.65962636 | 0.52809130 | 0.59385884 |

- H2 gain vs H0 validation primary: `-0.00017526`
- Validation GAUC and nDCG@5 both decreased; test GAUC, nDCG@5 and primary also decreased.
- No NaN/Inf was reported in feature diagnostics, and test was not used for epoch selection.
- Training time: not recorded end-to-end by the current script
- Checkpoint: `results_by_track/history/checkpoints/history_fm__base_video_history__seed0__best.npz`
- Git commit: `bb90e964d9d1109e0064e2491f2df5aeba196823`
- Classification: no meaningful improvement; video_history should not replace H0 based on this seed-0 ablation.
- Machine-readable result: `results_by_track/history/metrics/history_fm_video_history_seed0.json`
