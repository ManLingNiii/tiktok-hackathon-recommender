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
