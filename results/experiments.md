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
- Features: Same as the official FM baseline
- Embedding dimension: `k=16`
- Learning rate: `0.001`
- Batch size: `8192`
- Maximum epochs: `40`
- Early stopping patience: `4`
- Seed: 4

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
