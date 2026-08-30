# BPR Track

**狀態：開發中（Work in Progress）。** 本 track 目前由 BPR 負責人維護，方法與結果仍在調整，可能會變動；目前尚未 finalized，也不是 production-ready。

## 檔案與相互關係

- `src/baseline_bpr.py` 是原始實作，也是 `n_neg=1` 的 random BPR 版本，保留為 source of truth。
- `src/baseline_bpr_3neg.py` 支援 `n_neg=1`、`n_neg=3`、`n_neg=5` 的 random same-user negative sampling，目前仍是 WIP。
- `tracks/bpr/train_bpr.py` 與 `train_bpr_3neg.py` 是 wrapper，提供清楚的 track 執行入口，不改變原始訓練邏輯。
- `tracks/bpr/model.py` 只重新匯出 `FM`，沒有搬移或重寫模型邏輯。
- `src/baseline_bpr_semihard.py` 與 `tracks/bpr/train_bpr_semihard.py` 是獨立的 semi-hard 實驗，不修改既有 BPR 實作。
- `src/baseline_bpr_hybrid.py` 與 `tracks/bpr/train_bpr_hybrid.py` 是獨立的 BPR + pointwise hybrid loss 實驗。
- `src/checkpoint_utils.py` 提供 random BPR checkpoint export；semi-hard 與 hybrid 的演算法邏輯不受影響。

共用 import 為 `src/data.py`（資料載入與 encoding）及 `src/evaluate.py`（評估指標）。BPR 不依賴 `src/baseline.py`。

## Random negative sampling

Random sampling 只使用 train split 中、與 positive 屬於同一 user 的 negative rows。若某個 user 的 negative 數量少於 `n_neg`，會使用 `replace=True`。多個 negative 會 flatten 成多組 pair，維持既有的 `m.step(X_pos, X_neg)` 介面；FM/BPR model、loss、gradient 與 evaluator 不變。

## Semi-hard negative sampling

第一版 semi-hard 採用 static sampling：先進行一次 random same-user warm-up（預設 `warmup_epochs=1`），再由 warm-up 後的模型替每位 user 的 train negatives 打分。對每個 positive，從該 user 分數最高的 `semi_hard_fraction` 候選區間中隨機抽樣（預設為最高分的 30%），而不是永遠選擇單一最高分 negative。產生的 pairs 會在剩餘 epochs 中固定使用。這只是實驗版本，不能視為顯著改善的證據。

## 共用實驗規則

使用 `KuaiRand-Pure/data/`。固定 split 為 train `20220408–20220421`、valid `20220422–20220428`、test `20220429–20220508`。只使用 train 訓練，依 validation primary 選 checkpoint，test 只保留給最後確認。

第一階段使用 `seed=0`、共用 `src/evaluate.py`，primary 為 `mean(GAUC, nDCG@5)`，並與共用 baseline 比較 gain。

## 執行方式

在 repository 根目錄執行：

```bash
PYTHONPATH=src python tracks/bpr/train_bpr.py --data_dir ./KuaiRand-Pure/data --model fm --seed 0
PYTHONPATH=src python tracks/bpr/train_bpr_3neg.py --data_dir ./KuaiRand-Pure/data --model fm --n_neg 1 --seed 0
PYTHONPATH=src python tracks/bpr/train_bpr_3neg.py --data_dir ./KuaiRand-Pure/data --model fm --n_neg 3 --seed 0
PYTHONPATH=src python tracks/bpr/train_bpr_3neg.py --data_dir ./KuaiRand-Pure/data --model fm --n_neg 5 --seed 0
PYTHONPATH=src python tracks/bpr/train_bpr_semihard.py --data_dir ./KuaiRand-Pure/data --model fm --n_neg 1 --warmup_epochs 1 --semi_hard_fraction 0.3 --seed 0
```

上述指令透過 wrapper 使用既有 source 檔案；不代表 3-negative 或其他實驗已 finalized。

## Hybrid loss（獨立 WIP 實驗）

純 BPR 只使用 pairwise ranking loss。Hybrid 實驗在 positive/negative rows 上加入 pointwise logistic loss：

```text
L_total = alpha * L_pointwise + (1 - alpha) * L_BPR
```

`alpha=0` 等同純 BPR；alpha 越大，pointwise supervision 的比重越高。每個 batch 建立 `X_point = concat(X_pos, X_neg)`，label 分別為 `1` 與 `0`，合併兩種 gradient 後只執行一次 optimizer update，L2 只加入一次。第一輪比較 alpha `0.25`、`0.5`、`0.75`，固定 `n_neg=1`、`seed=0` 與其他設定。seed=0 結果只能作為初步比較，不能稱為統計顯著。

```bash
PYTHONPATH=src python tracks/bpr/train_bpr_hybrid.py --data_dir ./KuaiRand-Pure/data --model fm --n_neg 1 --alpha 0.5 --seed 0 --k 16 --lr 0.001 --l2 1e-6 --batch_size 8192 --epochs 40 --patience 4
```

## Checkpoint

Random BPR 會保存 validation primary 最高的 epoch 模型，而不是最後一個 epoch。預設位置為 `results_by_track/bpr/checkpoints/`，也可以使用 `--checkpoint_dir` 指定其他位置。

檔名範例：

```text
bpr__nneg1__seed0__best.npz
bpr__nneg3__seed0__best.npz
```

不同 `n_neg` 設定不會互相覆蓋。NumPy checkpoint 包含 `V`、`W`、`b`、模型維度、訓練 hyperparameters、seed、best epoch、validation metrics、n_neg、method、dataset、split version 與 git commit。未來 agent 可以讀取這些 arrays 與 metadata 來重現 scoring。BPR 目前仍是 WIP。

```bash
PYTHONPATH=src python tracks/bpr/train_bpr.py --model fm --seed 0 --checkpoint_dir results_by_track/bpr/checkpoints/
PYTHONPATH=src python tracks/bpr/train_bpr_3neg.py --model fm --n_neg 3 --seed 0 --checkpoint_dir results_by_track/bpr/checkpoints/
```

其他成員可以閱讀與參考本 track，但不應直接修改 `tracks/bpr/`。
