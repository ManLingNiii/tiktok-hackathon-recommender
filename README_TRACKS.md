# KuaiRand Pure Track 協作說明

本 repo 是在 `/Users/manling/Documents/kuairand-starter-kit` 之上進行四個 track 實驗的協作層。每個人 clone 後，應保留相同的官方資料、split、評估程式與 baseline；只修改自己負責的 track 檔案。

## 1. Clone 後的目錄結構

```text
kuairand-starter-kit/
├── KuaiRand-Pure/data/              # 共用：同一份 pure dataset
├── src/
│   ├── data.py                      # 共用：資料載入、固定 split、feature encoding
│   ├── evaluate.py                  # 共用且不可修改：唯一 metric authority
│   ├── submit.py                    # 共用：submission 對齊與檢查
│   ├── baseline.py                  # 共用：官方 FM / popularity / random baseline
│   ├── baseline_bpr.py              # BPR track：一個負例
│   └── baseline_bpr_3neg.py         # BPR track：三個負例
├── configs/
│   └── common.yaml                 # 共同規則：pure、seed=0、split v1、primary
├── tracks/
│   ├── bpr/                        # BPR 專屬入口與文件（WIP）
│   └── _template/                  # 未來 track 的格式模板
├── results/
│   ├── baseline_scores.json         # 現有：baseline 參考分數
│   ├── experiments.md              # 現有：實驗摘要
│   └── experiments.md              # 所有實驗的共同摘要與紀錄格式
├── results_by_track/
│   ├── README.md
│   └── bpr/                        # BPR 詳細 logs/checkpoints/metrics/top3
└── README_TRACKS.md
```

`kuairand_modular/` 不在 starter kit 內；它是另一個 workspace 的模組化參考實作。如果只使用 starter kit，可以忽略它。

執行時由 `--data_dir` 指向同一份資料：

```bash
cd /Users/manling/Documents/kuairand-starter-kit
PYTHONPATH=src python src/baseline.py --model fm --seed 0 --data_dir ./KuaiRand-Pure/data
```

## 2. BPR track 專屬檔案

| 路徑 | 用途 |
|---|---|
| `src/baseline_bpr.py` | BPR-FM：同一 user 的 positive/negative pairwise training |
| `src/baseline_bpr_3neg.py` | BPR-FM：每個 positive 搭配多個 negative |
| `tracks/bpr/train_bpr.py` | BPR 原始入口的 wrapper，不改變原始邏輯 |
| `tracks/bpr/train_bpr_3neg.py` | 3-negative 原始入口的 wrapper；目前仍是 WIP |
| `tracks/bpr/model.py` | 只 re-export 原始 `FM` 類別，不抽換模型邏輯 |
| `tracks/bpr/README.md` | BPR 狀態、執行方式與紀錄規則 |
| `results_by_track/bpr/` | BPR 的 logs、metrics、checkpoints、Top 3 manifest |

BPR pair 必須在同一個 user 內建立。訓練只能使用 train split；validation 用來選 checkpoint，不能用 test 反覆調參。

## 3. 必須共用的檔案

| 路徑 | 共用內容與規則 |
|---|---|
| `KuaiRand-Pure/data/` | 同一份 pure dataset；不要重切、重排或混入其他資料 |
| `src/data.py` | 官方資料載入與 train/valid/test 日期範圍 |
| `src/evaluate.py` | GAUC、nDCG@5、primary 的唯一實作；不要修改 |
| `src/submit.py` | row 對齊、submission 格式與 `--check` |
| `src/baseline.py` | 共同 baseline；所有 gain 都相對於同一 baseline |
| `results/baseline_scores.json` | baseline 參考結果 |
| `results/registry.json` | 需新增：composition agent 的候選模型清單 |

若需額外 feature、history 或 auxiliary label，請放在自己的 track 檔案中，不要直接改 shared loader。若確實要改共用檔案，先在 `results/experiments.md` 說明並通知所有 track。

## 4. 固定實驗規則

四個 track 固定 `seed=0`、相同 split、相同 `src/evaluate.py`、相同 primary（`(GAUC + nDCG@5) / 2`）與相同 baseline。

每組實驗都要記錄：

```text
track, method, hyperparameters, seed, checkpoint_path,
valid_GAUC, valid_nDCG@5, valid_primary, training_time, git_commit
```

```text
gain = valid_primary - baseline_valid_primary
```

每個 track 交付 validation primary 最好的 Top 3 checkpoint；不要用 test 分數排序或調參。

## 5. Results 與 checkpoint registry

```text
results_by_track/bpr/
├── logs/
├── checkpoints/
├── metrics/
└── top3.json
```

未來建立 `results/registry.json` 時，使用 repo-relative checkpoint path，避免不同 clone 的絕對路徑失效。每筆候選至少包含：`track`、`method`、`hyperparameters`、`seed`、`checkpoint_path`、`valid_GAUC`、`valid_nDCG@5`、`valid_primary`、`gain`、`training_time_sec`、`git_commit`。目前 BPR 沒有正式 Top 3，`results_by_track/bpr/top3.json` 保持空陣列。

## 6. Future track template

```bash
cp -r tracks/_template tracks/<your_track_name>
```

新 track 應共用 `configs/common.yaml`、`src/data.py` 與 `src/evaluate.py`，不要修改 `tracks/bpr/`。詳細結果放在 `results_by_track/<your_track_name>/`。

## 7. 完整 workflow

```text
共同 baseline(seed=0) → 各 track HPO(seed=0) → 各 track Top 3
→ 更新 results/registry.json → agent 組合候選 checkpoint
→ 依 validation primary 選最佳 composition
→ 對少數候選跑 seed 0,1,2,3 → 比較平均值與標準差
→ 最後才用 test 確認與 promote
```

本地檢查 submission：

```bash
PYTHONPATH=src python src/submit.py --check submission.csv \
  --data_dir ./KuaiRand-Pure/data --split test
```

不要把 test label 寫入訓練、HPO、Top 3 或 composition 選擇流程；test 只在最後確認或正式報告時使用。
