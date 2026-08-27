# KuaiRand TikTok Hackathon Agent

這個專案是 KuaiRand-Pure 的 ranking prototype，目標是預測 `long_view`，並以使用者內曝光排序評估模型。它提供官方 FM baseline、validation-only 實驗 runner，以及四個可插拔的 headroom 方向。

## 任務與規則

- Dataset：`KuaiRand-Pure`
- Target：`long_view`
- Metrics：`GAUC`、`nDCG@5`
- Primary：`mean(GAUC, nDCG@5)`
- 開發資料：train / validation only
- test / hidden test：禁止用於模型選擇、調參或 Gemini agent loop
- 官方 `evaluate.py` 不可修改
- 最終 submission 格式：`row_id,user_id,video_id,score`

規則設定位於 `agent/rules.json` 與 `agent/headroom_registry.json`。舊的 `agent/run_agent.py` 已停用，不能作為開發入口。

## 目錄

```text
agent/
  validation_only.py              # 只載入 train/validation
  validation_experiment_runner.py # allowlist + runner + leaderboard
  formal_trainer.py               # 共用 headroom trainer
  rich_data.py                    # auxiliary labels + train-only history
  experiment_specs.json           # 可執行實驗白名單
  headroom_registry.json          # 安全規則
  experiments/bpr_fm.py           # 正式 BPR FM
  modules/                        # loss、history、multi-task、CWM 元件

kuairand-starter-kit/
  baseline.py                     # 官方 baseline（只讀，不修改）
  data.py                         # 官方 encoder
  evaluate.py                     # 官方 evaluator（只讀，不修改）
  submit.py                       # submission checker

runs/                             # 實驗 JSON、log、leaderboard
outputs/                          # checkpoint 與 submission artifacts
```

## 環境與基本執行

不要上傳或複製本機 conda environment。請用 `environment.yml` 在新電腦重建：

```powershell
conda env create -f environment.yml
conda activate tiktok
```

接著將官方 KuaiRand-Pure Starter Kit 資料放到：

```text
kuairand-starter-kit\KuaiRand-Pure\data\
```

原始資料不放入 GitHub；請依主辦方提供的下載方式取得。

```powershell
conda activate tiktok
cd D:\tiktok
```

Validation-only baseline：

```powershell
python -u agent\validation_experiment_runner.py baseline_fm
```

BPR：

```powershell
python -u agent\validation_experiment_runner.py bpr_fm
```

Other registered modes：

```powershell
python -u agent\validation_experiment_runner.py listwise_fm
python -u agent\validation_experiment_runner.py history_fm
python -u agent\validation_experiment_runner.py multitask_fm
python -u agent\validation_experiment_runner.py cwm_fm
```

每個 runner 都會將 stdout/stderr 保存到 `runs/experiment_###_<name>.log`，並寫入對應 JSON。`runs/validation_leaderboard.json` 只比較 validation primary。

## 系統 workflow

```text
Gemini planner
    │ 只提出 allowlisted experiment
    ▼
Skill Maintainer / Governor
    │ 檢查 target、split、path、module
    ▼
validation_experiment_runner.py
    │ 建立 subprocess，禁止 test，記錄錯誤
    ▼
formal_trainer.py 或 reviewed experiment entrypoint
    │ train-only data / validation-only evaluation
    ▼
FM + headroom module
    │ gradient / feature transform / auxiliary objective
    ▼
GAUC、nDCG@5、primary
    │
    ▼
leaderboard + checkpoint + experiment log
```

正式模型選擇只依 validation primary。若新模型沒有超過 baseline 加上 `epsilon=0.002`，就保留實驗結果但不 promote，baseline 仍是候選模型。

## Headroom modules

1. **BPR**：同一訓練 batch 中建立 positive/negative pairs，使用 pairwise gradient。
2. **Listwise**：按 user exposure group 建立 softmax objective。
3. **History**：以時間順序建立 prior user、user-tab、user-author counts；validation 只能使用 train 結束時的歷史。
4. **Multi-task**：以 `long_view` 為主任務，並使用 click、like、follow、comment、forward 作 auxiliary signals。
5. **CWM**：將 `play_time_ms / duration_ms` normalize 至 0 到 1，對 censored observations 使用單側懲罰。

## 目前實驗結果

| Experiment | GAUC | nDCG@5 | Primary | Status |
|---|---:|---:|---:|---|
| Baseline FM | 0.6650 | 0.5342 | 0.6015 | success |
| BPR FM | 0.6673 | 0.5359 | 0.6026 | success |
| Listwise FM | 0.4942 | 0.4634 | 0.4800 | success, needs review |
| History FM | 0.6660 | 0.5353 | 0.6008 | success |
| Multi-task FM | 0.6611 | 0.5339 | 0.5987 | success |
| CWM FM | 0.6637 | 0.5345 | 0.5999 | success |

BPR 是目前 validation 最佳，但相對 baseline 只有 `+0.00115`，低於 `+0.002` promote 門檻，因此不能取代 baseline。

## 合規與安全

- Gemini 不可任意修改 repository，只能選擇 `experiment_specs.json` 中的實驗。
- 不要把 API key 寫入 repository；使用環境變數 `GEMINI_API_KEY`。
- 不要把 `KuaiRand-Pure` 原始資料、API key 或大型 checkpoint 提交到 GitHub。
- 最終 submission/test 流程只能在 validation 實驗與模型選擇完成後，由人工確認執行。
- `runs/` 中可能包含歷史失敗實驗；這些是 debugging evidence，不應當作模型成績。

## 已知限制與下一步

- Listwise 結果異常低，需檢查 gradient scaling、group loss 與 early stopping。
- Multi-task 目前是 shared FM representation 的加權 auxiliary gradient，不是完全獨立的多 head neural architecture。
- CWM 的 censor label 仍是近似定義，需要依資料語意確認。
- 最終最佳 checkpoint 尚未完成公開 test 上的人工 submission 流程。
- Gemini autonomous planner 尚未接入 runner；接入前必須維持 registry fail-closed。
