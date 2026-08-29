# System Workflow

## 1. Human setup

```powershell
conda activate tiktok
cd <repository-root>
```

確認資料位於 `kuairand-starter-kit\KuaiRand-Pure\data`，並先執行 baseline。

## 2. Experiment proposal

Gemini 或組員提出一個小型假設，例如：

```json
{
  "module": "listwise_loss",
  "experiment": "listwise_fm",
  "splits": ["train", "valid"],
  "files": ["agent/formal_trainer.py"]
}
```

## 3. Governor gate

`experiment_registry.py` 與 `validation_experiment_runner.py` 必須拒絕：

- test / hidden test
- 修改官方 evaluator
- 未在 `headroom_registry.json` 的 module
- 未審核 command 或路徑

## 4. Training contract

所有正式 trainer 必須：

1. 使用同一份 `long_view` target。
2. 使用相同 GAUC、nDCG@5 evaluator。
3. 對 user exposure 做排序 objective。
4. 只從 train 建立 vocabulary、duration buckets 與 history。
5. validation 不回寫 label 或 feature state。
6. 在 early stopping 後保存 validation 最佳 checkpoint。

## 5. Decision gate

```text
primary_new >= primary_baseline + 0.002
```

才可進入 candidate；否則保留 log 並回到 baseline。所有實驗都需記錄 status、metrics、runtime、error 與 recovery events。

## 6. Gemini integration boundary

Gemini 只負責：

- 讀取 leaderboard
- 選擇下一個 allowlisted experiment
- 產生 hypothesis/config
- 呼叫 validation runner
- 解讀 validation 結果

Gemini 不負責：

- 直接讀取 test
- 修改 `evaluate.py`
- 修改安全規則
- 任意改寫整個 repository
- 決定是否使用公開 test 成績

## 7. Experiment selection ranking

`agent/experiment_selector.py` 使用 UCB1 對 `agent/configs/search_space.json`
中的候選實驗排序。它只讀取 validation leaderboard，未測試候選會保留探索優先權，
已測試候選則依 validation primary 與探索 bonus 排序。

```powershell
python -u agent\experiment_selector.py --top 10
```

這個 selector 只決定「下一個實驗候選順序」，不改變 GAUC/nDCG@5，也不取代
`validation_experiment_runner.py` 的安全檢查。最終 promote 仍必須通過 baseline、
epsilon 與 confirmation guard。

## 8. Final submission

只有在 validation 實驗與人工 review 完成後，才可以執行 Starter Kit 的 submission 流程：

```powershell
python submit.py --check <submission.csv>
```

公開 test 的使用只限最終 submission 自檢，不得回饋到 Gemini 的開發迴圈或模型選擇。
