# System Workflow

## 1. Human setup

```powershell
conda activate tiktok
cd <repository-root>
```

確認資料位於 `kuairand-starter-kit\KuaiRand-Pure\data`，並先執行 baseline。

## 2. Task-based experiment proposal

Planner 每輪只提出一個新的 composition recipe。六個 task 的 budget 為 5/12/10/8/6/9，總上限 50 次：

1. five-family weight learning
2. additive / interaction learning with pure data
3. DNN composition with selected pure data
4. multi-seed and confirmation retention

每個 hypothesis 必須引用上一輪的 metrics、loss、prediction correlation、feature variance 與 failure/recovery evidence。

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
7. Composition 永遠載入 bpr/listwise/history/multitask/cwm 五個 frozen family，row alignment 後做 user-level z-score；composition code 固定為 `11111`，不得用 bitmask 排除 family。
8. 第一版只訓練 composition layer 的 nonnegative weights 與 bias；target 固定 `long_view`，loss 固定 `0.6 * within-user listwise + 0.4 * same-user BPR`。family checkpoint 不接受 composition gradient。
9. pure feature 與 interaction 只能使用 task template 指定的欄位；統計只由 train 建立。

## 5. Task transition and decision gate

```text
improvement = primary_current - primary_task_best
```

`improvement >= 0.002` 時更新 task best、stagnant 歸零並留在目前 task；低於門檻則增加 stagnant。連續兩次低於門檻或達 task budget 才切換下一 task，且 task convergence 不會停止 global agent。候選 checkpoint 會先保存，最終仍須由 validation 與 confirmation 評估。所有實驗需記錄 hypothesis、evidence、change plan、family list、weights、features、loss、target、GAUC、nDCG@5、primary、improvement、stagnant、runtime、error、recovery、test_access 與 checkpoint。

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

## 7. Composition-first experiment selection

目前 agent 只從受控 composition recipe 或 Gemini 提出的合法 composition config
選擇下一個候選；五個 family 的 initial checkpoint 維持固定，不會被 composition
搜尋覆蓋。每個候選都必須經過 validation、confirmation 與 promotion gate。

```powershell
python -u agent\autonomous_agent.py --composition-only --max-iterations 8
```

composition layer 只能調整 family 組合、融合權重與 composition seed；不能修改
family checkpoint、pretrained 路徑或 family 訓練參數。

## 8. Final submission

只有在 validation 實驗與人工 review 完成後，才可以執行 Starter Kit 的 submission 流程：

```powershell
python submit.py --check <submission.csv>
```

公開 test 的使用只限最終 submission 自檢，不得回饋到 Gemini 的開發迴圈或模型選擇。
