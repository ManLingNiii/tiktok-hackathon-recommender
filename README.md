# KuaiRand TikTok Hackathon Agent

這個專案是 KuaiRand-Pure 的 ranking prototype，預測 `long_view`，並用官方 evaluator 計算 GAUC、nDCG@5 與 Primary。

## 比賽限制

- 開發與 AutoML 只使用 `train` / `valid`。
- `test` / hidden-test 不得用於模型選擇、調參或 agent loop。
- 不修改官方 `data.py`、`evaluate.py` 或 submission contract。
- Agent 只能執行 allowlist 中的 reviewed experiment。
- Planner 不能產生或執行任意 shell command 或 Python 程式。
- 只有 validation primary 改善時才保存 family-best checkpoint；confirmation gate 用來降低 overfitting。
- API key、原始資料、實驗 runs 與 submission CSV 不提交到 GitHub；唯一例外是下方列出的 frozen Pure checkpoint，使用 Git LFS 版本化。

## 主要目錄

- `agent/autonomous_agent.py`：受控 AutoML loop。
- `agent/config_generator.py`：task-based、bounded、deterministic 參數組合器；每輪只產生一個 recipe。
- `agent/configs/search_space.json`：固定候選與合法參數 axes。
- `agent/experiment_specs.json`：experiment allowlist。
- `agent/validation_experiment_runner.py`：validation-only runner 與 audit。
- `agent/checkpoint_manager.py`：各 family 最佳權重管理。
- `agent/research_diagnosis.py`：從 validation／confirmation 指標產生原因分析與下一步策略。
- `agent/modules/composition.py`：五個 family 全部 frozen 的 composition contract；composition code 固定 `11111`。
- `agent/modules/context_composition.py`：使用五個 frozen family prediction、train-only user/video/context 特徵與正則化 BPR gate 的 context-aware composition。
- `agent/modules/`：BPR、Listwise、History、Multi-task、CWM 元件。
- `kuairand-starter-kit/`：官方 data、evaluate 與 submit checker，只讀使用。
- `submission_ready/`：frozen manifest、prediction adapter、generator 與 local checker。
- `runs/`：本地實驗紀錄，不追蹤。
- `outputs/pure/`：Task 1 與目前 prepared composition 所需的 frozen Pure checkpoint，使用 Git LFS 追蹤。
- `others/`：舊版權重、資料壓縮檔與整合 bundle 封存，不提交。

## 建立環境

```powershell
conda env create -f environment.yml
conda activate tiktok
cd <repository-root>
```

將主辦方提供的 KuaiRand-Pure 資料放在 `kuairand-starter-kit\KuaiRand-Pure\data\`。`full-system` branch 依需求使用 Git LFS 包含這些 CSV；其他 branch 可維持不包含資料集的輕量版本。

Clone `full-system` 後需先安裝並啟用 Git LFS，再執行 `git lfs pull` 取得資料與 frozen checkpoint。API key、runs、submission CSV 與 `others/` 封存內容不會上傳。

## 單一 validation 實驗

```powershell
python -u agent\validation_experiment_runner.py baseline_fm
python -u agent\validation_experiment_runner.py bpr_fm
python -u agent\validation_experiment_runner.py listwise_fm
python -u agent\validation_experiment_runner.py history_fm
python -u agent\validation_experiment_runner.py multitask_fm
python -u agent\validation_experiment_runner.py cwm_fm
```

結果會寫入 `runs\pure\`，包含 JSON、log、leaderboard、planner decision、checkpoint metadata 與 code-diff audit。

## 受控 AutoML（50 次 task workflow）

```powershell
python -u agent\autonomous_agent.py --max-iterations 50 --use-pretrained
```

Planner 依序執行四個 task：weight learning 12、additive/interaction learning 16、DNN composition 13、multi-seed/confirmation 9，合計最多 50 次。每輪只從 `search_space.json` task template 產生一個新 recipe，根據上一輪的 GAUC、nDCG@5、Primary、loss、feature variance、prediction correlation 與 recovery event 提出 hypothesis。`>=0.002` 改善會留在同一 task；低於門檻連續兩次或達 task 上限才切換 task。狀態保存在 `runs\pure\task_workflow_state.json`。

若所有受控候選耗盡，agent 會安全停止，避免重複實驗造成 validation overfitting。使用 Gemini 時，API key 只放環境變數：

Composition config 也由同一個 generator 管理，但五個 family 永遠全部載入，固定使用 `11111`；只訓練 final composition layer，不會把 loss gradient 傳回 family checkpoint。第一版是 trainable nonnegative linear fusion + bias，loss 固定為 `0.6 * within-user listwise + 0.4 * same-user BPR`，target 固定為 `long_view`。每輪會將 GAUC、nDCG@5、Primary、loss 與 confirmation drift 轉成 `research_analysis`，供下一輪 hypothesis 使用。

Task 2 只允許指定的 pure feature 或單一 prediction interaction；Task 3 才使用 selected pure data 與低容量 `small_mlp` DNN composition。Task 4 用 seed 0/1/2 與 confirmation 統計決定是否保留。所有 pure feature 統計只由 train 建立，validation 不使用 label 回寫。

```powershell
$env:PLANNER_BACKEND = "gemini"
$env:GEMINI_API_KEY = "<your-key>"
python -u agent\autonomous_agent.py --max-iterations 50 --use-pretrained
```

## 最佳模型 prediction 與 checker

`submission_ready\composition_manifest.json` 是目前 frozen composition 模型來源。舊的 listwise candidate 已移至 `others\legacy_candidates\`，不再作為 agent 或 submission baseline。`prediction_adapter.py` 會驗證 manifest 狀態、checkpoint 路徑、權重總和與檔案存在性。

先做 validation preview：

```powershell
python submission_ready\generate_submission.py --split valid --output submission_ready\validation_preview.csv
python submission_ready\local_checker.py submission_ready\validation_preview.csv --split valid --score
```

使用官方 checker 交叉驗證：

```powershell
$env:PYTHONIOENCODING = "utf-8"
python kuairand-starter-kit\submit.py --check --data_dir kuairand-starter-kit\KuaiRand-Pure\data --split valid submission_ready\validation_preview.csv
python kuairand-starter-kit\submit.py --score --data_dir kuairand-starter-kit\KuaiRand-Pure\data --split valid submission_ready\validation_preview.csv
```

正式 test submission 只能在 validation 與模型選擇完成後，由人工確認；test 結果不可回饋 AutoML。本 repository preparation 不執行 submit。

## GitHub push 前檢查

```powershell
python -m py_compile agent\*.py agent\modules\*.py
python -m json.tool agent\configs\search_space.json > $null
git diff --check
git status --short
git diff --cached --name-only
```

確認 staged files 不包含 `runs/`、非 frozen 的 `outputs/`、`*.csv`、原始資料、`.tar.gz`、`.env`、API key、token、private key 或機器專屬絕對路徑。唯一允許的 binary 是 `outputs/pure/` 下由 `.gitattributes` 交給 Git LFS 管理的 frozen `.npz`。最後的 `git add`、commit 與 push 由使用者人工執行。
