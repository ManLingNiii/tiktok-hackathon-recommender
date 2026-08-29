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
- `agent/config_generator.py`：bounded、deterministic 參數組合器。
- `agent/configs/search_space.json`：固定候選與合法參數 axes。
- `agent/experiment_specs.json`：experiment allowlist。
- `agent/validation_experiment_runner.py`：validation-only runner 與 audit。
- `agent/checkpoint_manager.py`：各 family 最佳權重管理。
- `agent/modules/`：BPR、Listwise、History、Multi-task、CWM 元件。
- `kuairand-starter-kit/`：官方 data、evaluate 與 submit checker，只讀使用。
- `submission_ready/`：frozen manifest、prediction adapter、generator 與 local checker。
- `runs/`：本地實驗紀錄，不追蹤。
- `outputs/pure/`：frozen Pure 最佳 checkpoint，使用 Git LFS 追蹤。

## 建立環境

```powershell
conda env create -f environment.yml
conda activate tiktok
cd <repository-root>
```

將主辦方提供的 KuaiRand-Pure 資料放在 `kuairand-starter-kit\KuaiRand-Pure\data\`。原始資料不放入 GitHub。

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

## 受控 AutoML

```powershell
python -u agent\autonomous_agent.py --max-iterations 100 --use-pretrained
```

Planner 會根據 validation 結果選擇下一個 family 與 config。`config_generator.py` 只從 `search_space.json` 登記的有限 axes 產生 deterministic、bounded 組合，並排除已嘗試 config。各 family 的最佳權重位於 `outputs\pure\{experiment}_best.npz`，metadata 位於 `runs\pure\best_models.json`。Pure 的 frozen 最佳 checkpoint 已使用 Git LFS 納入 repository；`runs/` metadata 仍保留在本地，避免上傳大量實驗紀錄。

若所有受控候選耗盡，agent 會安全停止，避免重複實驗造成 validation overfitting。使用 Gemini 時，API key 只放環境變數：

```powershell
$env:PLANNER_BACKEND = "gemini"
$env:GEMINI_API_KEY = "<your-key>"
python -u agent\autonomous_agent.py --max-iterations 100 --use-pretrained
```

## 最佳模型 prediction 與 checker

`submission_ready\manifest.json` 是 frozen 模型來源。`prediction_adapter.py` 會驗證 manifest 狀態、checkpoint 路徑、權重總和與檔案存在性。

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
