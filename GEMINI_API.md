# Google Gemini API 使用教學

本專案的 Gemini 只應作為「實驗規劃器」，不能繞過 validation-only runner，也不能讀取 hidden test。

## 1. 安裝 SDK

在 PowerShell：

```powershell
conda activate tiktok
cd D:\tiktok
python -m pip install -U google-genai
```

## 2. 設定 API key

只在目前 terminal 設定，離開 terminal 後會失效：

```powershell
$env:GEMINI_API_KEY = "你的_API_KEY"
```

永久環境變數不建議用於共用電腦；無論哪種方式，都不能把 key 寫入 Python、README 或 Git。

檢查是否存在但不要印出 key：

```powershell
if ($env:GEMINI_API_KEY) { "API_KEY_SET" } else { "API_KEY_MISSING" }
```

若 key 曾貼到聊天、截圖或 Git，請立刻在 Google AI Studio 撤銷並重建。

## 3. 最小連線測試

建立一次性測試，不要把 key 寫進檔案：

```powershell
python -c "from google import genai; import os; c=genai.Client(api_key=os.environ['GEMINI_API_KEY']); r=c.models.generate_content(model='gemini-3.7-flash', contents='Reply with API_OK only.'); print(r.text)"
```

成功時應看到：

```text
API_OK
```

## 4. 在 agent 中的安全呼叫方式

請使用 `client.chats.create()` 與 `chat.send_message()`。這是 SDK 對需要後續工具／function calling 的推薦模式；本專案的 planner 實作位於 `agent/gemini_planner.py`。

Gemini 的輸出只能被解析成 experiment plan，再送入 registry：

```python
import json, os
from google import genai
from agent.experiment_registry import validate_plan

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
chat = client.chats.create(model="gemini-3.7-flash")
prompt = """
Choose exactly one experiment from: listwise_fm, history_fm, multitask_fm, cwm_fm.
Return JSON only with keys: module, experiment, splits, files.
Use train and valid only. Target is long_view.
"""
response = chat.send_message(prompt)
plan = json.loads(response.text)
validate_plan(plan)  # fail closed before any subprocess is started
```

接著仍由固定 runner 執行：

```powershell
python -u agent\validation_experiment_runner.py listwise_fm
```

不要讓 Gemini 回傳 shell command 後直接用 `Invoke-Expression` 或 `subprocess(..., shell=True)` 執行。實驗名稱必須由本地 `experiment_specs.json` 解析，不能由模型新增。

## 5. 團隊協作規則

- 不共用 API key；每位成員使用自己的 key。
- 不把 `.env`、key、資料集或大型 checkpoint push 到 GitHub。
- Gemini 只讀取 curated leaderboard 與 allowlisted specs。
- 模型選擇只依 validation primary；最終公開 test 流程由人工執行。
- 發生 401/403 時重建 key；發生 429 時降低頻率或改用人工執行，不要在程式中硬編 key。
