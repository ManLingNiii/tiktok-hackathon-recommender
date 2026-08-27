# Team Contribution Guide

這個 branch 是可供組員延伸的基線。所有新訓練模式都必須以可插拔 module 加入，不要複製整份 trainer。

## 新增模式

1. 在 `agent/modules/` 新增一個小而單一職責的模組。
2. 實作 `validate(ctx)`，拒絕非 `long_view` 或非 `validation_only` context。
3. 在 `headroom_registry.json` 加入 allowlist。
4. 在 `experiment_specs.json` 加入固定 command 與 hypothesis。
5. 透過 `validation_experiment_runner.py` 執行。
6. 提交 metrics、失敗原因與是否改善 baseline 的說明。

## 不可做的事

- 修改 `kuairand-starter-kit/evaluate.py`。
- 在開發 loop 中讀取 test/hidden test。
- 把 validation label 帶入未來 history feature。
- 直接執行 Gemini 回傳的任意 shell command。
- 提交資料集、API key、`.env` 或大型生成檔案。

## 建議 commit 內容

每個 commit 只包含一個 headroom 或一個安全修正，並附：

```text
hypothesis
command
validation GAUC / nDCG@5 / primary
baseline delta
failure/recovery
```
