# Submission-ready package（尚未提交）

這個資料夾是目前 validation 選出的 frozen submission 設定，不包含 submission CSV，也沒有執行公開 test 評分。

目前候選是已匯入本專案並在 local 重現的 `listwise_ensemble`：

- validation Primary：`0.6040996313`
- GAUC：`0.6703423858`
- nDCG@5：`0.5378568172`

四個 ensemble member 與權重記錄在 `manifest.json`。`prediction_adapter.py` 將這份 manifest 視為唯一模型來源，會檢查 frozen 狀態、validation-only 契約、權重總和、checkpoint 路徑與檔案存在性，再依清單產生 prediction。產生正式 submission 前，必須由人工確認：固定這份 manifest、依官方 test row order 產生 `row_id,user_id,video_id,score`，再使用官方 checker 做最後格式檢查。公開 test 只可在此人工 finalization 階段使用，不能回饋 agent 的研究迴圈。

目前沒有 submission CSV，沒有 submit，也沒有 test score。

## Local preview/check

只做 validation preview：

```powershell
python submission_ready\generate_submission.py --split valid --output submission_ready\validation_preview.csv
python submission_ready\local_checker.py submission_ready\validation_preview.csv --split valid --score
```

正式 test submission 的 generator/checker 路徑需要人工 finalization 時額外指定 `--allow-final-test`；這次沒有執行該路徑，也不會把 test 結果回饋 agent。
