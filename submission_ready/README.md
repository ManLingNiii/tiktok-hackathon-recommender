# Submission-ready package（尚未提交）

這個資料夾是目前 validation 選出的 frozen submission 設定，不包含 submission CSV，也沒有執行公開 test 評分。

目前舊 manifest 是前一版 composition search 的歷史結果；新版 agent 不會直接沿用它，必須完成四個 task 的 multi-seed confirmation 後才會產生新的保留 manifest：

- composition：固定使用 BPR、Listwise、History、Multi-task、CWM，composition code `11111`
- composition layer：trainable nonnegative weights/bias，後續可進入 additive/interaction 與小型 DNN task
- target/loss：`long_view`，`0.6 * within-user listwise + 0.4 * same-user BPR`

composition 的 family checkpoint 與權重記錄在 `composition_manifest.json`。`prediction_adapter.py` 將人工確認後的 manifest 視為唯一模型來源，會檢查 validation-only 契約、權重總和、checkpoint 路徑與檔案存在性，再依清單產生 prediction。產生正式 submission 前，必須由人工確認 manifest、依官方 test row order 產生 `row_id,user_id,video_id,score`，再使用官方 checker 做最後格式檢查。公開 test 只可在此人工 finalization 階段使用，不能回饋 agent 的研究迴圈。

目前沒有 submission CSV，沒有 submit，也沒有 test score。

## Local preview/check

只做 validation preview：

```powershell
python submission_ready\generate_submission.py --split valid --output submission_ready\validation_preview.csv
python submission_ready\local_checker.py submission_ready\validation_preview.csv --split valid --score
```

正式 test submission 的 generator/checker 路徑需要人工 finalization 時額外指定 `--allow-final-test`；這次沒有執行該路徑，也不會把 test 結果回饋 agent。
