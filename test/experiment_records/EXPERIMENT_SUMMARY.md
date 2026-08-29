# KuaiRand 實驗紀錄整理

## 封存範圍

本資料夾封存先前的 experiment JSON、runner log、autonomous log、Multi-task log、失敗分析與 code-diff audit artifacts。這裡的 `test` 是紀錄 archive 名稱，不是 KuaiRand test split；本次整理沒有載入 test label。

## 共同流程

1. Local planner 從 `agent/configs/search_space.json` 選擇一個已登記候選。
2. Governor 檢查 module、command、split 與路徑是否 allowlisted。
3. `agent/validation_experiment_runner.py` 啟動固定 subprocess。
4. 模型只使用 train 資料訓練，以 validation 計算 GAUC、nDCG@5 與 Primary。
5. 使用 early stopping、confirmation validation 與 family-best checkpoint。
6. runner 記錄 hypothesis、config、metrics、runtime、recovery、planner decision 與 checkpoint。
7. 只有通過 baseline + epsilon 且 confirmation guard 同樣通過，才可 promote；否則 planner 排除該 config 並選下一個候選。

## 主要研究方向與結果

| 方向 | 目的 | 代表 validation Primary | 結果 |
|---|---|---:|---|
| Baseline FM | 建立官方 FM 參考線 | 0.6014695168 | 基準 |
| BPR FM | 直接最佳化正負曝光 pair 的排序差 | 0.6036759019 | 優於 baseline，主要改善 primary ranking |
| Listwise FM | 以完整 user exposure group 做 softmax ranking | 0.6011343598 | 單一 listwise 未超過 baseline gate |
| Listwise ensemble | 組合 E 槽四個 validation-selected member | 0.6040996313 | 目前最佳候選 |
| History FM | 加入 train-only prior user/tab/author history | 0.6010140181 | 未穩定改善 |
| Multi-task FM | `long_view` primary，其他行為作 auxiliary target | 0.6040930152 | exact BPR primary + isolated auxiliary path 後達標 |
| CWM | 使用 censored watch-time one-sided objective | 0.6008452773 | 未超過 baseline gate |

## Multi-task 失敗分析

早期 Multi-task 約為 `0.6024237275`，低於目標。原因是 pointwise、pairwise 與 click auxiliary gradient 同時更新 shared FM，形成 loss coupling。嘗試 primary-first、gradient projection、獨立 auxiliary head 與 BPR clone 後，仍未完全重現 standalone BPR。

最後採用 exact BPR primary optimizer，並將 auxiliary path 隔離；validation Primary 達到 `0.6040930152`。這支持「瓶頸主要在 primary ranking optimizer，而不是單純資料量不足」的判斷。

## 目前最佳 frozen candidate

目前 submission-ready candidate 是 `listwise_ensemble`，validation Primary `0.6040996313`。其 confirmation Primary 為 `0.5978333950`，因此封存為候選但不宣稱已通過 anti-overfitting confirmation gate。
