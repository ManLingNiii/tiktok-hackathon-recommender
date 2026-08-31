# GitHub push checklist

## 允許提交

- `agent/` source、reviewed modules、allowlist 與 bounded search-space generator
- `kuairand-starter-kit/` 官方 starter code
- README、WORKFLOW、環境設定與操作文件
- `submission_ready/composition_manifest.json`、checkpoint registry、adapter、generator、local checker 與 README

## 必須留在本地

- `runs/`、非 frozen 的 `outputs/`、`*.csv`
- 原始資料與壓縮資料集（一般 branch；`full-system` branch 依需求以 Git LFS 管理 KuaiRand-Pure CSV）
- `.env`、API key、token、private key
- transfer zip 與本機 logs／patches

`outputs/pure/` 下的 frozen `.npz` 是唯一例外，必須透過 Git LFS 追蹤；目前保留五個 family initial checkpoint 與一個 prepared composition checkpoint。

## Push 前命令

```powershell
python -m py_compile agent\*.py agent\modules\*.py
python -m json.tool agent\configs\search_space.json > $null
git diff --check
git status --short
git diff --cached --name-only
```

確認 `git lfs ls-files` 顯示所有 frozen weights 與 `full-system` branch 的 KuaiRand-Pure CSV，且不要使用 `git add -f` 將非 frozen checkpoint 或 submission CSV 加入 commit。此檔案只提供準備清單，不會自動 commit 或 push。
