# GitHub push checklist

## 允許提交

- `agent/` source、reviewed modules、allowlist 與 bounded search-space generator
- `kuairand-starter-kit/` 官方 starter code
- README、WORKFLOW、環境設定與操作文件
- `submission_ready/manifest.json`、adapter、generator、local checker 與 README
- `test/experiment_records/EXPERIMENT_SUMMARY.md`

## 必須留在本地

- `runs/`、非 frozen 的 `outputs/`、`*.csv`
- 原始資料與壓縮資料集
- `.env`、API key、token、private key
- transfer zip 與本機 logs／patches

`outputs/pure/` 下的 frozen `.npz` 是唯一例外，必須透過 Git LFS 追蹤；目前共 11 個檔案，約 34.6 MB。

## Push 前命令

```powershell
python -m py_compile agent\*.py agent\modules\*.py
python -m json.tool agent\configs\search_space.json > $null
git diff --check
git status --short
git diff --cached --name-only
```

確認 `git lfs ls-files` 顯示所有 frozen weights，且不要使用 `git add -f` 將 dataset、非 frozen checkpoint 或 submission CSV 加入 commit。此檔案只提供準備清單，不會自動 commit 或 push。
