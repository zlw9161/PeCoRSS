# Push 到 `PeCoRSS:tfoc`（含回滚 Tag）

本目录 `PeCoRSS_tfoc_export_clean/` 已包含已整理好的 `mvrss/` 与 `svrss/`（以及各自 README）。

由于当前运行环境无法解析 `github.com`（DNS 失败），我在这里提供一套**你本机可直接执行**的 push 流程：先对远端 `tfoc` 打 Tag 作为回滚点，然后把本目录内容覆盖到仓库对应目录并正常 push（不做强制推送）。

## 一键式流程（推荐）

```bash
set -euo pipefail

# 1) 准备一个临时 clone
WORKDIR="$(pwd)"
TMP="${WORKDIR}/_PeCoRSS_push"
rm -rf "${TMP}"
git clone --branch tfoc --single-branch https://github.com/zlw9161/PeCoRSS.git "${TMP}"

cd "${TMP}"

# 2) 远端当前 tfoc 打回滚 Tag（保留旧版本）
TS="$(date +%Y%m%d_%H%M%S)"
git fetch origin tfoc
git checkout tfoc
git pull --ff-only origin tfoc
git tag "backup-tfoc-${TS}"
git push origin "backup-tfoc-${TS}"

# 3) 将整理好的代码同步到仓库（覆盖同名文件，保留历史）
rsync -a --delete "${WORKDIR}/mvrss/" "${TMP}/mvrss/"
rsync -a --delete "${WORKDIR}/svrss/" "${TMP}/svrss/"

# 4) 提交并推送（正常 push，不重写历史）
git add mvrss svrss
git commit -m "chore: add cleaned mvrss/svrss bundles"
git push origin tfoc

echo "Done. Remote branch: tfoc, rollback tag: backup-tfoc-${TS}"
```

## 回滚方式示例

- 回到打 Tag 时的 `tfoc` 状态：

```bash
git checkout tfoc
git reset --hard "backup-tfoc-<TS>"
```

（注：回滚到远端需要你自行决定是否 push、以及是否使用 force；一般建议通过新分支/PR 回滚。）
