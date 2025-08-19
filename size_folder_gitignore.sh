#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"

if [ ! -d "$ROOT" ]; then
  echo "❌ 目录不存在: $ROOT" >&2
  exit 1
fi

cd "$ROOT"

# 优先使用 git ls-files（严格遵守 .gitignore）
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "📦 检测到 Git 仓库，使用 git ls-files ..."
  # 包括已追踪和未忽略的未追踪文件
  SIZE=$(git ls-files -z --cached --others --exclude-standard \
    | xargs -0 du -cb 2>/dev/null \
    | tail -n1 | awk '{print $1}')
else
  if [ -f ".gitignore" ]; then
    echo "📜 没有 Git 仓库，但检测到 .gitignore，尝试用 rsync 模拟 ..."
    # rsync filter 语法兼容 .gitignore，创建一个临时文件
    TMP=$(mktemp)
    # rsync 的 filter 需要前缀 ":" 来引入 gitignore 语法
    echo ": .gitignore" > "$TMP"
    SIZE=$(rsync -a --filter="merge $TMP" --dry-run --stats ./ /dev/null \
      | awk '/Total file size:/ {print $4; unit=$5}
             END{ if(unit=="bytes") print $4; else print $4 unit }')
    rm -f "$TMP"
  else
    echo "⚠️ 没有 Git 仓库，也没有 .gitignore，计算整个目录大小 ..."
    SIZE=$(du -sb . | awk '{print $1}')
  fi
fi

# 转换为人类可读
if [[ "$SIZE" =~ ^[0-9]+$ ]]; then
  HR=$(numfmt --to=iec-i --suffix=B "$SIZE")
  echo "✅ 有效大小: $HR ($SIZE bytes)"
else
  echo "✅ 有效大小: $SIZE"
fi
