#!/usr/bin/env bash
# Publish docs/ to the gh-pages branch (GitHub Pages source).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: not inside a git repository" >&2
  exit 1
fi

if [[ ! -f docs/index.html || ! -f docs/style.css || ! -f docs/site.js || ! -f docs/i18n.js ]]; then
  echo "error: docs/index.html, docs/style.css, docs/site.js, and docs/i18n.js are required" >&2
  exit 1
fi

MAIN_SHA="$(git rev-parse --short HEAD)"
WORKTREE="$(mktemp -d)"
trap 'git worktree remove -f "$WORKTREE" 2>/dev/null || true; rm -rf "$WORKTREE"' EXIT

git fetch origin gh-pages 2>/dev/null || true
if git show-ref --verify --quiet refs/remotes/origin/gh-pages; then
  git worktree add -B gh-pages-publish "$WORKTREE" origin/gh-pages
else
  git worktree add -B gh-pages-publish "$WORKTREE" --orphan gh-pages-publish
fi

find "$WORKTREE" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

cp docs/index.html docs/style.css docs/site.js docs/i18n.js docs/i18n-runtime.js docs/developer.html "$WORKTREE/"
cp docs/.nojekyll "$WORKTREE/" 2>/dev/null || : >"$WORKTREE/.nojekyll"
rm -rf "$WORKTREE/assets"
mkdir -p "$WORKTREE/assets"
cp -a docs/assets/. "$WORKTREE/assets/"

# Cache-bust marker for verifying deploys in page source.
sed -i "s/site-build-main/site-build-${MAIN_SHA}/" "$WORKTREE/index.html"

cd "$WORKTREE"
git add -A
if git diff --cached --quiet; then
  echo "gh-pages: no changes to publish"
  exit 0
fi

git commit -m "Publish project page from main (${MAIN_SHA})"
git push origin HEAD:gh-pages
echo "Published docs/ to origin/gh-pages (${MAIN_SHA})"
