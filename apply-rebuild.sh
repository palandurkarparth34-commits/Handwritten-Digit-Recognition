#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /path/to/Handwritten-Digit-Recognition" >&2
  exit 2
fi

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$1" && pwd)"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "Error: $REPO_DIR is not a Git working tree." >&2
  exit 1
fi

ORIGIN="$(git -C "$REPO_DIR" remote get-url origin)"
case "$ORIGIN" in
  https://github.com/palandurkarparth34-commits/Handwritten-Digit-Recognition|https://github.com/palandurkarparth34-commits/Handwritten-Digit-Recognition.git|git@github.com:palandurkarparth34-commits/Handwritten-Digit-Recognition.git|ssh://git@github.com/palandurkarparth34-commits/Handwritten-Digit-Recognition.git)
    ;;
  *)
    echo "Error: unexpected origin remote: $ORIGIN" >&2
    echo "This script is intended only for palandurkarparth34-commits/Handwritten-Digit-Recognition." >&2
    exit 1
    ;;
esac

if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
  echo "Error: the repository has uncommitted changes. Commit or stash them before applying the rebuild." >&2
  exit 1
fi

if git -C "$REPO_DIR" ls-remote --exit-code --heads origin rebuild >/dev/null 2>&1; then
  echo "Error: origin/rebuild already exists. Nothing was changed; inspect that branch before proceeding." >&2
  exit 1
fi

if git -C "$REPO_DIR" show-ref --verify --quiet refs/heads/rebuild; then
  echo "Error: a local rebuild branch already exists. Nothing was changed." >&2
  exit 1
fi

echo "Fetching main and creating the rebuild branch…"
git -C "$REPO_DIR" fetch origin
git -C "$REPO_DIR" switch main
git -C "$REPO_DIR" pull --ff-only origin main
git -C "$REPO_DIR" switch -c rebuild origin/main

# Replace the web app and its tests. Keep LICENSE and the historical docs/ folder.
cp "$SOURCE_DIR/app.py" "$REPO_DIR/app.py"
cp "$SOURCE_DIR/README.md" "$REPO_DIR/README.md"
cp "$SOURCE_DIR/requirements.txt" "$REPO_DIR/requirements.txt"
cp "$SOURCE_DIR/requirements-dev.txt" "$REPO_DIR/requirements-dev.txt"
cp "$SOURCE_DIR/.gitignore" "$REPO_DIR/.gitignore"

rm -rf "$REPO_DIR/templates" "$REPO_DIR/tests" "$REPO_DIR/src" "$REPO_DIR/data"
rm -f "$REPO_DIR/requirements-cnn.txt"
mkdir -p "$REPO_DIR/templates" "$REPO_DIR/tests"
cp "$SOURCE_DIR/templates/index.html" "$REPO_DIR/templates/index.html"
cp "$SOURCE_DIR/tests/test_app.py" "$REPO_DIR/tests/test_app.py"

git -C "$REPO_DIR" add -A
git -C "$REPO_DIR" commit -m "Rebuild digit recognition app with polished UI and reliable inference"
git -C "$REPO_DIR" push -u origin rebuild

echo
echo "Done. The rebuild has been pushed to origin/rebuild."
echo "Review it at: https://github.com/palandurkarparth34-commits/Handwritten-Digit-Recognition/tree/rebuild"
