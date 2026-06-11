#!/bin/bash

SOURCE="/d/DLProject2/DeepLearning"
TARGET_REPO="/d/ai_platforms"
TARGET_DIR="$TARGET_REPO/projects/DeepLearning"

COMMIT_MSG="Update DeepLearning $(date '+%Y-%m-%d %H:%M')"

echo "======================================="
echo "Deploy Start"
echo "======================================="

if [ ! -d "$SOURCE" ]; then
    echo "[ERROR] Source folder not found: $SOURCE"
    exit 1
fi

if [ ! -d "$TARGET_REPO/.git" ]; then
    echo "[ERROR] Git repository not found: $TARGET_REPO"
    echo "먼저 git clone을 해두셔야 합니다."
    exit 1
fi

echo ""
echo "[1] Clean target directory..."

rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo ""
echo "[2] Copy files..."

cd "$SOURCE" || exit 1

for item in * .*; do
    case "$item" in
        "."|".."|".git"|".venv"|"venv"|"__pycache__"|"datasets"|"dataset"|"checkpoints"|"deploy.sh")
            continue
            ;;
        *.pyc|*.h5|*.pt|*.pth)
            continue
            ;;
    esac

    if [ -e "$item" ]; then
        cp -r "$item" "$TARGET_DIR/"
    fi
done

if [ $? -ne 0 ]; then
    echo "[ERROR] Copy failed"
    exit 1
fi

echo ""
echo "[3] Git status..."

cd "$TARGET_REPO" || exit 1
git status

echo ""
echo "[4] Git add..."

git add projects/DeepLearning

echo ""
echo "[5] Git commit..."

if git diff --cached --quiet; then
    echo "[INFO] 변경사항이 없습니다. commit 생략."
else
    git commit -m "$COMMIT_MSG"
fi

echo ""
echo "[6] Git push..."

git push origin main

if [ $? -ne 0 ]; then
    echo "[ERROR] Push failed"
    exit 1
fi

echo ""
echo "======================================="
echo "Deploy Complete"
echo "======================================="