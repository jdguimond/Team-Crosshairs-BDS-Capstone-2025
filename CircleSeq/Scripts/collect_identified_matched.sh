#!/bin/bash
# Copy all *_identified_matched.txt into results/identified_off_targets/ 
# for downstream processing. 

set -u 

RESULTS_DIR="${1:-$(pwd)/results}"
DEST_DIR="$RESULTS_DIR/identified_off_targets"
mkdir -p "$DEST_DIR"

echo "[info] results dir: $RESULTS_DIR"
echo "[info] dest dir    : $DEST_DIR"

count=0
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  dest="$DEST_DIR/$base"
  cp -f "$f" "$dest" && echo "copied: $f -> $dest"
  ((count++))
done < <(find "$RESULTS_DIR" -type f -name '*_identified_matched.txt' -print0)

echo "[info] copied $count matched file(s)"
