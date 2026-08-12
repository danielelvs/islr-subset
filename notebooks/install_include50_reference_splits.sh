#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
DEST="$ROOT/data/raw/include50"
COMMIT="053ef098718fa5e9600a00508e1968e6b94bfdfd"
BASE="https://raw.githubusercontent.com/Nittaany/Major-Project/$COMMIT/src/ml_tools"

mkdir -p "$DEST"

download_and_check () {
  local name="$1"
  local expected="$2"
  local url="$BASE/$name"
  local out="$DEST/$name"
  local tmp="$out.download"

  if [[ -s "$out" ]]; then
    local current
    current="$(grep -Ev '^[[:space:]]*(#|$)' "$out" | wc -l | tr -d ' ')"
    if [[ "$current" != "$expected" ]]; then
      echo "ERROR: $out is non-empty and has $current references; expected $expected."
      echo "Refusing to overwrite it automatically."
      exit 1
    fi
    echo "OK existing: $name ($current)"
    return
  fi

  echo "Downloading $name ..."
  curl -fL "$url" -o "$tmp"

  local got
  got="$(grep -Ev '^[[:space:]]*(#|$)' "$tmp" | wc -l | tr -d ' ')"
  if [[ "$got" != "$expected" ]]; then
    echo "ERROR: downloaded $name has $got references; expected $expected."
    rm -f "$tmp"
    exit 1
  fi

  mv "$tmp" "$out"
  echo "OK installed: $out ($got)"
}

download_and_check "include50_train.txt" 689
download_and_check "include50_val.txt" 77
download_and_check "include50_test.txt" 192

echo
echo "Final counts:"
for f in include50_train.txt include50_val.txt include50_test.txt; do
  printf "%-22s %s\n" "$f" "$(grep -Ev '^[[:space:]]*(#|$)' "$DEST/$f" | wc -l | tr -d ' ')"
done
