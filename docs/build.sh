#!/bin/sh
# Rebuild the LS Architecture PDFs with tectonic (https://tectonic-typesetting.github.io).
# Usage: docs/build.sh [design|research]    (no argument: both)
set -e
cd "$(dirname "$0")"
STYLE="$(pwd)/style"
build() { (cd "$1" && tectonic -X compile -Z search-path="$STYLE" "$2.tex") }
case "${1:-all}" in
  design)   build design ls-architecture ;;
  research) build research sota-review ;;
  all)      build design ls-architecture; build research sota-review ;;
  *)        echo "usage: $0 [design|research]" >&2; exit 2 ;;
esac
