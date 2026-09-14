#!/usr/bin/env bash
#
# Sync game_data/ from the authoritative source: the RTS game repo.
#
# game_data/ (building costs, recipes, map, gameplay JSON) is OWNED by the game
# repo, which consumes it at runtime. This AI repo keeps its own committed copy
# so it stays clone-and-run without the (private) game repo present. This script
# is the one-way bridge: game repo -> this repo.
#
#   ./scripts/sync_game_data.sh          # copy from source into ./game_data
#   ./scripts/sync_game_data.sh --check  # exit 1 if our copy has drifted
#
# Source location defaults to the sibling checkout ../rts_rust_game; override with
#   GAME_REPO=/path/to/rts_rust_game ./scripts/sync_game_data.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO_ROOT/game_data"
GAME_REPO="${GAME_REPO:-$REPO_ROOT/../rts_rust_game}"
SRC="$GAME_REPO/game_data"

MODE="sync"
[ "${1:-}" = "--check" ] && MODE="check"

if [ ! -d "$SRC" ]; then
  if [ "$MODE" = "check" ]; then
    echo "sync_game_data: source not found ($SRC) — skipping drift check." >&2
    exit 0
  fi
  echo "sync_game_data: source not found: $SRC" >&2
  echo "  Set GAME_REPO to the rts_rust_game checkout." >&2
  exit 1
fi

# Only the JSON files this repo tracks are authoritative; sync exactly those.
files=$(cd "$DEST" && ls *.json)

drift=0
for f in $files; do
  if [ ! -f "$SRC/$f" ]; then
    echo "sync_game_data: MISSING in source: $f" >&2
    drift=1
    continue
  fi
  if ! diff -q "$SRC/$f" "$DEST/$f" >/dev/null 2>&1; then
    if [ "$MODE" = "check" ]; then
      echo "DRIFT: $f differs from source" >&2
      drift=1
    else
      cp "$SRC/$f" "$DEST/$f"
      echo "synced: $f"
    fi
  fi
done

if [ "$MODE" = "check" ]; then
  if [ "$drift" -ne 0 ]; then
    echo "sync_game_data: game_data has drifted from the game repo. Run ./scripts/sync_game_data.sh" >&2
    exit 1
  fi
  echo "sync_game_data: in sync with $SRC"
fi
