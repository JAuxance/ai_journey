#!/usr/bin/env bash
# Auto-increment the "Day N" counter in the repo-root README.md, at most once
# per calendar day. Wired as a SessionStart hook in .claude/settings.json.
#
# Repo root is derived from this script's location (robust across machines),
# so it works on any PC the repo is cloned to — no hardcoded paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
README="$REPO/README.md"
STATE="$REPO/.claude/.last-day-bump"
TODAY="$(date +%Y-%m-%d)"

# Already bumped today (or README missing) -> do nothing.
[[ -f "$README" ]] || exit 0
if [[ -f "$STATE" && "$(cat "$STATE" 2>/dev/null)" == "$TODAY" ]]; then
  exit 0
fi

# Read the current "Day N" (first occurrence).
current="$(grep -oE 'Day [0-9]+' "$README" 2>/dev/null | head -1 | grep -oE '[0-9]+' || true)"
[[ -n "$current" ]] || exit 0

next=$((current + 1))

# Bump only the first "Day <current>" occurrence.
sed -i "0,/Day ${current}/s//Day ${next}/" "$README"
echo "$TODAY" > "$STATE"

printf '{"systemMessage": "AI journey: bumped to Day %s (first session today)"}\n' "$next"
