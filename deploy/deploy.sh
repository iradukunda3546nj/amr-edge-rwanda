#!/usr/bin/env bash
# AMR-Edge Rwanda: atomic deploy of src/web to Linode.
# Uploads into a timestamped release, flips the "current" symlink, reloads Nginx,
# keeps the last 5 releases for instant rollback.
#
# Usage (Git Bash / WSL / macOS / Linux, from repo root):
#   DEPLOY_HOST=user@203.0.113.10 ./deploy/deploy.sh
# Rollback:
#   ssh user@host 'cd /var/www/amr_edge && ln -sfn "releases/$(ls -1 releases | tail -2 | head -1)" current'
set -euo pipefail

HOST="${DEPLOY_HOST:?set DEPLOY_HOST=user@linode-ip}"
BASE=/var/www/amr_edge
REL="$(date -u +%Y%m%d%H%M%S)"
SRC="$(cd "$(dirname "$0")/.." && pwd)/src/web"

[ -f "$SRC/index.html" ] || { echo "missing $SRC/index.html" >&2; exit 1; }

echo "→ uploading release $REL to $HOST"
tar -C "$SRC" -czf - . | ssh "$HOST" "
  set -e
  mkdir -p $BASE/releases/$REL
  tar -C $BASE/releases/$REL -xzf -
  ln -sfn $BASE/releases/$REL $BASE/current
  cd $BASE/releases && ls -1 | head -n -5 | xargs -r rm -rf
  sudo nginx -t && sudo systemctl reload nginx
"
echo "✓ live: https://amr-edge.zolilabs.com"
