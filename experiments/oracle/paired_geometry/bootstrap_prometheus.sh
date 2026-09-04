#!/usr/bin/env bash
# Clone + install Prometheus OUTSIDE the repository.
#
# Prometheus ships PPC ice tables, PROPOSAL tables and cross-section splines.
# None of that belongs in a git repo that already says "nothing raw enters the
# repo" (DATASETS.md).  Default root is a sibling of the repo.
#
#   usage: ORACLE_SIM_ROOT=/ceph/dwong/oracle_sim ./bootstrap_prometheus.sh
set -euo pipefail

ROOT="${ORACLE_SIM_ROOT:-$HOME/Documents/oracle_sim}"
PROM_REF="${PROM_REF:-main}"          # pin this before any production run
mkdir -p "$ROOT"

if [ ! -d "$ROOT/prometheus/.git" ]; then
  git clone https://github.com/Harvard-Neutrino/prometheus.git "$ROOT/prometheus"
fi
cd "$ROOT/prometheus"
git fetch --all --tags
git checkout "$PROM_REF"
git rev-parse HEAD > "$ROOT/prometheus_commit.txt"
echo "Prometheus pinned at $(cat "$ROOT/prometheus_commit.txt")"

# Licence note: Prometheus is LGPL-2.1.  Using it as a library and shipping
# our own configs/scripts is fine.  If we ever modify prometheus/ itself, the
# modified files must be offered under LGPL-2.1 -- keep our changes in THIS
# repo and inject them via config, not by patching the clone.
echo "LGPL-2.1: keep local modifications out of the clone." 

python3 -m venv "$ROOT/.venv" 2>/dev/null || true
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
pip install -U pip
pip install -r requirements.txt
pip install -e .

echo
echo "Smoke test (3 events, demo ice geo, CPU):"
python examples/02_basic_ice.py && echo "PROMETHEUS OK"
