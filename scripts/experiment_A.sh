#!/usr/bin/env bash
# Experiment A — plain MD on ubiquitin (1UBQ): how long, and how many replicas?
#
# Hypothesis A: online, uncertainty-aware allocation reaches the same accuracy of an observable as
# fixed-budget MD, for less computing. README "实验 A" explains the design; the decisions are fixed here,
# before any data exist, so that the verdict cannot be tuned afterwards.
#
#   Observable   chi1 rotamer populations (g+, t, g-) of side chains that plain MD can answer. They are
#                chosen by a separate pilot run (step 1-2), never from the pool itself.
#   Accuracy     every population within TOLERANCE of the reference in at least TARGET_RATE of the repeats.
#   Reference    HOLDOUT pool replicas at their full length, used by no strategy.
#   Baselines    the protocols people use: 1 x 100 ns (CHARMM36, 300 K, one trajectory; Huang & MacKerell
#                2013, doi:10.1002/jcc.23354), and 3 x 100, 5 x 100 ns (the replica counts recommended by
#                Grossfield 2018 and Knapp 2018).
#   Agents       early_stop: 5 replicas, decides only how long. mdagent: starts with 5, may add up to 10.
#                Both stop with the "confidence" criterion (95% interval, alpha shared over the looks).
#   Policy       frozen: the git commit is written into the protocol file before the pool is analysed.
#
# Steps (from the MDAgent1 folder, mdagent environment active, inside tmux over SSH):
#   bash scripts/experiment_A.sh pilot     # 3 x 20 ns, all residues (60 ns, a few hours)
#   bash scripts/experiment_A.sh select    # which side chains plain MD can answer -> runs/expA_items.txt
#   bash scripts/experiment_A.sh pool      # 16 x 100 ns (1600 ns: about 2 days at 850 ns/day)
#   bash scripts/experiment_A.sh analyze   # replay; seconds. Result: runs/expA_pool/experiment_A.md
# Each step can be run again: a stopped run continues where it stopped.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
STEP=${1:-}

TOLERANCE=${TOLERANCE:-0.1}
TARGET_RATE=${TARGET_RATE:-0.9}
POOL_REPLICAS=${POOL_REPLICAS:-16}
POOL_NS=${POOL_NS:-100}
HOLDOUT=${HOLDOUT:-6}
REPEATS=${REPEATS:-50}
PILOT=runs/expA_pilot
POOL=runs/expA_pool
ITEMS=runs/expA_items.txt
COMMON="--pdb structures/1UBQ.pdb --uniprot P0CG48 --termini charged --residues A:1-76 --allow none"

[ -e structures/1UBQ.pdb ] || python -m mdagent fetch 1UBQ

run_or_resume() {   # $1 = folder, rest = run options
  local out=$1; shift
  if [ -e "$out/run.json" ]; then
    echo "$(date '+%F %T') continuing $out"
    python -m mdagent resume --run "$out" >> "$out.log" 2>&1
  else
    echo "$(date '+%F %T') starting $out"
    python -m mdagent run $COMMON "$@" --out "$out" > "$out.log" 2>&1
  fi || { echo "$out ended with an error, see $out.log"; exit 1; }
}

case "$STEP" in
  pilot)
    # --allow none: the pool and the pilot make no decisions of their own. They are data, not strategies.
    run_or_resume "$PILOT" --replicas 3 --first-ns 20 --max-ns 20 --seed 101
    ;;
  select)
    # Rule fixed in advance: a side chain is in the experiment when it changed rotamer state at least
    # 4 times in every pilot replica, which is the same rule the agent uses for "plain MD can answer it".
    python - "$PILOT/status.json" "$ITEMS" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))["residues"]
keep = [f"{r['resname']}{r['resseq']}" for r in rows if r.get("transitions") and min(r["transitions"]) >= 4]
slow = [f"{r['resname']}{r['resseq']}" for r in rows if r.get("transitions") and min(r["transitions"]) < 4]
open(sys.argv[2], "w").write(",".join(keep) + "\n")
print(f"In the experiment ({len(keep)}): {', '.join(keep) or 'none'}")
print(f"Left out, too slow for plain MD in 20 ns ({len(slow)}): {', '.join(slow) or 'none'}")
PY
    ;;
  pool)
    [ -e "$ITEMS" ] || { echo "Run the pilot and 'select' first."; exit 2; }
    {
      echo "Experiment A protocol, frozen $(date '+%F %T')"
      echo "commit: $(git rev-parse HEAD)"
      echo "items: $(cat $ITEMS)"
      echo "tolerance: $TOLERANCE, target rate: $TARGET_RATE, holdout: $HOLDOUT, repeats: $REPEATS"
      echo "pool: $POOL_REPLICAS x $POOL_NS ns"
    } > runs/expA_protocol.txt
    # Rounds of 20 ns, so a stop loses at most 20 ns per replica.
    run_or_resume "$POOL" --replicas "$POOL_REPLICAS" --first-ns 20 --extend-ns 20 --max-ns "$POOL_NS" --seed 202
    ;;
  analyze)
    [ -e "$POOL/run.json" ] || { echo "Run the pool first."; exit 2; }
    python -m mdagent budget --run "$POOL" --items "$(cat $ITEMS)" --holdout "$HOLDOUT" \
      --replicas 5 --first-ns 5 --extend-ns 5 --max-ns "$POOL_NS" --extend-factor 1 --max-replicas 10 \
      --precision "$TOLERANCE" --criterion confidence --alpha 0.05 --target-rate "$TARGET_RATE" \
      --fixed-grid 1x100,3x100,5x100,3x20,3x50,5x20,5x40,5x60,5x80,10x20,10x50 \
      --baseline 1x100,3x100,5x100 --repeats "$REPEATS" --name experiment_A
    ;;
  *)
    echo "Usage: bash scripts/experiment_A.sh pilot|select|pool|analyze"; exit 2 ;;
esac
