#!/usr/bin/env bash
# The main experiment: how should simulation time be spent to reach a precision for the lowest cost?
#
# It runs ONE expensive pool of independent plain MD replicas, and then replays every budget strategy
# on that pool (python -m mdagent budget, docs/developer.md "How much compute: comparing strategies"). Replaying is exact for
# strategies that only run longer or start another independent replica, so the whole comparison costs
# one simulation campaign instead of one per strategy.
#
# The pool must be generous in both directions, because the strategies are cut out of it:
#   - enough replicas: the strategies use some, and the rest are the independent reference. With
#     POOL_REPLICAS=10 and strategies starting at 3, about 5 are left over.
#   - long enough: longer than the budget any strategy may reach (MAX_NS x the extend factor).
#
# Usage (from the MDAgent1 folder, mdagent environment active; inside tmux over SSH):
#   bash scripts/budget_experiment.sh                 # 1UBQ, 10 replicas x 60 ns (about a day on one GPU)
#   bash scripts/budget_experiment.sh 181L
#   POOL_REPLICAS=6 POOL_NS=20 bash scripts/budget_experiment.sh      # a shorter first try
# Running it again does not repeat the pool: it continues an unfinished one and replays again.
# Results: runs/pool_<protein>/budget.md and budget.json, and the three criteria side by side in
# runs/pool_<protein>/budget_<criterion>.md.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
SYSTEM=${1:-1UBQ}
POOL_REPLICAS=${POOL_REPLICAS:-10}
POOL_NS=${POOL_NS:-60}
# What every strategy is allowed to do. Keep it the same for all of them, or the comparison is unfair.
START_REPLICAS=${START_REPLICAS:-3}
FIRST_NS=${FIRST_NS:-2}
EXTEND_NS=${EXTEND_NS:-2}
MAX_NS=${MAX_NS:-20}
PRECISION=${PRECISION:-0.05}
REPEATS=${REPEATS:-10}
PARALLEL=${PARALLEL:-1}

case "$SYSTEM" in
  1UBQ)
    [ -e structures/1UBQ.pdb ] || python -m mdagent fetch 1UBQ
    COMMON="--pdb structures/1UBQ.pdb --uniprot P0CG48 --termini charged"
    RESIDUES="A:16,A:32,A:63,A:26" ;;
  181L)
    COMMON="--pdb examples/181L.pdb --accept-problems"
    RESIDUES="A:80,A:135,A:162,A:111" ;;
  *) echo "Unknown protein $SYSTEM. Choose 1UBQ or 181L."; exit 2 ;;
esac

POOL=runs/pool_$SYSTEM
if [ -e "$POOL/run.json" ]; then
  echo "$(date '+%F %T') pool exists: continuing it to $POOL_NS ns per replica."
  python -m mdagent resume --run "$POOL" --max-ns "$POOL_NS" >> "$POOL.log" 2>&1 \
    || { echo "the pool ended with an error, see $POOL.log"; exit 1; }
else
  # --allow none: the pool makes no decisions of its own. It is data, not a strategy.
  echo "$(date '+%F %T') running the pool: $POOL_REPLICAS replicas x $POOL_NS ns."
  python -m mdagent run $COMMON --allocation rounds --residues "$RESIDUES" --replicas "$POOL_REPLICAS" \
      --first-ns "$POOL_NS" --max-ns "$POOL_NS" --allow none --out "$POOL" > "$POOL.log" 2>&1 \
    || { echo "the pool ended with an error, see $POOL.log"; exit 1; }
fi

COMPARE="--run $POOL --replicas $START_REPLICAS --first-ns $FIRST_NS --extend-ns $EXTEND_NS --max-ns $MAX_NS
         --precision $PRECISION --repeats $REPEATS --parallel $PARALLEL"
for criterion in standard_error confidence sequence; do
  echo; echo "=== precision criterion: $criterion ==="
  python -m mdagent budget $COMPARE --criterion "$criterion" --name "budget_$criterion" \
    || echo "the replay failed for $criterion"
done
# The default report, for the criterion the agent uses by default.
python -m mdagent budget $COMPARE --criterion standard_error --name budget > /dev/null
echo
echo "Written: $POOL/budget.md and $POOL/budget_<criterion>.md (also .json)."
echo "The column that decides is 'accurate': cheaper only counts if the answer is still within the tolerance."
