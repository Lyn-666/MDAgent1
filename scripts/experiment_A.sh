#!/usr/bin/env bash
# Experiment A — task-aware budget allocation for unbiased MD, on ubiquitin (1UBQ).
#
# Question: given an MD task and a precision fixed in advance, can the agent decide how many replicas,
# how long each one runs and when to stop, so that the result is reliable for the least computing?
# docs/developer.md "Experiment A" explains the design. Every rule below is fixed before any data exist; the code version
# is written into runs/expA_protocol.txt before the pool is analysed.
#
#   Observables  backbone basins. A-Easy: phi/psi Ramachandran basins of residues that plain MD can answer,
#                chosen by a separate pilot. A-Hard: the D52/G53 beta-I/beta-II turn (psi D52, phi G53), which
#                flips on the 10 us scale (Arasteh 2021, doi:10.1021/acs.jpclett.1c00778; Smith 2016,
#                doi:10.1073/pnas.1519609113): a 1.6 us pool cannot give it a reference, so A-Hard tests only
#                whether a method honestly says INSUFFICIENT_EVIDENCE instead of a false STOP.
#   Arms         fixed plans of 1, 3, 4, 5, 8 replicas (scanned over length), Sequential-CI with 4 and 8
#                replicas, a pilot followed by a fixed plan, and the agent. All unbiased; new replicas start
#                from the prepared structure. Every method leaves out each replica's non-equilibrated start
#                (automated detection) and uses Student's t where the replica spread is real (chi-square test).
#   Reference    6 pool replicas held out at full length, used by no method.
#   Independence Trials that draw from one pool share most replicas. With 16 replicas the report says so
#                (about 1 independent unit). For claims, use DISJOINT=1 with a larger pool, for example
#                POOL_REPLICAS=40 HOLDOUT=8: the other 32 replicas form 4 groups of 8 that share no replica.
#
# Steps (from the MDAgent1 folder, mdagent environment active, inside tmux over SSH):
#   bash scripts/experiment_A.sh pilot     # 3 x 20 ns (60 ns, a few hours)
#   bash scripts/experiment_A.sh select    # which residues plain MD can answer -> runs/expA_easy.txt
#   bash scripts/experiment_A.sh pool      # 16 x 100 ns (1600 ns: 2 to 5 days, depending on the GPU)
#   bash scripts/experiment_A.sh analyze   # replay, a few minutes -> runs/expA_pool/experiment_A_*.md
# Each step can be run again: a stopped run continues where it stopped.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs
STEP=${1:-}

POPULATION=${POPULATION:-0.05}     # tolerance on every basin population
RELIABILITY=${RELIABILITY:-0.9}   # share of trials that must succeed
POOL_REPLICAS=${POOL_REPLICAS:-16}
POOL_NS=${POOL_NS:-100}
HOLDOUT=${HOLDOUT:-6}
TRIALS=${TRIALS:-50}
DISJOINT=${DISJOINT:-0}           # 1: trials use groups of replicas that share no replica (independent trials)
PILOT=runs/expA_pilot
POOL=runs/expA_pool
EASY=runs/expA_easy.txt
COMMON="--pdb structures/1UBQ.pdb --uniprot P0CG48 --termini charged --residues A:1-76 --allow none --allocation rounds"

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
    # --allow none: the pilot and the pool make no decisions of their own. They are data, not methods.
    run_or_resume "$PILOT" --replicas 3 --first-ns 20 --max-ns 20 --seed 101
    ;;
  select)
    # Rule fixed in advance: a residue enters A-Easy when its phi/psi basin changed at least 4 times in
    # every pilot replica; if more qualify, the 5 with the fewest transitions are kept.
    python -m mdagent experiment-a --run "$PILOT" --observables "rama:A:2-75" --select --at-most 5 --out "$EASY"
    echo
    echo "The beta-I/beta-II turn (A-Hard) in the pilot:"
    python -m mdagent experiment-a --run "$PILOT" --observables "turn:A:52" --select --at-most 1 || true
    ;;
  pool)
    [ -e "$EASY" ] || { echo "Run the pilot and 'select' first."; exit 2; }
    {
      echo "Experiment A protocol, frozen $(date '+%F %T')"
      echo "commit: $(git rev-parse HEAD)"
      echo "A-Easy items: $(cat $EASY)"
      echo "A-Hard item: turn:A:52"
      echo "population tolerance: $POPULATION, reliability: $RELIABILITY, holdout: $HOLDOUT, trials: $TRIALS"
      echo "pool: $POOL_REPLICAS x $POOL_NS ns"
    } > runs/expA_protocol.txt
    # Rounds of 20 ns, so a stop loses at most 20 ns per replica.
    run_or_resume "$POOL" --replicas "$POOL_REPLICAS" --first-ns 20 --extend-ns 20 --max-ns "$POOL_NS" --seed 202
    ;;
  analyze)
    [ -e "$POOL/run.json" ] || { echo "Run the pool first."; exit 2; }
    ARGS="--run $POOL --holdout $HOLDOUT --trials $TRIALS --max-ns $POOL_NS --population $POPULATION
          --reliability $RELIABILITY"
    [ "$DISJOINT" = 1 ] && ARGS="$ARGS --disjoint"
    python -m mdagent experiment-a $ARGS --observables "$(cat $EASY)" --name experiment_A_easy
    python -m mdagent experiment-a $ARGS --observables "turn:A:52" --name experiment_A_hard
    ;;
  *)
    echo "Usage: bash scripts/experiment_A.sh pilot|select|pool|analyze"; exit 2 ;;
esac
