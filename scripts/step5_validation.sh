#!/usr/bin/env bash
# Step 5: check that the agent judges correctly on real simulations, not only that it runs.
#
# Two test proteins. The plan is the same for any protein (docs/developer.md "第 5 步"); pick, for each scenario,
# residues of that kind in your protein.
#
# 1UBQ (human ubiquitin, 76 residues, no ligand; default). The file names UniProt P62988, which is no
# longer active, so the reference is P0CG48. Free ubiquitin is cut after Gly76, so its chain ends stay
# charged (--termini charged).
#   reach_chi1     exposed side chains Glu16, Asp32, Lys63: plain MD should reach the goal and stop.
#   reach_dist     the surface salt bridge Lys11-Glu34: should reach the goal and stop.
#   trapped_chi1   buried Val26, plain MD only (guided): every replica stays in one state -> "stalled".
#   trapped_dist   the buried salt bridge Lys27-Asp52: likely never breaks -> "stalled" (cannot be told
#                  apart from a single stable state; the report must say so).
#   budget_chi1    exposed side chains, precision 0.01 in 2 ns, no permission to extend -> "budget used up".
#   extend_chi1    the same with permission to extend -> runs longer, at most 2 x the budget.
#   enhanced_chi1  Val26, auto: switches to enhanced sampling and stops when the precision is reached.
#   ref_*          longer independent runs (another seed, no early stop): "mdagent compare" must say the
#                  early-stopped results agree with them within their errors.
# 181L (T4 lysozyme L99A): the same scenarios with Arg80/Lys135/Lys162, Lys162-Asp159, Val111, Arg145-Glu11.
#
# Over SSH, start it inside tmux (or with nohup), so that closing the window does not stop it.
# Running the script again skips finished scenarios and continues one that was stopped halfway.
# Usage (from the MDAgent1 folder, mdagent environment active):
#   bash scripts/step5_validation.sh                         # 1UBQ, all scenarios (about a night)
#   bash scripts/step5_validation.sh 1UBQ budget_chi1 extend_chi1   # only some
#   bash scripts/step5_validation.sh 181L                    # the other protein
# Results: runs/val_<protein>_<name>/report.en.md and report.zh.md; summary in runs/val_<protein>_summary.txt.
set -u
cd "$(dirname "$0")/.."
mkdir -p runs   # a fresh clone has no runs folder, and the logs below are written into it
SYSTEM=${1:-1UBQ}
[ $# -gt 0 ] && shift

case "$SYSTEM" in
  1UBQ)
    [ -e structures/1UBQ.pdb ] || python -m mdagent fetch 1UBQ
    COMMON="--pdb structures/1UBQ.pdb --uniprot P0CG48 --termini charged --replicas 3"
    FAST="A:16,A:32,A:63"; SLOW="A:26"; DIST_FAST="A:11:NZ-A:34:CD@0.45"; DIST_SLOW="A:27:NZ-A:52:CG@0.45" ;;
  181L)
    COMMON="--pdb examples/181L.pdb --accept-problems --replicas 3"
    FAST="A:80,A:135,A:162"; SLOW="A:111"; DIST_FAST="A:162:NZ-A:159:CG@0.45"; DIST_SLOW="A:145:CZ-A:11:CD@0.5" ;;
  *) echo "Unknown protein $SYSTEM. Choose 1UBQ or 181L."; exit 2 ;;
esac

declare -A RUN=(
  [reach_chi1]="--residues $FAST --first-ns 1 --extend-ns 2 --max-ns 10"
  [ref_reach_chi1]="--residues $FAST --first-ns 10 --max-ns 10 --allow none --seed 7"
  [reach_dist]="--distances $DIST_FAST --first-ns 1 --extend-ns 2 --max-ns 10"
  [ref_reach_dist]="--distances $DIST_FAST --first-ns 10 --max-ns 10 --allow none --seed 7"
  [trapped_chi1]="--residues $SLOW --mode guided --first-ns 1 --extend-ns 2 --max-ns 6 --no-flip-patience-ns 3"
  [trapped_dist]="--distances $DIST_SLOW --first-ns 1 --extend-ns 2 --max-ns 6 --no-flip-patience-ns 3"
  [budget_chi1]="--residues $FAST --precision 0.01 --first-ns 1 --extend-ns 1 --max-ns 2 --allow switch_method,stop_early"
  [extend_chi1]="--residues $FAST --precision 0.01 --first-ns 1 --extend-ns 1 --max-ns 2 --allow extend,switch_method,stop_early"
  [enhanced_chi1]="--residues $SLOW --first-ns 1 --extend-ns 2 --max-ns 3 --no-flip-patience-ns 3 --enhanced-ns 20 --allow switch_method,stop_early"
  [ref_enhanced_chi1]="--residues $SLOW --first-ns 1 --extend-ns 2 --max-ns 3 --no-flip-patience-ns 3 --enhanced-ns 20 --allow switch_method --seed 7"
)
ORDER=(reach_chi1 ref_reach_chi1 reach_dist ref_reach_dist trapped_chi1 trapped_dist budget_chi1 extend_chi1
       enhanced_chi1 ref_enhanced_chi1)
[ $# -gt 0 ] && ORDER=("$@")
P=runs/val_${SYSTEM}

for name in "${ORDER[@]}"; do
  out=${P}_$name
  if [ -e "$out/report.en.md" ]; then echo "$name: already done"; continue; fi
  if [ -e "$out/run.json" ]; then
    # Stopped halfway (for example the SSH window closed without tmux): continue where it stopped.
    echo "$(date '+%F %T') $name: continuing (python -m mdagent resume --run $out)"
    python -m mdagent resume --run "$out" >> "$out.log" 2>&1 || echo "$name: ended with an error, see $out.log"
    continue
  fi
  echo "$(date '+%F %T') $name: python -m mdagent run $COMMON ${RUN[$name]} --out $out"
  python -m mdagent run $COMMON ${RUN[$name]} --out "$out" > "$out.log" 2>&1 || echo "$name: ended with an error, see $out.log"
done

{
  echo "Step 5 summary for $SYSTEM, $(date '+%F %T')"
  for name in "${ORDER[@]}"; do
    f=${P}_$name/run.json
    [ -e "$f" ] && python - "$f" "$name" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
o = [s.get("notes", {}).get("plain_md_outcome"), (s.get("enhanced") or {}).get("outcome"), s.get("notes", {}).get("outcome")]
print(f"{sys.argv[2]:<20}", "; ".join(f"{x.get('phase')}: {x.get('reason')} at {x.get('ns_per_replica', '?')} ns" for x in o if x))
PY
  done
  for pair in "reach_chi1 ref_reach_chi1" "reach_dist ref_reach_dist" "enhanced_chi1 ref_enhanced_chi1"; do
    set -- $pair
    if [ -e ${P}_$1/run.json ] && [ -e ${P}_$2/run.json ]; then
      echo; echo "compare $1 with its reference $2:"
      python -m mdagent compare --run ${P}_$1 --reference ${P}_$2
    fi
  done
} > ${P}_summary.txt 2>&1
cat ${P}_summary.txt
