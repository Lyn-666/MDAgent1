#!/usr/bin/env bash
# Step 5: check that the agent judges correctly on real simulations, not only that it runs.
#
# Scenarios (181L, T4 lysozyme L99A; the same plan works for any protein, see README "第 5 步"):
#   reach_chi1      exposed side chains (Arg80, Lys135, Lys162): plain MD should reach the goal and stop.
#   reach_dist      the C-terminal salt bridge Lys162-Asp159: should reach the goal and stop.
#   trapped_chi1    Val111, plain MD only (guided): every replica stays in one state -> "stalled", not "done".
#   trapped_dist    the salt bridge Arg145-Glu11: likely never breaks -> "stalled" (cannot be told apart from
#                   a single stable state; the report must say so).
#   budget_chi1     exposed side chains, precision 0.01 in 2 ns, no permission to extend -> "budget used up".
#   extend_chi1     the same with permission to extend -> runs longer, at most 2 x the budget.
#   enhanced_chi1   Val111, auto: switches to enhanced sampling and stops when the precision is reached.
#   ref_*           longer independent runs (another seed, no early stop) to check the early-stopped results:
#                   "mdagent compare" must say they agree within their errors.
#
# Usage (from the MDAgent1 folder, mdagent environment active):
#   bash scripts/step5_validation.sh                       # all scenarios, one after another (about a night)
#   bash scripts/step5_validation.sh reach_chi1 ref_reach_chi1   # only some
# Results: runs/val_<name>/report.en.md and report.zh.md; comparisons in runs/val_summary.txt.
set -u
cd "$(dirname "$0")/.."
PDB=examples/181L.pdb
COMMON="--pdb $PDB --accept-problems --replicas 3"
EXPOSED="A:80,A:135,A:162"

declare -A RUN=(
  [reach_chi1]="--residues $EXPOSED --first-ns 1 --extend-ns 2 --max-ns 10"
  [ref_reach_chi1]="--residues $EXPOSED --first-ns 10 --max-ns 10 --allow none --seed 7"
  [reach_dist]="--distances A:162:NZ-A:159:CG@0.45 --first-ns 1 --extend-ns 2 --max-ns 10"
  [ref_reach_dist]="--distances A:162:NZ-A:159:CG@0.45 --first-ns 10 --max-ns 10 --allow none --seed 7"
  [trapped_chi1]="--residues A:111 --mode guided --first-ns 1 --extend-ns 2 --max-ns 6 --no-flip-patience-ns 3"
  [trapped_dist]="--distances A:145:CZ-A:11:CD@0.5 --first-ns 1 --extend-ns 2 --max-ns 6 --no-flip-patience-ns 3"
  [budget_chi1]="--residues $EXPOSED --precision 0.01 --first-ns 1 --extend-ns 1 --max-ns 2 --allow switch_method,stop_early"
  [extend_chi1]="--residues $EXPOSED --precision 0.01 --first-ns 1 --extend-ns 1 --max-ns 2 --allow extend,switch_method,stop_early"
  [enhanced_chi1]="--residues A:111 --first-ns 1 --extend-ns 2 --max-ns 3 --no-flip-patience-ns 3 --enhanced-ns 20"
  [ref_enhanced_chi1]="--residues A:111 --first-ns 1 --extend-ns 2 --max-ns 3 --no-flip-patience-ns 3 --enhanced-ns 20 --allow switch_method --seed 7"
)
ORDER=(reach_chi1 ref_reach_chi1 reach_dist ref_reach_dist trapped_chi1 trapped_dist budget_chi1 extend_chi1
       enhanced_chi1 ref_enhanced_chi1)
[ $# -gt 0 ] && ORDER=("$@")

for name in "${ORDER[@]}"; do
  out=runs/val_$name
  if [ -e "$out/report.en.md" ]; then echo "$name: already done"; continue; fi
  echo "$(date '+%F %T') $name: python -m mdagent run $COMMON ${RUN[$name]} --out $out"
  python -m mdagent run $COMMON ${RUN[$name]} --out "$out" > "$out.log" 2>&1 || echo "$name: ended with an error, see $out.log"
done

{
  echo "Step 5 summary, $(date '+%F %T')"
  for name in "${ORDER[@]}"; do
    f=runs/val_$name/run.json
    [ -e "$f" ] && python - "$f" "$name" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
o = [s.get("notes", {}).get("plain_md_outcome"), (s.get("enhanced") or {}).get("outcome"), s.get("notes", {}).get("outcome")]
print(f"{sys.argv[2]:<20}", "; ".join(f"{x.get('phase')}: {x.get('reason')} at {x.get('ns_per_replica', '?')} ns" for x in o if x))
PY
  done
  for pair in "reach_chi1 ref_reach_chi1" "reach_dist ref_reach_dist" "enhanced_chi1 ref_enhanced_chi1"; do
    set -- $pair
    if [ -e runs/val_$1/run.json ] && [ -e runs/val_$2/run.json ]; then
      echo; echo "compare $1 with its reference $2:"
      python -m mdagent compare --run runs/val_$1 --reference runs/val_$2
    fi
  done
} > runs/val_summary.txt 2>&1
cat runs/val_summary.txt
