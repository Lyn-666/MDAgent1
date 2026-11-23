# MDAgent as a general scientific MD agent: assessment, architecture, plan

Status: 2026-11-23 (section 3.4: revisions after an internal review). Phase 1 (the general decision core) and phase 2 (OpenMM adapters and shadow mode) are
implemented. The core now decides next to every run without acting (shadow mode); the run itself still
follows `policy.py` and `agent.run_all`. Phase 2 ran no new simulation.

Ubiquitin (1UBQ) and T4 lysozyme L99A (181L) are development and evaluation cases. The agent must not
contain assumptions about them, and nothing in the new core does (a test checks this).

## 1. Assessment of the current repository

### 1.1 What is sound and should be kept

- **The observable layer works on states.** Every quantity becomes per-frame states (`diagnose.assign_states`,
  `observables.two_state`). One statistical machinery then handles all of them. This is the right seam
  for generality.
- **The uncertainty code is careful.** It uses block averaging with the largest error over block sizes,
  combines replicas as max(within, between), and applies error floors from the number of transitions
  or the Kish effective sample size (`uncertainty.py`). The Markov state model of seeded segments
  (`msm.py`) and Tiwary–Parrinello reweighting with an effective-sample check (`reweight.py`) are both
  tested against known answers.
- **The stop reasons are kept apart.** They are target_reached, stalled, not_reachable, budget_exhausted
  and failed. A stable result is never itself a reason to stop (`reweight.assess`).
- **Permissions exist.** They are extend, add_replicas, switch_method and stop_early, with mode defaults
  that follow the user's decision of 2026-11-20 (`agent.DEFAULT_PERMISSIONS`). Actions without permission
  become suggestions.
- **Simulation and checkpoints are separate.** Resumable segments (`simulate.run_segment`) and
  metadynamics replicas (`enhanced.run_replica`) are isolated functions. They can become tools without
  a rewrite.

### 1.2 Where the design is tied to one observable, method or heuristic

| Finding | Where | Effect |
|---|---|---|
| A method switch exists only for chi1, and only to chi1 metadynamics | `agent.py:95-98` (`can_switch`), `agent.py:346-348` (`bias_available=quantity == "chi1_populations"`), `agent.py:525-701` | There is no tool registry. Each new method or observable needs new branches in `agent.py`. |
| Two separate decision procedures with different rules | `policy.plain_md_next` (plain MD); a loop inside `agent.run_all` (enhanced sampling: 3 stalled checks, add replicas only in the first half, extend cap) | The same reasoning is written twice, with different constants. They can drift apart. |
| The evidence carrier is shaped around side chains | `diagnose.ResidueReport` (chain, resseq, resname, "flips"); distances are put in as `resseq=k` | Verdict strings (`no_flip`, `rare_flips`, ...) are the interface between analysis and decisions. |
| One precision for every estimand | `Goal.precision` | Different estimands in one task cannot have different success criteria. |
| Fixed numbers act as general thresholds | `LoopSettings.no_flip_patience_ns=10`, `min_flips_per_replica=4` (chi1 and distances alike), `StopRule` (min_ns 5, min_transitions 10, min_effective_samples 50), `NO_GAIN_MARGIN=1.5`, the 0.5 × budget limit for adding replicas, `extend_factor=2`, `combine_replicas(min_diff=0.05)`, `SAME_WITHIN=0.01`, `hold_ps=100` | An absolute patience time in ns is specific to the system. The other numbers are fair defaults but sit in the policy, not with the estimator that needs them. |
| Decision logic and messages are mixed | bilingual strings inside the rules of `policy.py` and `agent.py` | The rules are hard to test alone and hard to extend. |
| The decision record is prose | `decisions.log`, `notes.actions` (action, reason) | The record does not keep the evidence snapshot, the alternatives that were considered, or the hypotheses. A reader cannot reproduce a decision. |
| Budgets are split by method | `--max-ns` for plain MD and `--enhanced-ns` for enhanced sampling | In step 5, `budget_chi1` used 2 ns of plain MD and then the default 20 ns of enhanced sampling. This breaks "the user's numbers are limits". |
| One system-specific fact in the planning prompt | `chat.py:152` (rate of T4L Val111) | This is harmless as a literature example, but it belongs in a cited method guide, not in the default prompt. |
| Simulation backend | OpenMM only (`simulate.py`, `enhanced.py`) | This is acceptable. The tool interface keeps it replaceable. |

No 1UBQ- or 181L-specific values appear in the decision code. They appear only in examples, help text
and `scripts/step5_validation.sh`.

### 1.3 Scientific issues found in the audit

1. **1UBQ is no longer a clean evaluation case.** Three rules were changed after looking at 1UBQ runs:
   the replica-adding rule, the error floor, and the `compare` tolerance (CLAUDE.md, 11-19 and 11-20).
   1UBQ and 181L are therefore calibration systems. A held-out set must be chosen and frozen before
   any further tuning (section 5).
2. **Repeated looks.** The agent checks "error ≤ target" after every round and stops at the first pass.
   With noisy error estimates, this favors stopping when the error happens to be underestimated. The
   new core has a per-requirement guard (`confirm_looks`). Its default stays 1 until calibration shows
   whether 2 is needed.
3. **Block-averaging errors are noisy at short lengths.** In a synthetic trace, the ratio of error to
   target went 1.59, then 1.97, then 1.45 over three looks. A single pessimistic look made the
   projection call the goal unreachable, although a later look met it. The core now needs two looks in
   a row before it calls a goal unreachable. A steadier error estimator (a fitted plateau, or an
   autocorrelation-time estimate) is a later item.
4. **Zero events cannot separate a trap from a single stable state.** The current code says so in
   its messages. The new core keeps this as two competing hypotheses and marks the estimand
   "indeterminate". Such an estimand is never reported as converged.
5. **An event seen under bias proves nothing.** A bias can push the system into any state. The new
   registry has no estimator for (event, biased). An event goal therefore cannot switch to
   metadynamics by itself.

## 2. Target architecture

```
          user words ──► Scientific Task Interpreter (llm.py / chat.py → TaskSpec)
                                   │  estimands, requirements, budget, permissions, mode
                                   ▼
 ┌──────────────────────────── general decision loop (mdagent/core) ────────────────────────────┐
 │ observe ─► assess ─────────► hypothesize ─────► propose + select ─────► record ─► execute     │
 │  (tool)   (estimator by      (reasoner.py)      (controller.py:         (ledger.py, (only     │
 │           kind × regime)                         value, cost,            decisions   authorized│
 │                                                  permissions, budget)    .jsonl)     actions)  │
 └───────▲───────────────▲──────────────────────────────────────────────────────────────────────┘
         │               │
  Tool registry    Estimator registry               ◄── task-specific (mdagent/*.py)
  plain MD,        unbiased populations, reweighted populations,
  metadynamics,    Markov-model populations, first-entry events, ...
  adaptive,        featurizers: chi1 states, distance states, ...
  synthetic
                                   ▼
                     Reproducible reporting (report.py reads decisions.jsonl)
```

| Required component | Module | Kind |
|---|---|---|
| Scientific Task Interpreter | `core/task.py` (`TaskSpec`, `Requirement`, `Budget`); `llm.py` and `chat.py` fill it later | general |
| Observable / estimator registry | `core/registry.py` (interface, lookup by kind × regime); `estimators.py` (implementations) | interface general; estimators task-specific |
| Simulation and analysis tool registry | `core/registry.py` (interface); `synthetic.py` now; OpenMM adapters in phase 2 | interface general; tools task-specific |
| Evidence and uncertainty assessment | `core/evidence.py` (`Assessment`, `Check`, shared signal names); numbers from `uncertainty.py`, `msm.py`, `reweight.py` | carrier general; numbers task-specific |
| Hypothesis-driven reasoner | `core/reasoner.py` | general |
| Budget- and permission-constrained controller | `core/controller.py`, guards in `core/loop.py` | general |
| Reproducible reporting | `core/ledger.py` (`decisions.jsonl`); `report.py` to read it in phase 3 | general record; task-specific wording |

### 2.1 What separates general reasoning from task-specific numbers

- The core imports only the standard library. It does not name any observable, engine or test system.
  `tests/test_core_architecture.py` checks both rules.
- **Success criteria belong to the estimand.** Each `Requirement` carries its own params, for example
  `{"max_se": 0.05}` or `{"target": "b", "hold_frames": 50, "seen_in": 1}`. The estimator returns
  met, not_met or indeterminate. The core never compares an estimate with a threshold. It only adds one
  guard: "met" counts only when every blocking check passed.
- The reasoner reads a small shared vocabulary of signals (`evidence.SIGNALS`): events seen and
  needed, uncertainty ratio and its scaling, error within and between replicas, replica disagreement,
  whether projection is meaningful, and the data regime. An estimator decides what an "event" is
  (a state change, an effective sample, an entry into a target state).
- Policy constants that remain in the core are about decisions, not convergence: the unreachable
  margin (1.5), the plausibility weights and the action-effect table. All are listed in section 6 and
  are marked as heuristics in every record.

### 2.2 The reasoning protocol at every decision point

1. **Estimand and requirement** come from `TaskSpec.requirements`.
2. **Evidence and validity:** the loop takes, for each estimand, the newest tool of the run that serves
   its kind. The registry gives the estimator valid for that tool's data regime. The estimator returns
   an estimate, uncertainty, validity checks and signals.
3. **Unresolved uncertainty and failure mechanisms:** open estimands are those not met at
   `confirm_looks` consecutive looks.
4. **Competing hypotheses** (all labeled heuristic): slow_mixing, replica_dependence, partial_trapping,
   trapped_or_single_state against not_yet_observed, estimator_invalid, bias_ineffective,
   requirement_unreachable, awaiting_confirmation and unexplained. With zero events, trap plausibility
   uses the 95 % upper bound of a Poisson rate (3 / exposure). It asks whether even that rate could give
   the needed events in the remaining budget. This replaces the fixed patience time.
5. **Candidate actions** are continue, extend_budget, add_replicas, switch_method and stop. Each has a
   cost in budget units and a value: the sum over open estimands of plausibility × effect.
6. **Selection:** the allowed candidate with the highest value. Ties go to the lower cost, then to a
   fixed order. If no allowed action has value, the agent follows the plan when stopping early is not
   allowed, and otherwise stops with a reason. Better actions that are not allowed become suggestions,
   with what the user must allow.
7. **Execution:** only authorized actions run. Cost and replica limits are checked again after every
   action, and a tool that overruns stops the run with an error.
8. **Record:** each decision goes to `decisions.jsonl` before the action runs. The record holds the
   progress, every assessment, the hypotheses, every candidate with cost, value and authorization, the
   choice, the stop reason and the suggestions. Records are deterministic for given inputs.

Modes: **guided** follows the plan and never switches method, even with the permission. **auto** acts
within the granted permissions. The default is stop_early only, as the user decided on 2026-11-20.

## 3. What phase 1 delivered (this change)

New files: `mdagent/core/{task,evidence,registry,reasoner,controller,ledger,loop}.py`,
`mdagent/estimators.py`, `mdagent/synthetic.py`, and three test files.

New tests (35, all deterministic, about 7 s):

- **Simulation execution** (`test_core_loop.py`): a synthetic Markov-chain tool runs replicas. The
  tests check that a failed run resumes and ends like an uninterrupted one, that a finished run does
  not run again, and that the same inputs give byte-identical ledgers.
- **Uncertainty assessment** (`test_estimators.py`): error bars cover the truth at about the nominal
  rate (at least 25 of 30 seeds). Zero events give "indeterminate", never "met". Replicas trapped in
  different states fail the consistency check. Reweighting recovers the true populations, and
  degenerate weights block the result. The Markov model corrects segments started from chosen states.
  Event detection needs an entry and a hold. The registry picks the estimator for the data regime.
- **Permissions** (`test_core_loop.py`): every one of the 16 permission sets is run in both modes on 4
  systems. Only authorized actions run. Spending stays within the cap, and replicas within their limit.
  Guided mode never switches. Without stop_early, the run never stops before the plan's end, except
  when the target is reached at the end. A switch target must be listed as allowed.
- **Decision safety:** "target_reached" appears only when every assessment is met with all blocking
  checks passed. A trapped system is never reported as converged. Each stop reason (target_reached,
  stalled, not_reachable, budget_exhausted, failed) has its own test, and the repeated-look guard is
  tested.
- **Architecture:** the core imports only the standard library and names no observable, engine or system.
- **Against a fixed budget:** on a fast synthetic system the agent never spends more than a fixed-budget
  run, spends less in at least 8 of 10 seeds, and its 2-SE interval covers the truth in at least 8.

### 3.1 Phase 2 (OpenMM adapters and shadow mode)

- `mdagent/tools_openmm.py`:
  - **Featurizers:** chi1 states, distance states, and event items. `event_series` is now shared with
    `agent._diagnose_events`, so the old and new decisions read the same states.
  - **`PlainMDTool`** wraps `run_segment`. New replicas catch up first, then a round runs. A slow CPU
    test runs this with real MD.
  - **`MetadTool`** wraps `run_replica`. Its `observe` rebuilds the bias with `reweight.replay` and
    returns per-frame weights.
  - **Views of earlier moments:** `PlainMDTool(parts=k, n_replicas=n)` and `MetadTool(until_ns=d)`. The
    bias rebuild only uses earlier samples, so a cut-off view gives the same weights as the run had at
    that time. A test checks this.
  - **Limit:** the adaptive-sampling tool only observes for now. Its rounds are still run by
    `agent.run_loop`.
- `mdagent/shadow.py`:
  - **Task mapping:** `task_from_state` turns a run into a task. The precision becomes each estimand's
    requirement. The budget is the first plain MD limit times the first number of replicas, plus the
    enhanced sampling length times its replicas when switching is allowed. This is one shared budget.
  - **Live record:** `record` writes `decisions.core.jsonl` after every plain MD round and every
    enhanced-sampling check. It never raises. The environment variable `MDAGENT_SHADOW=0` switches it off.
  - **Replay:** `replay` rebuilds every past decision point of a finished run from its files.
  - **Comparison:** `compare` sets each old action next to the core's choice, with the strongest
    hypothesis of every open estimand. The categories are same, other_stop_reason,
    core_stops_earlier, core_goes_on and different.
- `mdagent shadow --run <runs...> [--live]` prints the comparison.
- The core interface changed in one place: `Tool.supports(kind, estimand)`. Metadynamics serves only
  chi1, and only the biased side chains. More of a tool now gives value only to the estimands that tool
  serves. Before this fix, a side chain that was not biased still counted as a reason to continue
  metadynamics.
- 11 new tests (`tests/test_shadow.py`; one is slow and runs real MD on the CPU).

### 3.2 First new observable through the registries: backbone phi/psi (2026-11-21)

A user asked for the Ramachandran basin populations, transition statistics and phi/psi free-energy landscape of
three residues, with the agent choosing replicas, length and stopping within 8 replicas × 50 ns. This was the
first test of whether the architecture is general. What was needed:

- **Task-specific code only:**
  - `backbone.py`: basins as data with a documented default; core-set assignment; rates with Poisson
    errors; dwell times as time / exits; a 2D landscape with errors from the replica spread.
  - one featurizer in `tools_openmm.py`.
  - two estimators in `estimators.py`: `transition_rates` and `free_energy_surface`.
- **Nothing in `mdagent/core/` changed.** Each residue becomes three estimands with their own criteria:
  population SE, worst relative error of the leading rates, and worst F error where F ≤ 2 kcal/mol.
- **Calibration of the errors, checked over 30 seeds:** rate errors cover the truth at about the nominal
  rate. The first dwell-time estimator (mean of complete visits) was biased short by about 0.5 standard
  errors, because it leaves out the long visits that are cut off at the trajectory ends. It was replaced by
  the maximum-likelihood estimate time / exits.
- **Current run path:** the old policy now judges a backbone residue "ok" only when all three criteria
  hold.
- **Planning:** the chat prompt now has a rule. When the user leaves replicas, length and stopping to the
  agent, the agent proposes a plan within the user's limits and states it: start with 3 replicas (at least
  2, so that the replica spread can be measured), allow add_replicas up to the user's limit, use the user's
  length as the limit, and allow stop_early. This is a planning default, not a decision rule. The stopping
  itself comes from the estimators.

**Backbone enhanced sampling (2026-11-21).** The plan was well-tempered metadynamics on phi and psi, and
the existing chi1 machinery already supports any number of one-dimensional dihedral biases. So the change
was small:
- `enhanced.dihedral_quads(kind)` chooses which dihedrals to bias (one per residue for chi1, two for the
  backbone).
- `reweight.analyze(kind="backbone")` assigns basins from the phi and psi columns and adds a reweighted
  landscape.
- `assess` gains the landscape criterion.
- The metadynamics tool serves backbone "basins" and "landscape" estimands but not "transitions", because
  biased kinetics give no valid rates.

The estimator registry already reflected this, because `transition_rates` is registered only for unbiased
data. A toy model with known basin populations (0.70/0.25/0.05) is recovered within errors.

**Open design question for phase 3,** shown by the shadow: after a switch, estimands that only the old
tool serves stay open. Example: a side chain that was not biased. The agent must either run both tools,
switch back, or report these estimands as resolved by the first method. The current policy reports
them from plain MD.

### 3.3 Merge with the other development line (2026-11-22)

A second session had developed `claude/admiring-noether-6mlzki` from the same base. Its additions:

- a research goal of accuracy-constrained budget optimization;
- `allocation.py`: the agent chooses replicas, lengths and the stop, with Bonferroni-calibrated looks;
- `cost.py`;
- `budget.py` and `experiment_a.py`: replay benchmarks against fixed plans and Sequential-CI, with an
  independent reference. These implement much of the evaluation plan in section 5.
- `uncertainty.Precision`: standard error, confidence and confidence-sequence criteria;
- `basins.py`: `rama:` and `turn:` items.

It was merged into this line. The user decided three points:

1. Ramachandran basins default to regions that cover the whole map. The three centres stay available as
   `--rama-basins centres`. Runs started before the merge keep the definition they began with.
2. When the user leaves replicas, length and stopping to the agent, the run uses projection allocation.
3. Projection runs use the look-calibrated confidence interval; rounds runs keep the standard error.

One backbone engine remains:

- `basins.BasinSet` defines the items and their basins.
- `backbone.diagnose_backbone` computes populations, all-pair rates and dwell times, the landscape with
  errors, and the criteria.
- `basins.summarize` adds the free-energy difference and the split-half check, shared with Experiment A.

The give-up rule for an item that never changes state is one function,
`uncertainty.never_changing_hopeless`, used by both run modes and mirrored in `core/reasoner.py`. The
changes needed come from adequacy and, when known, from the precision. It has no fixed waiting time; the
other line had a 20 ns floor.

Problems found and fixed during the merge:

- A ubiquitin-specific statement had been hard-coded in user advice. It was removed.
- Projection allocation had left out items handed to enhanced sampling even when enhanced sampling was off.
- The new `transition_rates` and `free_energy_surface` estimators keep their own criteria. The new core is
  still in shadow mode.

### 3.4 Revisions after an internal review (2026-11-23)

A review written as for a top venue found that the main claim (the least compute for a given precision) had no
real-MD evidence yet. It also found that the benchmark would overstate reliability. The order of work was:
first repair the measurement, then spend GPU time. These changes need no GPU.

**Which agent a paper evaluates.** It is `allocation.next_step`: the projection allocation of live runs
(`--allocation projection`), replayed in Experiment A. The other three decision paths are not the subject of
a paper:
- `policy.py`: the rounds mode;
- `core/`: shadow mode only;
- `budget._decide`: a mirror of `policy.py`.

Freeze the code with a git tag before the held-out runs, and quote the tag.

**What the guarantee is.** The intervals are calibrated per estimand, that is, per item and state. Over the
looks that may stop, they use Bonferroni. The claim to test is this: at the moment a method stops, each item's
interval holds the reference in at least 1 − alpha of independent trials (`per_item`, `coverage_at_stop`).
Experiment A reports separately whether all items are right at once. This is the old "success" column.

Changes:

| Review point | Change | Where |
|---|---|---|
| Trials share data | Three modes, and the report states the number of independent units. Fresh pools: each trial gets a new model pool, and the reference is the truth. Disjoint groups: trials share no replica. Shared pool: allowed, but the intervals use about usable/needed units, often 1. | `experiment_a.run`, `summarize._interval` |
| Normal factor with 2 to 4 replicas | Student's t with n − 1 degrees of freedom where the spread between replicas is real by the chi-square test that the allocation already used (`uncertainty.replica_spread_real`). A spread that is larger only by chance keeps z, because max(within, between) already covers it. A first version used t whenever the spread was merely larger; the plan (z) and the verdict (t) then disagreed, and on the easy family the agent ran out of looks in 23 % of trials. It is used in the benchmark verdicts, the coverage check, the allocation projection (more replicas also narrow the factor), the live verdicts with the confidence criterion, and the backbone landscape error. The standard-error criterion keeps z = 1, because it is not an interval. | `uncertainty.t_factor`, `uncertainty.replica_spread_real`, `Precision.factor`, `basins.state_factors`, `allocation.allocate(t_dist)` |
| Initial-structure bias | Automated equilibration detection (Chodera 2016), with one start t0 for all replicas (they start from the same structure). t0 maximises the pooled number of independent samples. It is used only if it raises that number at least 1.5 times and the replicas changed state at least 20 times (median); otherwise nothing is cut. Each state indicator is checked, and the latest accepted t0 is used. At most half of the series is cut. The cut part is still counted as cost, and the rules use only the kept time (`ItemState.cut_ns`, `ResidueReport.burn_in_ns`): the never-changing rule and the projection. A first version chose t0 per replica; an independent review showed that on short equilibrium series it followed the noise and shifted populations by 0.014 to 0.023. The joint rule leaves equilibrium data unbiased (shift below 0.001) and still finds the relaxing start of `biased_start` in 8 of 10 seeds. New runs with projection allocation use it (`--burn-in auto`). Older runs and the rounds mode keep everything. Every method in Experiment A uses it, so the comparison stays fair. | `uncertainty.joint_equilibration_start`, `basins.trim`, `LoopSettings.burn_in` |
| The synthetic model cannot fail | `family_pool` has four families with known truth: `equilibrium`; `biased_start` (a "crystal" substate that relaxes in about 10 ns); `two_timescale` (hidden groups that exchange about every 2 ns); `hidden_slow_start` (exchange about every 10 ns, and every replica starts on one side). | `experiment_a.FAMILIES` |
| A weak baseline | `pilot_fixed`: a pilot, one projection with the agent's own `allocate`, then a fixed plan judged once. It is now part of condition 2 of the verdict. | `experiment_a.run_pilot_fixed` |
| Constants set by judgement | `--sensitivity`: change one constant at a time (margin, final_margin, give_up_factor, agent_looks, starting replicas, min_transitions) on every family. | `experiment_a.sensitivity` |
| Two different starting replica counts | Experiment A now starts the agent with 3 replicas, the same as the chat plan of a live run. | `Design.replicas` |

**Adding replicas when the spread is only suggestive** (the user's choice of 2026-11-23, for the weakness on
`two_timescale`). The projection counts the spread between replicas as extra error when the chi-square test passes
at 0.8 (`Limits.spread_level`), not only at 0.95. With 3 replicas the spread rests on 2 degrees of freedom, so the
agent gives up only when even the optimistic spread (its 95 % lower bound, `allocation.optimistic`) cannot reach
the precision. When the point projection fails but the optimistic one does not, the agent adds replicas to measure
the spread (`explore_replicas`), and every replica runs one chunk more (the live loop runs new replicas only in a
round of positive length). This follows the same idea as the never-changing rule: give up only when even the most
favourable reading of the data says so.

**Independent code review.** A separate review of the diff found seven problems, all fixed: the per-replica
burn-in bias and the uncounted cut time (above); coverage that ignored `Design.t_dist`; condition 2 of the verdict
reported as failed when there were no baselines (it is now `None`); the live verdict used a slightly different
test for t than the plan (now the same chi-square test); `two_timescale` and `hidden_slow_start` violated detailed
balance (the rates between groups are now a flux divided by the equilibrium probability, so the chains are
reversible like equilibrium MD, with the same truth); a distance replica with no frame in a core gave NaN.

`allocation.RULE_VERSION` is now 6. A resumed projection run decides again under the new rule.

**First results on the synthetic families (2026-11-23, models only, not real MD).** The settings were the
defaults: 30 independent trials per family, each with a fresh pool of 16 × 100 ns; the reference is the known
truth; the budget is 800 ns; the tolerance is 0.05; alpha is 0.05. The commands to reproduce them:
`mdagent experiment-a --family <name> --trials 30` and `--sensitivity <names> --trials 20`.

First round (before the independent review and the replica rule):

| Family | Agent: success / false stops / median compute | Best other method |
|---|---|---|
| equilibrium | 90 % / 0 % / 242 ns | Sequential-CI with 8 replicas: 100 % / 0 % / 185 ns |
| biased_start | 93 % / 0 % / 259 ns | Sequential-CI with 4 replicas: 100 % / 0 % / 242 ns |
| two_timescale | 47 % / 3 % / 302 ns | Sequential-CI with 8 replicas: 77 % / 0 % / 505 ns |
| hidden_slow_start | 3 % / 7 % / 242 ns | all methods fail; their false stops are 3 to 10 % |

Final rules (joint burn-in, suggestive spread adds replicas, reversible families), the current state:

| Family | Agent: success / false stops / median compute | Others (success / false stops / compute) |
|---|---|---|
| equilibrium | 93 % / 0 % / 196 ns | Sequential-CI 8: 100 % / 0 % / 205 ns; Sequential-CI 4: 93 % / 0 % / 222 ns; pilot: 57 % / 0 % / 64 ns |
| biased_start | 70 % / 0 % / 290 ns | pilot: 83 % / 0 % / 302 ns; Sequential-CI 4: 70 % / 0 % / 332 ns; Sequential-CI 8: 67 % / 0 % / 645 ns |
| two_timescale | 60 % / 0 % / 476 ns | Sequential-CI 8: 83 % / 0 % / 565 ns; pilot: 47 % / 0 % / 302 ns; Sequential-CI 4: 0 % |
| hidden_slow_start | 0 % / 0 % / 330 ns | all fail; only the pilot plan stops wrongly (7 %) |

The agent makes no false stop in 120 independent trials. On `biased_start` all methods lost success when the
burn-in became joint and cautious: some replicas stay 20 to 40 ns in the start substate, a little bias remains,
and the errors at the end (0.05 to 0.067) are just above the tolerance. The methods then say INSUFFICIENT_EVIDENCE
instead of a wrong answer. This is the intended trade: the per-replica rule corrected more here but biased
equilibrium data, which no reference from a real pool could reveal.

What this shows:
- Hypothesis A is still not supported. The agent is the cheapest reliable method only on `equilibrium` (196 ns
  against 205 ns), and not by a clear margin. The old "100 % success, 32 % cheaper" came from a design that did
  not use independent trials, t or burn-in.
- The automated burn-in works. In the first runs on `biased_start` without it, Sequential-CI and the pilot
  plan never succeeded. With it, they succeed in 38 % or more of trials.
- `two_timescale` is the agent's structural weakness. It stays at 3 replicas, and in 9 of 30 trials it gives up
  because the projected cost is too high. With 3 replicas, the chi-square test for a real spread has little
  power, and t with 2 degrees of freedom makes the projection large. Sequential-CI with 8 replicas from the
  start does better.
- `hidden_slow_start` shows a limit of plain MD from one start. A run cannot see a state that no replica has
  visited. Every method sometimes stops wrongly.
- Sensitivity: in these two families, no single constant (margin, final margin, give-up factor, looks,
  starting replicas 2 to 4, minimum transitions) changes the success rate much. Their failures are therefore
  not a tuning problem. With 2 starting replicas, `biased_start` gets false stops (5 %). With 3 it gets none,
  so 3 stays.

These families are the development set, so changing the agent's rules on them is allowed. Which change to make
is for the user to decide (see the decisions below).

**Deferred (smaller points).**
- Rate errors: check whether the Poisson errors cover the truth when the system recrosses a boundary; else
  use a bootstrap over trajectory segments.
- Kish effective sample size: make it aware of time correlation for the error floor of enhanced sampling.
- Basin boundaries: test how much the populations change when the boundaries move.
- Landscape JS threshold (0.05 bits): it has no stated reason. It is not part of "success" in Experiment A.

**Decisions only the user can make.**
1. Where to publish. A computational-chemistry journal tells a statistical-design story. An AI venue then needs
   a benchmark of how well user requests are translated into settings.
2. Who chooses the held-out systems, and when the sealed list is committed.
3. What counts as a correct answer for A-Hard.
4. How to repair the agent's weakness on `two_timescale`, before any GPU run. Options: start with more
   replicas when the budget allows; add replicas when the spread between them is only suggestive, not just
   when it is significant; or accept the weakness and report it.
5. Whether to start replicas from different structures. Only that can reveal a hidden slow state
   (`hidden_slow_start`). The populations then need a correction, for example a Markov state model, which
   adds a method.
6. How large the GPU pool is. A paper needs independent trials, for example 40 × 100 ns with 8 held out and
   4 disjoint groups of 8. That is several days of GPU time.

## 4. Minimal prioritized implementation plan

Each phase is small and can be checked on its own. Phases 2 and 3 change no scientific result until the
new loop has shown the same or better behavior.

1. **Done: phase 1, the general core** with synthetic tools and estimators.
2. **Done: phase 2, adapters and shadow mode** (no change in behavior; see 3.1).
   - `tools_openmm.py`: plain MD tool (wraps `run_segment`; `advance(n, until)` = rounds; new replicas
     catch up as now), metadynamics tool (wraps `run_replica`; `observe` replays the bias with
     `reweight.replay` and returns weights, which `WeightedPopulations` uses), adaptive tool (seeded
     segments, `SegmentMSMPopulations`).
   - Featurizers: chi1 → states (`diagnose.assign_states`), distance → states (`observables.two_state`).
   - `TaskSpec.from_run_state(RunState)`: maps `Goal`, `LoopSettings` and the enhanced settings into one
     shared budget.
   - In `agent._apply_policy`, also run the core on the same evidence and write its decision to
     `decisions.core.jsonl`, without acting on it. Next: replay the step-5 runs that already exist
     (`mdagent shadow`) and review every disagreement before phase 3.
3. **Phase 3: switch over behind a flag** (`--controller core`).
   - `agent.run_all` drives `core.loop.run` with the OpenMM tools.
   - A `messages` adapter turns `Decision` and suggestions into the current bilingual chat messages.
   - `report.py` reads `decisions.jsonl` for the "why it stopped" and "actions" sections, and marks
     heuristic hypotheses as such.
   - One shared budget across methods (fixes the `budget_chi1` issue).
4. **Phase 4: remove the duplicates** once phase 3 matches or beats the old path on calibration
   systems: the rules in `policy.py` and the loop inside `agent.run_all`. Keep `diagnose.py` for the
   human-readable table.
5. **Phase 5: widen the scope** only through the registries. Examples: new featurizers for any discrete
   observable, a biasing tool for distances, and a steadier error estimator (an autocorrelation-time
   fit) registered as a new estimator.

Not planned: a universal convergence threshold, LLM-made numerical decisions, and running alchemical
free-energy calculations (still only recommended as external tools).

## 5. Independent evaluation plan

### 5.1 Systems, split three ways and fixed before evaluation

| Set | Purpose | Contents |
|---|---|---|
| Development | write and debug code | synthetic families (below); the small peptide fragments in `tests/data` |
| Calibration | choose the remaining heuristic constants (`confirm_looks`, margin, effect table) | 1UBQ and 181L, which are already used for tuning |
| Held-out benchmark | final claims only; never run during development | 4 to 6 proteins chosen by someone other than the developer, each with independent reference data (long reference MD with other seeds and at least 5 times the agent's budget, and NMR observables where they exist). The list is sealed with a commit hash before the first held-out run. |

Synthetic families with known truth (the first four are in `experiment_a.FAMILIES`; every trial gets a fresh
pool): fast mixing; slow but ergodic; fully trapped; single stable state;
partial trapping (replicas start in different basins); replica-dependent starts; a hidden slow coupled
coordinate (the bias pushes the wrong coordinate); degenerate reweighting; rare events with known
first-passage times; several estimands with different requirements in one task.

### 5.2 Baselines

- **Fixed budget:** run the whole planned budget, then estimate (`permissions=()`).
- **Conventional sequential rule:** check after every round and stop at the first look where the block
  standard error is at or below the target and every replica has at least k transitions. This is the
  usual practice and close to the current `policy.py` rule 1. A variant with two consecutive passes
  is also run.
- **Current MDAgent policy** (`policy.py` and `run_all`), as a regression baseline.
- **Pilot, then a fixed plan** (`experiment_a.run_pilot_fixed`): what a careful user does without an agent. It
  separates the value of the projection from the value of looking more than once.

All policies run on the same seeds and budgets (paired design).

### 5.3 Metrics

- **Scientific correctness:**
  - coverage of the reported 95 % intervals against truth (synthetic) or the reference (real), at the
    moment of stopping;
  - the false-convergence rate: "target_reached" while the reference disagrees by more than 2 combined
    errors;
  - stop-reason accuracy on synthetic families, where the right reason is known;
  - the rate at which trapped or single-state cases are wrongly reported as converged (must be 0).
- **Efficiency:**
  - cost at stop;
  - cost relative to a fixed budget;
  - cost to a correct answer;
  - wasted cost (spent after the evidence was already sufficient, or on actions without value).
- **Safety:** permission and budget compliance (must be 100 %), and whether ledger records alone
  reproduce every decision.
- **Generalization:** the gap between calibration and held-out results on every metric above. Results
  are reported per system and pooled, with bootstrap intervals over seeds (at least 10 seeds per
  synthetic case and 3 to 5 per real system).

### 5.4 Rules

Fix every constant before the held-out runs. Report every run, including failures. Do not change code
between held-out runs. The reference runs must be independent of the agent runs (other seeds, other
starting structures where possible).

## 6. Heuristics still in the core (to calibrate, not to trust)

| Constant | Where | Meaning |
|---|---|---|
| `UNREACHABLE_MARGIN = 1.5` | `core/reasoner.py` | Unreachable only if an optimistic projection, at two looks in a row, is over 1.5 × the limit |
| `MIN_LOOKS_FOR_PROJECTION = 2` | `core/reasoner.py` | No projection from a single look |
| Poisson 95 % upper rate, 3 / exposure | `core/reasoner.py` | Trap plausibility with zero events |
| `PLAUSIBILITY_WEIGHT`, `EFFECT` | `core/reasoner.py`, `core/controller.py` | Ranking of actions; not probabilities |
| Two replicas per add_replicas | `core/controller.py` | Step size for adding replicas |
| `confirm_looks = 1` | `core/task.py` (per requirement) | Repeated-look guard |

Estimator defaults (overridable per requirement): `min_events_per_replica = 4` (unbiased),
`min_events_per_replica_biased = 10`, `min_effective_samples = 50`, `min_events_total = 12` (Markov
model), `hold_frames = 1`, `seen_in = 1`.
