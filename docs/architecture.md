# MDAgent as a general scientific MD agent: assessment, architecture, plan

Status: 2026-11-20. Phase 1 (the general decision core) and phase 2 (OpenMM adapters and shadow mode) are
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

Synthetic families with known truth: fast mixing; slow but ergodic; fully trapped; single stable state;
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
