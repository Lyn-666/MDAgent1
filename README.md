# MDAgent: molecular dynamics from one sentence

Tell MDAgent in one sentence what you want to know, for example *"What conformations do the side chains in this pocket take?"* The agent does the rest: it fetches and checks the structure, builds the solvated system, runs the simulations, analyses them as they run, and stops when the answer is reliable. You get a report in English and Chinese.

The main goal is to **avoid unnecessary simulation**. If you run too short, the result is unreliable. If you run too long, you waste days of GPU time. MDAgent starts with a few short replicas and looks at the results. Then it decides whether to extend them, add replicas, or stop. It stops as soon as the results are precise enough, which is usually long before the limit. If plain MD cannot answer the question, the agent says why and suggests a better method. It never exceeds your limits. The default limit is 10 replicas of at most 2 µs each.

## What it can answer

| Question | Example | You get |
|---|---|---|
| Side-chain conformations | Which rotamers do the side chains around the ligand take? | chi1 populations (g+, t, g−) with errors |
| A distance between two atoms | Is this salt bridge mostly formed? Does the pocket open? | "near" and "far" populations with errors |
| Backbone conformations | Which Ramachandran regions do these residues visit? Do they switch? | Basin populations, transition rates and dwell times, φ/ψ free-energy surface with errors |
| Whether a change happens | Does the Val111 side chain reach the t rotamer? | Which replica saw it, and when. The run stops once it is seen. |
| Structural stability | Does this small protein stay folded? Does the loop open? | Backbone RMSD and radius of gyration (Rg): steady state, means with errors, time spent near the start structure, RMSD–Rg free-energy surface, conformational clusters with representative structures |

You can add conformational clustering to any question: say *"also cluster the conformations"*.

**Methods.** Plain MD with independent replicas is the default. When a side chain or backbone conformation does not change for tens of ns, the agent can switch to **enhanced sampling** (well-tempered metadynamics) and reweight the results back to unbiased populations. This works for side chains and backbone, but not yet for distances. **Adaptive sampling** (optional) restarts from the least-visited states. It needs the extra AdaptivePy package. MDAgent does not compute binding free energies. For those, use a dedicated tool such as OpenFE or FEP+.

## Requirements

- Linux with an NVIDIA GPU and conda. macOS works without GPU acceleration, so it is only good for short tests. On Windows, use WSL2.
- [Claude Code](https://claude.com/claude-code) (run `claude` and log in) or Codex (run `codex login`). The chat uses your own login. You do not need an API key.

## Install (about 5 minutes)

```bash
git clone https://github.com/Lyn-666/MDAgent1.git && cd MDAgent1
```

```bash
conda env create -f environment.yml && conda activate mdagent && python -m pip install -e .
```

Check that the GPU works. The CUDA line of the output should say "Successfully computed forces".

```bash
python -m openmm.testInstallation
```

Optional, for adaptive sampling:

```bash
python -m pip install adaptivepy-sampling
```

## Quick start

```bash
conda activate mdagent && mdagent start
```

1. Choose the interface language, the language model (Claude or ChatGPT), and a web or terminal chat. For the web chat, open `http://localhost:8765`.
2. Say what you want, for example: *"Download 1UBQ (reference sequence UniProt P0CG48, keep charged termini) and look at the backbone conformations of A:8 to A:10."* The part in brackets is specific to 1UBQ, whose PDB file lists an obsolete UniProt entry. Your own system usually does not need it.
3. The agent checks the structure first. If it finds a problem, it shows you the report and asks what to do. Examples: duplicate chains, or a sequence that does not match the reference. Answer in one sentence, for example *"keep chain A only"* or *"continue anyway"*.
4. The agent then shows a card with its understanding of the task, all settings, and the command. Nothing runs until you click the button, or type `y` in the terminal.

The simulation runs in the background, and closing the chat does not stop it. The agent reports its progress and decisions in the chat, and tells you where the report is when it finishes.

To check that everything works before a real run, add *"development test"* to your sentence. The run then takes only a few minutes. Example: *"Download 181L, look at the side chain of A:111, development test, continue despite structure problems."*

## How to ask

You must give two things:

1. **The system:** a PDB ID (the agent downloads it) or the path to a PDB file.
2. **The question:** which side chains (for example *"around ligand BNZ"* or *"A:99 to A:120"*), which atom pair, which residues' backbone, or which change to watch for. Use the residue numbers from the PDB file.

The agent decides everything else. If your sentence is unclear, for example without residue numbers, it asks you instead of guessing. To run your own fixed plan, give exact numbers, for example *"plain MD only, exactly 3 replicas of 10 ns"*. The agent then follows your plan and only makes suggestions.

**Optional settings:**

| Setting | Example | Default |
|---|---|---|
| Precision | "error at most 5 %" | 0.05 (populations) |
| Limits | "at most 4 replicas of 500 ns" | 10 replicas × 2 µs (stops early when done) |
| Method | "plain MD only", "use adaptive sampling" | Agent's choice. It asks before switching method. |
| Permission to switch method | "switch to enhanced sampling if plain MD fails" | Suggest only, and wait for you |
| Temperature | "at 310 K" | 300 K |
| Ions | "0.1 M KCl" | 0.15 M NaCl |
| Force field, pH, protonation | "Amber ff19SB", "pH 6", "check protonation with PROPKA first" | CHARMM36m, pH 7 |
| Save location | "save results to /data/me/md" | `runs/<run name>/` |
| Stability precision | "RMSD error at most 0.03 nm" | 0.02 nm |
| "Left the start structure" | "RMSD above 0.25 nm counts as unfolded" | 0.3 nm |
| Clustering cutoff | "cluster cutoff 0.2 nm" | 0.15 nm |

**Examples to adapt:**

- *Side chains:* "Download 3HTB and look at the side chains within 5 Å of ligand JZ4. Switch to enhanced sampling if plain MD fails."
- *Salt bridge:* "Download 1UBQ (reference UniProt P0CG48, charged termini). Look at the distance between Lys11 NZ and Glu34 CD; within 0.45 nm counts as formed. Error at most 5 %."
- *Backbone:* "Use my file /data/me/protein.pdb. Estimate the Ramachandran basin populations, transitions and φ/ψ free-energy surface of A:45 to A:48. At most 6 replicas of 200 ns."
- *Stability:* "Download 1YRF and check the stability of the whole protein: backbone RMSD, radius of gyration, free-energy surface and clusters. At most 3 replicas of 200 ns."
- *Wait for a change:* "Download 181L. Stop once the Val111 side chain reaches t and stays for 100 ps. At most 20 ns per replica; continue despite structure problems."
- *Fixed plan:* "Download 1UBQ (reference UniProt P0CG48, charged termini) and look at the side chain of A:26. Plain MD only, exactly 3 replicas of 10 ns."

## How the agent saves compute

- **It checks as it runs.** After each segment, the agent computes every result and its error. The error estimate accounts for correlation between frames, so many frames do not fake precision.
- **It stops when done.** It stops when every result reaches your precision and every replica has switched states often enough.
- **It picks the cheaper fix: extend or add replicas.** If the replicas disagree, extending them does not help, so the agent adds replicas. Otherwise it extends the existing ones, because each new replica needs its own equilibration.
- **It drops the start that has not equilibrated.** Automatic detection removes the initial relaxation from the crystal structure. It does this only when the data clearly support the cut.
- **It gives up early when it must.** If a change is too slow to observe within your limits, the agent stops and reports *insufficient evidence*. It makes this decision only when even the fastest rate the data allow would not fit. It then suggests another method or larger limits.
- **It does not wait for items that never change.** If all other results are done and only items that never changed remain, the agent stops plain MD. It then suggests enhanced sampling for those items, or switches to it if you allowed that.
- **Your limits are ceilings.** The agent may use less, but never more. Extending past the limits or switching method needs your confirmation.

This is the default behaviour. With a fixed plan, adaptive sampling, or a "watch for a change" goal, the agent still checks every segment, but it only suggests extending or adding replicas.

**Insufficient evidence is an answer too.** It means your limits do not give enough data for the precision you asked for. The report says why and what to try next.

**Stability counts as done** when RMSD and Rg no longer drift and their means are precise enough. "No drift" means the second half of each replica agrees with the first half. A structure that never left its start state might still leave later. For that case, the report gives an upper bound, for example *"at most 0.01 departures per µs"*. Only longer runs lower that bound.

**Biased results are corrected.** Enhanced sampling is reweighted (Tiwary–Parrinello) and adaptive sampling goes through a Markov state model before populations are reported. Rates come only from plain MD, because the bias changes the dynamics.

## How long a run takes

Replicas run one after another on one GPU. So the run time is the total simulated time divided by your GPU's speed. For example, 1UBQ (17 k atoms) runs at about 850 ns/day on an RTX 4090, and 181L (35 k atoms) at about 470 ns/day on a consumer GPU.

- Fast questions, such as surface side chains and most backbone conformations, usually finish within hours to a day.
- Slow changes can take much longer. The default limit of 10 × 2 µs can mean weeks in the worst case.
- **For a new system, start small.** Add *"at most 3 replicas of 100 ns"*, which is less than a day on a recent GPU. Later you can say *"extend run1 to 500 ns"* or *"add 2 replicas to run1"*.

## During a run

Talk to the agent in the same chat: *"How far is run1?"*, *"stop run1"*, *"continue run1 with enhanced sampling"*, *"add 2 replicas to run1"*, *"extend run1 to 50 ns"*, *"where are run1's trajectories?"*. Starting, stopping and continuing always show a card first.

Terminal commands that do not call the language model: `/runs` lists the runs, `/report` shows the latest report, `/last` shows the previous reply, `/new` starts a new chat, and `/quit` leaves the chat (simulations keep running).

## What you get

Each run has a folder, `runs/<run name>/` by default. The report is `report.en.md` (Chinese: `report.zh.md`). It covers:

- the question and the precision you asked for;
- what was simulated, and the total compute used;
- every result with its error, and whether it met your precision;
- each decision the agent made, and why it stopped;
- the limitations and the open questions.

The folder also holds:

- the trajectories `replica_<i>_part<k>.dcd`, with protein atoms only; use `protein.pdb` as the topology;
- the backbone free-energy plots `landscape_*.svg`;
- the stability results in `stability/` (`rmsd_rg.svg`, `fes_rmsd_rg.svg` and data tables);
- the clusters in `clusters/`, with one representative structure `cluster_<k>.pdb` per cluster for PyMOL or VMD.

The files stay on the computer that ran the simulation. In the web chat, the run card shows the computer name and full path, with download links.

| Stop reason | Meaning |
|---|---|
| Target reached | Every result met your precision. You can quote the results for this force field and these conditions. |
| Insufficient evidence / stalled | This method cannot give a reliable answer within your limits. The report suggests the next step. |
| Not reachable | Things change, but at this rate the time needed is far beyond your limits. |
| Budget used up | The limits ran out before the precision was reached. |
| Failed | The error is in the report and the chat. The run can continue after the fix. |

## How the system is built

You do not need to set any of this, but you should know what happens:

- **Structure check.** The agent compares the structure with the UniProt reference sequence. It looks for tags, duplicate chains, metal ions, missing segments and chain breaks. If it finds a problem, it stops and asks you.
- **Ligands are removed.** The built-in force fields have no ligand parameters. To keep a ligand, prepare a solvated Amber system yourself with tleap. Then give the paths of the `.prmtop` and `.inpcrd` files, and the agent uses them as they are.
- **Defaults.** CHARMM36m with TIP3P water, pH 7, 0.15 M NaCl, a cubic box with at least 1.0 nm of padding, and 300 K. Each replica is equilibrated for 600 ps, with backbone restraints released step by step. A chain end gets a neutral cap if the real protein continues past it. Otherwise it stays charged.

You can change any of these choices in your sentence. If you ask for a value that is already the default, the agent tells you so.

## Troubleshooting

**`No module named 'mdagent'` when running `mdagent start`.** The project folder was moved or cloned again. Reinstall in the current folder:

```bash
conda activate mdagent && python -m pip install -e .
```

**`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.** The CUDA packages are newer than your driver. Run `nvidia-smi`, read the CUDA version in the first line (for example 13.3), and run this command with your version:

```bash
conda install -c conda-forge cuda-version=13.3 --solver=libmamba
```

**conda hangs at "Collecting package metadata".** Add `--solver=libmamba` to `conda env create`.

**"Defaulting to user installation" and the install fails.** Use `python -m pip`, not `pip`.

**Random "Segmentation fault".** Some CPU cores may be unstable, which is a known problem on some Intel 13th and 14th generation CPUs. This script finds the bad cores and prints a `taskset -c ...` command that avoids them:

```bash
python scripts/find_bad_cores.py
```

**Scrolling in tmux scrolls the input history.** Use `/last`, or run `tmux set -g mouse on`.

**The web page did not change after an update.** Press Ctrl+C in the chat terminal and run `mdagent start` again. This stops only the chat, not the simulations.

**Does it use a lot of Claude or ChatGPT quota?** No. Each sentence calls the language model once. The model only translates your sentence into settings. It cannot see your files or run commands. All simulation and analysis is done by MDAgent itself.

**I prefer the command line.** Every confirmation card shows the full `mdagent run ...` command. See `mdagent run --help` and [docs/developer.md](docs/developer.md) for all options.

## Feedback

Open a GitHub issue with the "Problem report" template. Include your sentence and the log `runs/<run name>.log`. Its first line gives the MDAgent version. To join the test, see [TESTING.md](TESTING.md).

## For developers

- [docs/developer.md](docs/developer.md): all options, algorithms, budget comparisons, experiment A and validation.
- [docs/architecture.md](docs/architecture.md): architecture, evaluation plan and revisions.
- Tests: `pytest -q tests -m "not slow"` (about 3 minutes); `pytest -q tests` (includes real MD on the CPU, about 15 minutes).

## License

MIT, see `LICENSE`.
