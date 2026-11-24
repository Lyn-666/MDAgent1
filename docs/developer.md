# MDAgent1: developer and research notes

**Defaults from 2026-11-23 (the user's decision):** `mdagent run` uses `--allocation projection` by default (the agent decides how many replicas, how long and when to stop). The limits are `--max-replicas 10` and `--max-ns 2000` (2 µs). The permissions are `stop_early,add_replicas`, and `--burn-in auto`. Only with `--mode guided`, `--sampling adaptive` or `--quantity event` is the default the fixed plan in rounds (3 replicas, at most 20 ns each). Where the text below says "default `rounds`, at most 20 ns per replica", it means this fixed plan. The developer scripts (`scripts/*.sh`) all set `--allocation rounds` explicitly.

This file is the full README from before 2026-11-23. It is kept for developers and for the paper: all options, algorithm details, the budget comparison (`budget`), Experiment A, step 5 validation and the general decision core. The user-facing guide is the [README](../README.md) on the repository home page.


An MD agent that answers one question: **given a system, a quantity to measure and a precision requirement, get an equally reliable result with the least compute.**

Before an MD run, nobody knows how to divide the time: 5 replicas of 100 ns each, 10 of 50 ns, or 1 of 500 ns? Nobody knows either whether the data at 200 ns is already enough. MDAgent analyzes while it runs. It decides where to spend the next time (extend, add independent replicas, change method) and when the data is enough to stop.

The user gives a protein PDB and a region of interest. The agent builds the system and runs MD automatically. It finds the side chains in the region that are not sampled well. Side chains that plain MD cannot sample go to enhanced sampling. The agent removes the effect of the bias potential, computes the true population of each state, and delivers the results and the stop reasons in the chat. The user can also specify the method. Then the agent follows it and only gives suggestions in the chat.

**"Cheaper" must be verifiable.** A strategy is better only when it is cheaper and its result is still correct. So the program records compute cost separately (`cost.py`). It can also compare several budget allocation strategies on the same data (the `budget` command, see "How much compute: comparing strategies (`budget`)").

**You set the limits.** The user sets the simulation time and the number of replicas as limits. The agent diagnoses after each round. It stops early when it gets the result the user wants. It also stops when more simulation gives no benefit, and then it suggests a different method (for example adaptive sampling, or enhanced sampling for side chains that plain MD cannot sample). A method change and going past the limits both need the user's confirmation.

## Quick start (for testers)

**Requirements:** a Linux computer with an NVIDIA GPU and conda. For the chat web page you also need [Claude Code](https://claude.com/claude-code) installed and logged in (run `claude` and follow the prompts) or Codex (run `codex login`). The command line alone does not need them. The program runs on macOS, but without CUDA, so it is only good for very short tests. On Windows, run it in WSL2.

**1. Install (about 5 minutes):**

```bash
git clone https://github.com/Lyn-666/MDAgent1.git && cd MDAgent1
conda env create -f environment.yml && conda activate mdagent && python -m pip install -e .
pytest -q tests -m "not slow"     # fast tests, about 2 minutes, all should pass
```

If conda stays at "Collecting package metadata" for a long time, look at `conda config --show-sources`. If it contains `solver: classic`, conda uses the slow old solver. Add `--solver=libmamba` after `conda env create` to use the fast solver for this install only. Use `python -m pip`, not plain `pip`. Plain `pip` can run the old system pip, which says "Defaulting to user installation" and fails.

After the install, run `python -m openmm.testInstallation` and check that the CUDA line says "Successfully computed forces". If CUDA reports `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, the CUDA components from conda are newer than the GPU driver. Run `nvidia-smi` and read "CUDA Version" in the first line (newer drivers write "CUDA UMD Version", for example 13.3). Then run `conda install -c conda-forge cuda-version=13.3 --solver=libmamba` with the version number you see. At startup, MDAgent also computes a small system on the selected platform. If that fails, it stops and gives the same advice.

If tests or simulations crash at random with "Segmentation fault", the CPU may have unstable cores (some Intel 13th and 14th generation CPUs have this problem). Run `python scripts/find_bad_cores.py`. It does the same calculation on each core, finds the cores that compute wrong results or crash, and gives a `taskset -c ...` command that uses only the good cores.

**2. Run a test of a few minutes:** use the included T4 lysozyme (`examples/181L.pdb`) and look at the pocket around benzene. 181L has two differences from its UniProt reference sequence, so add `--accept-problems`.

```bash
python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/first_test --accept-problems \
    --replicas 2 --first-ns 0.1 --extend-ns 0.1 --max-ns 0.3 --no-flip-patience-ns 0.25 --enhanced-ns 0.1 --equil-ps 60
```

The first line of the log gives the full path of the result folder. At the end, the Summary at the end of the log lists all result files.

**3. Run through the chat:** run `python -m mdagent start`. The program asks you in turn: the interface language, Claude or ChatGPT (Codex), and chat in the web page or the terminal. Then say in Chinese or English what to run, for example "Use 181L, look at the pocket around benzene, development test, continue even if there are problems." See "What to say in the chat" below, and "Starting a chat (`start`)" for more.

**4. Report problems:** open an Issue on GitHub, choose the "bug report" template, and attach the log file `runs/<name>.log`.

### What to say in the chat

Say what you want in one or two sentences. The language model in the chat only fills your words into a fixed settings form. It does not run anything itself. It gives you a card with all settings and the matching command. The run starts only after you click the button in the web page or type y in the terminal. So read the card before you confirm.

A good request contains the items below. You must give the first two. For the others, the defaults apply if you say nothing.

- **System:** a PDB ID (for example 1UBQ, downloaded automatically) or an existing file. For 181L, add "continue even if the structure check finds problems". For 1UBQ, say "use UniProt P0CG48 as the reference sequence, keep both termini charged".
- **What to look at:** side-chain conformations (the side chains around a ligand, or a set of residues); a distance between two atoms (which two atoms, and how close counts as "near"); or "stop when you see a certain change" (for example the Val111 side chain enters the t state).
- **Precision:** for example "error at most 5%". This means the standard error of each state population. The default is 0.05.
- **Limits:** how many replicas, and at most how many ns each. The default is 3 replicas, at most 20 ns each. The agent does not go past your limits.
- **Method:** if you name no method, the agent decides (auto mode). If you say "only plain MD" or similar, the agent follows it and only gives suggestions in the chat.
- **What the agent may do by itself:** by default it may only stop early. If you want it to extend, add replicas or switch to enhanced sampling by itself, say so explicitly. If you say "you decide the number of replicas, the length and when to stop, at most 8 replicas, at most 50 ns each", the agent gives a plan: start with 3 replicas, allow itself to add up to 8 (only when the replicas clearly disagree), at most 50 ns each, and stop when all results reach the precision or when more simulation within the limits gives no benefit. The plan is on the card and in the reply. It runs only after you confirm.
- **Development test:** say "development test", and the run takes only a few minutes.
- **Save location (optional):** for example "save the results to /data/me/md".

Examples you can copy:

| What you want | Say this |
|---|---|
| A workflow test of a few minutes | Use 181L, look at the side chains around benzene, development test, continue even if the structure check finds problems. |
| Side-chain conformations with precision and limits | Download 1UBQ, use UniProt P0CG48 as the reference sequence, keep both termini charged. Look at the side-chain conformations of A:16, A:32 and A:63, error at most 5%. 3 replicas, at most 10 ns each. |
| Allow a method change by itself | Use 181L, look at the side-chain conformation of Val111, continue even if there are problems. If plain MD cannot flip it, you may switch to enhanced sampling by yourself. |
| Distance between atoms | Use 1UBQ (UniProt P0CG48, charged termini), look at the distance between NZ of Lys11 and CD of Glu34, within 0.45 nm counts as near. 3 replicas, at most 10 ns each. |
| Stop when a change is seen | Use 181L, stop when the Val111 side chain enters the t state and stays for 100 ps. At most 20 ns per replica, continue even if there are problems. |
| Only the specified method | Use 1UBQ (UniProt P0CG48, charged termini), look at the side chain of A:26. Only plain MD, 3 replicas of 6 ns each. |
| Backbone φ/ψ, the agent makes the plan | Use 181L, estimate the Ramachandran basin populations, transition statistics and φ/ψ free-energy surface of A:110–112. You decide the number of replicas, the length and when to stop, at most 8 replicas, at most 1 µs each. Use as little compute as possible while keeping the result accurate, and continue even if there are problems. You may decide whether to switch from plain MD to enhanced sampling. |

After the run starts, ask in plain words in the same chat, for example "How far is first_test?", "Stop first_test", "Continue first_test with enhanced sampling", "Add 2 replicas to first_test", "Extend first_test to 30 ns". Start, stop and continue each show a card first and run only after you confirm.

Some notes: use the residue numbers from the PDB file. The chat does not guess numbers; if they do not match, it asks you. For distances, use standard atom names (CA, CB, NZ, OH, CD, CG1 and so on). You do not need to mention the shadow decisions of the new decision core (see "General decision core (in development)") in your request. Every run records them automatically. The chat cannot run commands. To see the comparison, run in the terminal: `python -m mdagent shadow --run runs/<run name> --live`.

## Workflow

1. **Read the target region.** You can give the residues directly (for example `A:99-120`), or give "the residues within N Å of a ligand".
2. **Build the system.** Every setting has a default, and the user can change it. See "System setup options" below.
3. **Run MD in rounds.** Several independent replicas advance together. They differ only in the random seed. By default, round 1 is 1 ns per replica, and each later round adds 5 ns per replica. When the limit is very long, each round is 1/20 of the limit (for example 50 ns per round for a 1 µs limit), so there are about 20 decisions.
4. **Diagnose after each round.** For each target residue, count how many times the chi1 dihedral flips between the three states g+, t and g-. Then sort the residues by how stuck they are.
5. **Decide after each round.** For each residue, the agent makes one of the decisions below and writes down the reason:

| Decision | Condition |
|---|---|
| `done` | Each replica flipped enough times, and the population error reached the precision the user asked for. Sampling is enough. |
| `extend` | Plain MD still helps: flips were seen but not enough, or the run has not reached the give-up time yet. |
| `enhanced_sampling` | No flip after a long run (by default 10 ns per replica), or the total budget is used up. Plain MD cannot answer this question. |
| `skip` | Gly or Ala, which have no chi1. |

While any residue is `extend` and the budget is not exceeded (by default 20 ns per replica), the agent runs another round. Otherwise it stops.

**Meaning of the diagnosis labels:**

| Label | Meaning |
|---|---|
| `replicas_disagree` | Different replicas stay in different states. The result depends on which replica you look at, so it has not converged. |
| `no_flip` | No replica flipped. The side chain may be stuck, or it may have only one stable state. Plain MD cannot tell these two cases apart. |
| `rare_flips` | It flipped, but too few times to estimate the state populations. |
| `imprecise` | It flipped enough times, but the population error of some state is larger than the precision the user asked for (default 0.05). |
| `ok` | Each replica flipped enough times, and the error reached the precision. |
| `fixed_chi1` | Proline or a cysteine in a disulfide bond. chi1 is fixed, so it is not judged. |

**Sorting:** residues are sorted by the mean time between flips, slowest first. This time is the total simulation time divided by the number of flips. With no flip, we only know that it is longer than the total simulation time, and the table shows `>`. For equal times, the residue with more disagreement between replicas comes first.

## Full workflow (in order)

1. **Input:** a PDB ID (downloaded to `structures/<ID>.pdb`), a PDB file, or an Amber system you prepared yourself (`--prmtop`/`--inpcrd`, which skips steps 2 and 3 and is used as is).
2. **Structure check:** compare with the UniProt reference sequence and look for tags, multiple copies, metals, unknown residues, long gaps and chain breaks. If there are problems, stop, unless `--accept-problems`. See "Structure check (PDB file or PDB ID)" below.
3. **Process the structure (PDBFixer), in this order:**
   1. Keep only the selected chains (`--chains`). Remove tags with coordinates at the chain ends.
   2. Remove all non-protein molecules: crystal water, ions, ligands, cofactors, terminal caps (ACE, NH2, NME).
   3. Model missing residues inside the chain. Do not add missing residues at the chain ends.
   4. Replace non-standard residues with standard amino acids (for example MSE → MET).
   5. Add missing heavy atoms.
   6. Remove the existing hydrogens and add hydrogens again for pH 7 (or by PROPKA, or by manual assignment). Hydrogen bonds decide whether His gets its hydrogen on ND1 or NE2.
   7. **Chain ends (`--termini`, default `auto`):** the structure check maps the first and the last residue with coordinates of each chain onto the full UniProt sequence. If the real protein continues past this end (for example only one domain was solved, or residues are missing at the ends), this end gets a neutral cap: ACE at the N terminus, NME (N-methylamide) at the C terminus. If this end is the real start or end of the protein, it keeps the charged terminus: NH₃⁺ at the N terminus, COO⁻ at the C terminus. Without a reference sequence, the decision uses whether residues are missing at the structure ends. `--termini charged` makes all ends charged, and `--termini capped` caps all ends. With CHARMM, the cap atoms are merged into the terminal residue (the CHARMM ACE and CT3 patches). With Amber, ACE and NME are separate residues.
   8. Add a water box (cube by default, at least 1.0 nm from the protein to each face, and the narrowest box width at least twice the cutoff plus 10%). First neutralize the charge, then add 0.15 M NaCl.
   9. Force field: by default CHARMM36m + CHARMM TIP3P (cutoff 1.2 nm, smooth switch from 1.0 nm, no long-range dispersion correction). Other choices are Amber ff14SB + TIP3P or ff19SB + OPC (cutoff 1.0 nm, with long-range dispersion correction). Electrostatics use PME. Bonds with hydrogen are rigid constraints.
4. **Equilibration (600 ps per replica):** restraints of 1 kcal/mol/Å² on backbone N, CA and C → energy minimization → NVT heating from 5 K to 300 K (100 ps, in 20 steps) → restrained NPT (200 ps) → restraint removal in 20 steps (200 ps) → unrestrained NPT (100 ps). Langevin integrator, 2 fs time step, friction 1/ps, pressure 1 bar.
5. **Plain MD in rounds:** 3 replicas. Round 1 is 1 ns per replica, each later round adds 5 ns, at most 20 ns per replica. Protein coordinates are saved every 5 ps. Adaptive sampling is optional (`--sampling adaptive`).
6. **Diagnose after each round:** count the chi1 flips of each side chain in the target region, and decide "sampling is enough", "continue" or "hand over to enhanced sampling" (no flip so far, and the remaining budget is not enough to see enough flips; see "When to give up on an item that never changes" below).
7. **Enhanced sampling:** for the side chains that need it, run well-tempered metadynamics on chi1 (bias factor 10, Gaussian height 1.2 kJ/mol, width 12°, one Gaussian every 1 ps), 2 replicas, at most 20 ns each, starting from the final state of plain MD. Every 1 ns, reweight and compute the state populations. Stop when the result is stable (see "Enhanced sampling").
8. **Output:** everything is in `runs/<name>/`. See "Files in a run folder" below.

## Two kinds of input

- **PDB ID (`--pdb-id 3HTB`):** MDAgent downloads the structure from RCSB PDB, saves it as `structures/3HTB.pdb` in the project folder, and then uses it like `--pdb`. A structure that is already downloaded is not downloaded again. To download only, without a run: `python -m mdagent fetch 3HTB 181L`. It prints the location and a content overview of each file. Large structures that exist only in mmCIF format are not supported yet. `structures/` is not uploaded to GitHub.
- **Protein PDB file (`--pdb`):** MDAgent builds the system itself. You can adjust it with "System setup options" below. Ligands are removed, because the OpenMM built-in force fields have no ligand parameters.
- **An Amber system you prepared yourself (`--prmtop X.parm7 --inpcrd X.rst7`):** for example a protein-ligand complex prepared with tleap and GAFF parameters. MDAgent uses it as is. It adds or removes no atoms, and the ligand and all parameters stay. The file must have a periodic box (already solvated). Nonbonded cutoff 1.0 nm, PME. Residues in Amber files are renumbered from 1. Use `python -m mdagent info --prmtop X.parm7` to see the numbers, and write `--residues` without a chain name, like `3-5`. "System setup options" have no effect on this input. If you use them, the program tells you.

## Structure check (PDB file or PDB ID)

Before it builds the system, MDAgent checks the structure. Only a "clean" structure goes to the next step. When it finds problems, the run stops (exit code 4) and explains what to do. To check only, without building: `python -m mdagent check --pdb-id 4OBE`.

What it checks:

- **Comparison with the reference sequence:** the PDB file records the UniProt entry and residue range of each chain (DBREF), and each known difference with its reason (SEQADV, for example expression tags or engineered mutations). MDAgent downloads the UniProt sequence (including all isoforms), saves it in `structures/uniprot/<ID>.fasta`, and compares position by position. Differences with a reason in the file are only recorded. Differences without a reason are problems. If the numbering mapping in the file is clearly wrong, the program aligns the sequences again.
- **Tags:** residues marked as expression tags or similar, or outside the range of the reference sequence, count as tags. Tags with coordinates at the chain ends are removed by default (`--keep-tags` keeps them). A tag inside the chain is a problem.
- **Multiple chains:** multiple copies of the same protein are a problem. Use `--chains A` to choose one, or `--chains A,B` to keep all of them explicitly.
- **Other molecules:** water, ions and ligands are removed as usual. Structural metals (Zn, Fe, Cu and so on) are a problem, because removing them can break the structure. To keep them, prepare an Amber system yourself and use `--prmtop`.
- **Completeness:** unknown residues, 5 or more consecutive missing residues inside a chain, and chain breaks not explained in the file are problems. Short gaps are modeled. Missing residues at chain ends, alternate conformations and multiple models are only recorded.
- **Files without a reference sequence** (for example your own models): a problem. You can give a reference sequence with `--uniprot <ID>`.

When you are sure the problems do not matter, add `--accept-problems` to continue. The check report is saved as `structure_check.txt` in the run folder.

Note: `examples/181L.pdb` has two unexplained differences from UniProt P00720 (R12G, I137R). So add `--accept-problems` when you test with it.

## Equilibration

Before the first production segment of each replica, the program equilibrates, by default 600 ps in total (change it with `--equil-ps`):

1. Restrain the protein backbone atoms (N, CA, C) (1 kcal/mol/Å²) and minimize the energy;
2. Heat in NVT from 5 K to the target temperature (default 300 K), 1/6 of the time;
3. Keep the restraints in NPT, 2/6 of the time;
4. Reduce the restraints to zero in 20 steps in NPT, 2/6 of the time;
5. Run unrestrained NPT, 1/6 of the time.

The restraint forces exist only in the equilibration simulation. Production is a new simulation. It receives only the coordinates, velocities and box from the end of equilibration. So all three come from the same moment, and no restraint force remains in production.

## System setup options

| Option | Default | Description |
|---|---|---|
| `--forcefield` | `charmm36m` | `charmm36m`: CHARMM36m + CHARMM TIP3P, van der Waals cutoff 1.2 nm with a smooth switch from 1.0 nm, no long-range dispersion correction. `amber14`: ff14SB + TIP3P. `amber19`: ff19SB + OPC. Each option is a full matched set of force field, water model and cutoff settings. You cannot split and mix them. |
| `--ph` | 7 | Sets the default protonation states of titratable residues. |
| `--check-protonation` | off | When on, PROPKA first predicts the pKa of each Asp, Glu, Lys and His from the protein environment. The protonation states follow the predictions, and then MD starts. The report lists all changed residues and marks those in the target region. |
| `--protonation` | none | Manual assignment, for example `A:31=HIP,A:70=ASH`. It applies after PROPKA, so it overrides the PROPKA result. Residues at the chain ends cannot be changed. |
| `--ion-conc` | 0.15 | Salt concentration (M). The system is always neutralized first. |
| `--cation` | `Na+` | `Na+` or `K+`. The anion is Cl⁻. |
| `--padding-nm` | 1.0 | Minimum distance from the protein to each face of the water box. The box is periodic, so the protein and its image in the next box are at least twice this distance apart. |
| `--box-shape` | `cube` | `cube`, `dodecahedron` or `octahedron`. A rhombic dodecahedron box has about 30% less water than a cube, so the simulation is faster. |
| `--setup-only` | off | Only build the system and stop, so you can check the report first. After you confirm, start MD with `resume`. |

**Missing residues:** the program lists the missing residues segment by segment. Missing residues at the chain ends are not added, because they are usually disordered regions. PDBFixer adds those inside the chain. The program warns when an added segment is longer than 5 residues or within 4 residues of the target region. If the PDB file has no sequence records (SEQRES), PDBFixer cannot find missing residues. Then the program checks the peptide bond lengths between neighboring residues and warns when it finds a chain break.

The full setup report is shown on screen and saved as `setup_report.txt`. All options are also recorded in `run.json`.

**Speed:** OpenMM must compute some CHARMM van der Waals parameters in a slower way, so CHARMM36m is somewhat slower than Amber.

## Usage

```bash
git clone https://github.com/Lyn-666/MDAgent1.git && cd MDAgent1
conda env create -f environment.yml && conda activate mdagent && python -m pip install -e .

# Start. The agent decides how long to run, at most 20 ns per replica.
# 181L has two differences from the UniProt reference sequence, so add --accept-problems (see "Structure check (PDB file or PDB ID)").
python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/181L --accept-problems

# First check protonation states with PROPKA, use Amber ff19SB + OPC and 0.1 M KCl, only build the system without MD.
python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/181L_b --accept-problems \
    --check-protonation --forcefield amber19 --ion-conc 0.1 --cation K+ --setup-only
python -m mdagent resume --run runs/181L_b   # After reading the report, start MD.

# Continue: resume an interrupted run, or give a finished run a larger budget.
python -m mdagent resume --run runs/181L --max-ns 40

# Show the latest sorted diagnosis.
python -m mdagent diagnose --run runs/181L

# Replay every look with only the data it had: which item and replica held the run back.
python -m mdagent diagnose --run runs/181L --each-round

pytest -q tests                # all tests, about 1 minute
pytest -q tests -m "not slow"  # tests without MD, about 1 second
```

Common options: `--replicas` (number of replicas, default 3), `--first-ns` (length of round 1, default 1), `--extend-ns` (length of each later round, default 5), `--max-ns` (total budget per replica, default 20), `--no-flip-patience-ns` (how long without a flip before plain MD gives up; by default no fixed time is used, and the decision uses the remaining budget, see below).

**When to give up on an item that never changes:** when an item (a side chain, a distance, a backbone residue) has not changed state since the start, there are two possibilities: it is trapped, or it has only one stable state. The data cannot tell these apart. The agent does not use a fixed waiting time (that would depend on the system). Instead it asks: from the current data, how fast can the transition rate be at most (with no transition seen, the 95% upper confidence limit is 3 ÷ total time run so far)? Even at this fastest rate, can the remaining budget show the number of transitions the target needs (by default 4 per replica)? Only if not does it give up. The reason is recorded as "stalled", and it suggests a different method. For example, with 3 replicas and a limit of 20 ns each, it gives up at about 5 ns per replica. With a limit of 1000 ns, it gives up at about 250 ns per replica. When the user gives `--no-flip-patience-ns`, the agent uses that time.

Speed reference: the 181L system has about 35,000 atoms. On a consumer GPU it runs about 470 ns per day, and a full run with default settings takes about 3 hours. On a 4-core CPU it runs about 3 ns per day, which is only good for tests.

## Files in a run folder

| File | Content |
|---|---|
| `run.json` | Setup options, MD settings, target residues, length of each round. `resume` reads from here. |
| `setup_report.txt` | Setup report: force field, missing residues, protonation, water box, ions and warnings. |
| `decisions.log` | Decisions and reasons of each round, plain text. |
| `status.json` | Latest diagnosis and decisions, for other programs to read. |
| `replica_<i>_part<k>.dcd` | Trajectory of replica i in round k, protein atoms only. |
| `replica_<i>_part<k>.state.xml` | State at the end of round k. It appears only after this round finishes completely. |
| `solvated.pdb`, `system.xml`, `protein.pdb` | The built system and the protein structure for analysis. |

If a round stops in the middle, `resume` first completes this round. Finished trajectory segments are not run again.

Files are saved on the computer that runs the simulation. The first line of the log gives the full path of the run folder and the computer name. When the run ends, the Summary at the end of the log lists them again and explains each kind of file. All `.dcd` files contain only protein atoms. Use `protein.pdb` as the topology.

## Test systems

`examples/181L.pdb` is the complex of T4 lysozyme L99A with benzene. The Val111 side chain in the pocket rarely flips in conventional MD. The literature reports this.

## Current limitations

- With `--pdb` input, the ligand is removed and only the protein is simulated. To keep the ligand, prepare an Amber system first, then input it with `--prmtop`/`--inpcrd`.
- PROPKA runs only when the user asks for it, and it predicts for the protein without the ligand.
- Only chi1 is analyzed, not chi2 yet.
- Prolines and cysteines in disulfide bonds get no diagnosis and no enhanced sampling, because the ring or the S-S bond fixes their chi1, so it does not flip between states. Their label in the report is `fixed_chi1`.
- Enhanced sampling biases at most 10 side chains at a time (change it with `--enhanced-max-residues`). It chooses the slowest ones in plain MD. Biasing many side chains at once pushes the whole protein, and a small protein can unfold. If the target region is too large, make it smaller first.

## Enhanced sampling

After plain MD stops, the agent runs enhanced sampling on the side chains labeled `enhanced_sampling` (it starts automatically in "agent decides" mode; in "user specifies" mode it needs the user's consent). The method is well-tempered metadynamics, implemented directly in OpenMM, with no need for PLUMED. Each side chain's chi1 has its own bias potential, and all side chains are handled in one simulation. Each enhanced sampling replica starts from the final state of plain MD. By default there are 2 replicas, at most 20 ns each. Extend with `resume --enhanced-ns 30`.

**Analysis:** the two enhanced sampling replicas run in turns, 1 ns each time. After each round, the agent removes the effect of the bias potential and computes the true population of each side chain in the states g+, t and g-. The method is the time-dependent reweighting of Tiwary and Parrinello (2015). The program saves every chi1 sample point, and with them it can rebuild the bias potential at any moment exactly. The first 10% of each replica is not used, because the bias potential grows fastest then. For the error calculation, see "How errors are computed" below.

**The judgement for each side chain:**

| Judgement | Condition |
|---|---|
| `reached` | Target reached: the error of each state population is at most the precision the user asked for, the replicas agree, each replica moved between states at least 10 times, the weights are not degenerate, and each replica ran at least 5 ns. |
| `stalled` | Stalled: some replica almost never moves between states. The bias potential cannot push this side chain. So even a small error cannot be trusted, because all replicas may be trapped in the same state. |
| `unreliable` | Degenerate weights: fewer than 50 effective samples, so the result depends on only a few frames. |
| `not_yet` | Not precise enough yet, or the replicas disagree. |

The change in populations since the last round (`change_since_last`) is recorded only as a diagnostic. It is not a reason to stop.

**Tested cases:** in a model with known populations, reweighting gives 0.21 / 0.69 / 0.10, and the true values are 0.20 / 0.70 / 0.10. Time fractions without reweighting are 0.25 / 0.29 / 0.24. For two coupled side chains (the state of one changes the populations of the other), each with only a one-dimensional bias potential, the populations differ from the true values by less than the error, for example 0.19 ± 0.01 / 0.61 ± 0.02 / 0.20 ± 0.02 against true values 0.20 / 0.60 / 0.20.

Output is in `enhanced/` in the run folder:

| File | Content |
|---|---|
| `replica_<i>_chunk<k>.dcd` | Protein trajectory of enhanced sampling replica i, 1 ns segment k. Use `protein.pdb` in the run folder as the topology. |
| `replica_<i>_chi.npy` | chi1 (radians) of each biased residue, one row every 1 ps. |
| `replica_<i>_bias.npy` | Final bias potential (kJ/mol) of each biased residue on a chi1 grid from -180° to 180°, 361 points. |
| `populations.json` | For each side chain after reweighting: state populations and errors, relative free energies and errors (kcal/mol), populations and flip counts of each replica, effective sample size, judgement. The end of the log also has a table. |
| `replica_<i>.state.xml`, `replica_<i>.json` | State and progress for `resume`. |

The order of residues in the `.npy` files is the same as the order of `enhanced.residues` in `run.json`.

**Note:** the enhanced sampling trajectories and `replica_<i>_chi.npy` are sampled under the bias potential. Counting time fractions directly gives wrong, nearly equal populations. Use the reweighted results in `populations.json`.

## Adaptive sampling (optional)

With `--sampling adaptive`, from round 2 of plain MD on, each replica no longer continues from its own last segment. It restarts from frames that AdaptivePy (the adaptive sampling package of the Shukla group) chooses. The default policy `least_counts` chooses frames from the "least visited" regions. Install it first: `pip install adaptivepy-sampling`.

- **Features:** sine and cosine of chi1 of each side chain in the target region.
- **Spread-out starts:** the number of clusters is one cluster per 5 frames (at most `--adaptive-clusters`). AdaptivePy first gives 3 times the needed number of candidate frames. The agent then chooses the ones that differ most from each other as starts, so that replicas do not start from almost the same structure.
- **Full coordinates:** in this mode, each segment also saves one frame of the full system (with water) every 10 ps, in `replica_<i>_part<k>.full.dcd`. Starts are chosen only from these frames. At a restart, velocities are drawn again at random for the set temperature.
- **Judgement rule:** the starts of the segments are chosen to be different on purpose. So in this mode the agent looks only at the total flip count of all segments. It no longer checks whether the replicas stay in the same state.
- **How populations are computed:** the starts are chosen (for example only rare states), so counting the time in each state directly gives wrong populations. So this mode uses a Markov state model (`msm.py`). It counts the transitions between states in each segment, estimates transition probabilities that satisfy detailed balance (maximum likelihood), and takes the equilibrium distribution of this model as the populations (Prinz et al. 2011; Bowman et al. 2009). A frame between two states counts as the previous state (the core-set method, Buchete and Hummer 2008). The error comes from resampling the segments (bootstrap), and it is not smaller than the error floor. A state that is only entered and never left (or only left and never entered) has no computable population. The result says so, and the judgement is too few flips. After the required number of flips, the error must also be at most the required precision to pass. In a test where all segments start from a state with only 5% population, counting time directly gives about 28%, and the model gives 4.9% ± 0.7%.
- **Records:** the starts chosen in each round are in `starts` in `run.json`. The AdaptivePy input and output are in `adaptive/round_<k>/`. The program checks the frame numbers that AdaptivePy gives. If they do not match, it stops with an error.

Options: `--adaptive-policy` (`least_counts` or `random`), `--adaptive-clusters` (number of clusters, default 20), `--full-save-ps` (save interval of full coordinates, default 10 ps).

## What is measured (quantities)

The agent decides "is it enough" from the quantity the user cares about. Each quantity is first turned into "states": which state each frame belongs to. Then one shared method counts the switches between states, computes the population and error of each state, and judges whether it is enough (see "How errors are computed" and "Goal, permissions and stop reasons"). There are now three state-based quantities and one continuous quantity (structural stability, which has no states and uses the mean directly). You can also set the goal to "stop when you see a certain change" (see below):

| Quantity | When to use | States | How to specify | Enhanced sampling |
|---|---|---|---|---|
| Side-chain conformation (`--quantity chi1`, default) | You care about the conformations of side chains in a pocket | chi1 of each side chain: g+, t, g- | `--residues A:99-120` or `--around-ligand BNZ` | Yes (metadynamics on chi1) |
| Distance between atoms (`--quantity distance`) | You care about a pocket opening and closing, contact between two residues, or a channel opening | For each distance: "near" (below the threshold) and "far" | `--distances "A:88:OH-A:111:CG1@0.6"`, several separated by semicolons | Not yet |
| Backbone basins (`--quantity backbone`) | You care about the Ramachandran basin populations of any residue, transitions between basins, the φ/ψ free-energy surface, or the type of a β turn | `rama`: by default alpha_R, beta (with PPII), alpha_L, other; `turn`: β-I, β-II | `--backbone "rama:A:110-112;turn:A:52"`, or `--residues A:110-112` (one rama item per residue) | Yes (metadynamics on the two dihedrals of each item) |
| Structural stability (`--quantity stability`) | You care whether the protein or a region keeps its starting structure | Not states, but two continuous quantities: backbone RMSD relative to the starting structure and radius of gyration Rg (nm) | With no residues, the whole protein; `--residues A:42-76` for one region only | No |

**Backbone basins:** these work for any protein and any residue. `rama:A:73` is φ/ψ of residue 73. `turn:A:52` is the β turn whose second residue (i+1) is residue 52. It uses ψ(52) and φ(53) to tell type I from type II (the sources of the two centres are in `basins.py`). With only `--residues`, each residue becomes one rama item. The first residue of a chain has no φ and the last has no ψ, so they are skipped with a note. Each item gives the results below. Each has its own precision requirement. The item passes only when the populations, the main transition rates and the free-energy surface all pass:

- **Basin populations**, error at most `--precision` (default 0.05; for the precision standard see "How errors are computed"). A frame counts as a basin change only when it enters the "core" of the basin. Jitter back and forth on a boundary does not count.
- **Transition statistics between basins:** for each pair of basins, the number of transitions, the rate (per ns) and the mean residence time (total time in this basin ÷ number of exits), all with errors. The relative error of the main transitions (between basins with at least 5% population, both directions) is at most `--rate-rel-error` (default 0.25, which needs about 16 transitions).
- **Free-energy difference between the two main basins** (kT), with error.
- **φ/ψ free-energy surface:** 15° bins, F = −kT ln p, lowest point at 0. The error comes from the differences between replicas, so at least 2 replicas are needed. In bins with free energy at most 2 kcal/mol, the error is at most `--fes-error-kcal` (default 0.5 kcal/mol). There is also a map with 10° bins (`landscape_*.svg`, data in the `.csv` with the same name).

**How the basins are defined:** the definition affects the conclusions. By default, rama divides the whole Ramachandran map into four regions (alpha_R, beta with PPII, alpha_L, other). A frame must be at least `--basin-margin` degrees (default 10°) from a boundary to count as entering. So the region that glycine often visits (φ≈80°, ψ≈180°) also has its own basin. You can replace the regions with your own through `--basins`, written as `name:φ from,φ to,ψ from,ψ to` in degrees, several separated by semicolons. The same name can appear several times, which means the basin has several parts. `--rama-basins centres` uses three centres instead (alphaR, beta, alphaL), and within 50° of a centre counts as entering. `rama:` runs started before 2026-11-22 use this automatically when they resume, so the definition does not change in the middle of a run. β turns always use two centres. Results are in `backbone/` in the run folder: one `.json` per item (transition statistics, lowest free-energy point of each basin), and one `_fes.csv` and `_fes.npz` (free-energy surface and errors, readable by any plotting program).

**Enhanced sampling for backbone basins:** an item that never changes basin in plain MD, and whose remaining budget is not enough to see enough transitions, can switch to enhanced sampling. Each of the two dihedrals of the item (φ and ψ for rama, ψ(i+1) and φ(i+2) for turn) gets a one-dimensional well-tempered metadynamics bias potential (at most `--enhanced-max-residues` items at a time, default 10). Tiwary–Parrinello reweighting then gives back the basin populations and the free-energy surface (`backbone/<item>_fes_reweighted.csv`). With the `switch_method` permission, the agent switches by itself. Otherwise it only suggests it in the chat and switches after you confirm. The bias potential changes the dynamics, so **transition rates and residence times come only from plain MD**. The report says so. If an item never changed basin in plain MD, its rates cannot be given, and it needs longer plain MD. Checked with a model of known answers: true populations of three basins 0.70, 0.25, 0.05; reweighting gives 0.67 ± 0.05, 0.25 ± 0.05, 0.075 ± 0.015. Finished trajectory segments are read only once and stored in `backbone/cache/`, so in a very long run each round reads only the new segment.

**Structural stability (2026-11-23, proposed by a tester on 1YRF):** each round computes from the trajectory the backbone RMSD (after superposition) and the radius of gyration Rg relative to the starting structure (`protein.pdb`). The atoms are the Cα atoms of the region. When the region has fewer than 5 residues, heavy atoms are used. The unequilibrated start is removed with the same automated detection as for the other quantities (`uncertainty.joint_equilibration_start_values`, one shared start for all replicas). Each quantity is judged separately, and the item passes only when both are "ok":

- **Steady state:** for each replica, the mean of the second half minus the mean of the first half, then averaged over replicas. The two halves are paired within the same replica, so replicas that stay at different levels are not taken as drift. The quantity is at steady state only when the difference plus its interval half-width is at most 3 × precision (default 0.06 nm). This is an equivalence test, not "no visible difference". When the difference alone is above this limit, the label is drifting.
- **Precision:** the interval half-width of the mean is at most `--stability-tolerance` (default 0.02 nm). The error is the larger of the within-replica error (block averaging) and the between-replica error. When a chi-square test shows that the differences between replicas are real, the t distribution is used.
- **Independent samples:** each replica needs at least 4 independent samples (frames ÷ statistical inefficiency). Otherwise the label is rare_flips, and the error is not trusted.

The report also gives: the time fraction with RMSD below `--rmsd-cutoff` (default 0.3 nm, with a ±0.05 nm buffer) and the number of exits; when there is no exit, the 95% upper limit of the exit rate 3/T (T is the total time of all replicas); and the RMSD–Rg free-energy surface (25 × 25 bins, error from differences between replicas). Results are in `stability/`: `summary.json`, `rmsd_rg.csv/.svg`, `fes_rmsd_rg.csv/.svg`. The allocation rule is shared with the other quantities through `allocation.next_step`: errors are converted to one unit by precision/stability tolerance (the allocation calculation does not depend on scale). The code is in `stability.py`.

**Conformational clustering (`--cluster`, automatic for structural stability):** the GROMOS method (Daura 1999). The equilibrated frames (all replicas together, sampled evenly when there are more than 2500 frames) are superposed in pairs to compute RMSD. The frame with the most neighbors (RMSD ≤ `--cluster-cutoff`, default 0.15 nm) is the centre. It forms a cluster with its neighbors, these are removed, and the process repeats. The report gives the fractions of the top 10 clusters (error from the differences in fraction between replicas), the replicas in which they appear, and representative structures `clusters/cluster_<k>.pdb`. The data is in `clusters/clusters.json`. The atoms are the same as for structural stability. Clustering runs when the report is written. If it fails, it prints only a note and does not affect the report. The code is in `clustering.py`.

**Not supported yet:** a free-energy surface for distances between atoms.

**How to write a distance:** `chain:residue number:atom name-chain:residue number:atom name@threshold`, threshold in nm. Atom names are standard PDB names, for example CA, CB, OH, NZ, CG1. Amber systems have no chain names, so you can omit the chain, for example `88:OH-111:CG1@0.6`. The distance must cross the threshold by ±0.05 nm to count as a state change. Jitter around the threshold does not count. The program checks that the atoms exist once before and once after the system build. If it cannot find them, it stops and explains why. In the chat you can say directly "look at the distance between OH of Tyr88 and CG1 of Val111, within 0.6 nm counts as contact".

**When a distance is stuck:** there is no enhanced sampling method for distances yet. When plain MD shows no change and the remaining budget is not enough to see a change, the agent says so in the chat and suggests extending, adding replicas or switching to adaptive sampling. It does not change the method by itself. The stop reason is recorded as "stalled".

**Stop when a change is seen (`--observe`):** sometimes the user needs no populations and only wants to know "does a certain change happen". When it is seen, the run can stop. Write it as "item=state", several separated by semicolons:

- `--observe "A:111=t"`: chi1 of side chain 111 enters the t state. Side-chain states are g+, t and g-.
- `--observe "A:88:OH-A:111:CG1@0.6=far"`: this distance becomes "far". Distance states are near and far.

It counts as seen only when the item enters the target state from another state and stays there continuously for at least `--hold-ps` (default 100 ps). So the state of the starting structure and short flickers do not count. When it is seen in `--seen-in` replicas (default 1), the run stops with the reason "target reached". The chat and the report say in which replica and at which ns it was seen. When the item never changes and the remaining budget is not enough, the run stops and suggests a different method. When it changes but has not entered the target state yet, the run goes to the limit and then suggests extending or switching to adaptive sampling. This kind of goal needs no population precision. The time in each state in the report is only for reference. In the chat you can say "Run 181L, stop when you see Val111 enter the t state".

## Letting the agent decide replicas, length and stopping (`--allocation projection`)

This is the core ability of the agent: the user only says what to measure and how precisely, and the agent decides how to do it with the least compute. The chat uses this mode when you say, for example:

> Use Ubiquitin (1UBQ) and estimate with standard unbiased MD the Ramachandran basin populations, transition statistics and φ/ψ free-energy landscape of residues 8, 9, 10 and the D52/G53 turn in the 50 loop. You may decide the number of replicas, the length of each simulation and when to stop. The goal is to minimize the total simulation cost while keeping the result accurate.

The language model fills this in as: backbone basins `rama:A:8-10;turn:A:52`, allocation by cost, extend, add replicas and stop early allowed, but no switch to enhanced sampling (because the user limited it to standard unbiased MD). If the user only says "the given residues" and gives no numbers, it first asks which residues. It does not guess.

What the agent does (the rule is in `allocation.py`; the Experiment A replay uses the same code):

1. **Start:** run the first segment with the set number of replicas (by default 1 ns each). This look is only for planning. It cannot stop, so it uses none of the significance level.
2. **Each look:** first remove the unequilibrated start automatically (`--burn-in auto`, the default in this mode). This uses the automated equilibration detection of Chodera 2016. All replicas start from the same structure, so they share one start point. The start is removed only when this raises the effective sample size to at least 1.5 times, and the median number of conformational changes per replica is at least 20. Otherwise nothing is removed. Only the agent's own trajectories are used. The removed part still counts as cost, and the "never changes" judgement uses only the kept time. Then the agent computes, for each quantity, the populations, the error with time correlation, the interval calibrated for the number of looks, and how much of the error is "within replicas" and how much is "between replicas". When a chi-square test shows that the differences between replicas are real, the interval uses the t distribution with n−1 degrees of freedom, because then the error comes only from the numbers of a few replicas and is itself very uncertain (with 3 replicas, the factor of the 95% interval is 4.30, not 1.96). When the differences are only large by chance, t is not used, because the error already takes the larger of the two. When there is "a sign" of differences (chi-square test at the 80% level), the projection includes them and prefers adding replicas. With only 3 replicas the differences cannot be measured well. So the agent gives up only when even the optimistic estimate (differences at the 95% lower confidence limit) cannot reach the precision. Otherwise it first adds replicas to measure the differences well.
3. **Allocation:** choose between "extend" and "add replicas" the combination with the lowest projected cost, and jump directly to that length instead of looking every few ns. Replicas are added only when the differences between replicas are statistically significant, because each new replica needs a new equilibration. A new replica starts from the same structure with a new random seed, and first catches up to the same length.
4. **Stop:** when all quantities are precise enough and each replica has moved between states enough times, the agent declares convergence (STOP). When a quantity never changed, the agent estimates from the 95% upper confidence limit how fast it can change at most. Only when even this most optimistic rate would need more than the budget to reach the precision does it mark that quantity alone as insufficient evidence. The other quantities continue. When the looks or the budget are used up, an item never changes, or the projected cost is far above the budget, the output is "insufficient evidence" (INSUFFICIENT_EVIDENCE). This is also a conclusion: the data cannot support the required precision.
5. **Delivery:** the report says what each look saw, why the agent allocated as it did, and why it stopped. It judges convergence for each quantity separately, and gives how much compute was used and what share of the budget.

The limits are `--budget-ns` (total production time of all replicas, by default the maximum allowed number of replicas × the limit per replica), `--max-ns` (limit per replica, never exceeded) and `--max-replicas` (default 10). There are at most `--stop-looks` looks that can stop (default 3). The default allocation mode is still `rounds` (a fixed extension per round), so the meaning of old runs does not change.

## Goal, permissions and stop reasons

**Goal:** each run records in `goal` in `run.json` what the user wants:

- **Target quantity:** one is supported now: the populations of the side chains in the target region in the three chi1 states. The user's own words are also recorded (`--goal`).
- **Precision:** the error of no state population may exceed this tolerance (`--precision`, default 0.05). `--criterion` decides how this error is computed:
  - `standard_error` (default): the standard error directly. This is what the program always did, so old runs keep their meaning.
  - `confidence`: the half-width of the 1−α confidence interval, with α divided over the number of looks known in advance (Bonferroni). The agent looks at the same growing data every round. The more it looks, the more likely it is to see "it looks precise enough" by chance. Dividing α is the price for this.
  - `sequence`: a confidence sequence (Robbins 1970 normal mixture). It holds for any number of looks. It is the widest and slowest of the three.

  The `budget` command can measure the cost and the correctness of each of the three criteria directly, before you choose one.
- **Limits:** at most how many ns per replica of plain MD (`--max-ns`), how many replicas (`--replicas`), and at most how many ns per replica of enhanced sampling (`--enhanced-ns`). These are limits: the agent can use less, but it does not go past them by itself.

**Permissions:** what the agent may do without asking the user (`--allow`, comma-separated, or `none`):

| Permission | Meaning | Used now |
|---|---|---|
| `switch_method` | Switch to enhanced sampling by itself when plain MD cannot sample | Yes |
| `stop_early` | Stop when the target is reached, without waiting for the budget to run out; also stop early when enhanced sampling cannot push an item | Yes |
| `extend` | When the target is not reached but nothing is stuck, continue past the budget, up to 2 times the first budget (`--extend-factor`) | Yes |
| `add_replicas` | When the error comes mainly from differences between replicas, add independent replicas, up to 10 (`--max-replicas`) | Yes |

**Defaults (from 2026-11-20):** "agent decides" mode allows only `stop_early`. "User specifies" mode allows nothing. So the agent stops early when the target is reached or when more simulation gives no benefit. But extending, adding replicas and changing the method are only suggested in the chat and happen only after the user confirms. To confirm, say in the chat "Extend run1 to 30 ns", "Add 2 replicas to run1", "Continue run1 with enhanced sampling", "Run run1 again with adaptive sampling", or use the command `python -m mdagent resume --run runs/run1 --max-ns 30` (`--replicas 5`, `--enhanced-sampling`). The user can also give the agent more permissions at the start with `--allow`.

**Stop reasons:** when each stage stops, the agent records one reason. It writes it to `run.json`, tells the user in the chat, and also puts it in the Summary of the log:

| Reason | Meaning |
|---|---|
| `target_reached` | Target reached: all side chains are `reached` (`ok` in plain MD). |
| `stalled` | Stalled: some side chain does not move between states with this method. In plain MD this means "no flip so far, and the remaining budget is not enough to see enough flips". In enhanced sampling it means "the bias potential cannot push it". |
| `not_reachable` | Not reachable within the limits: there are changes, but at the current rate the time needed to reach the target is far above the user's limits (see rule 6 below). |
| `budget_exhausted` | The budget is used up and the target is not reached. |
| `failed` | The run had an error. The error goes to `run.json` and the chat. After a fix, the run can continue. |

"The populations no longer change" is not a stop reason: when all replicas are trapped in the same state, the populations do not change either.

## Choosing the next action after each round (step 4)

After the analysis of each round, the agent chooses one action in the order below (`policy.py`). An action without permission is not done. It becomes a suggestion in the chat instead. Each choice is recorded in `run.json` (`notes.actions`) and `decisions.log`, with whether it was done and why.

Plain MD:

1. **All pass:** stop. Without the `stop_early` permission, run as planned until the budget is used up.
2. **Some item never changes:** if this quantity has enhanced sampling (side-chain conformation) and the `switch_method` permission is given, switch to enhanced sampling. Without this permission but with `stop_early`, this item no longer runs with plain MD (the other items continue as usual if they still need it), and at the end the agent suggests a different method. With neither permission (user specifies), continue as planned to the limit.
3. **Changes but not precise enough, and the error comes mainly from differences between replicas:** with the `add_replicas` permission, add 2 independent replicas (at most 10). The new replicas first catch up to the length of the others. The literature says that more replicas help more than longer runs in this case (Knapp 2018; Grossfield 2018).
4. **Changes but not precise enough, and the budget is used up:** with the `extend` permission, continue past the budget, one round at a time, up to 2 times the first budget. Otherwise stop with the reason "budget exhausted". Stuck items do not count here: more of the same method does not help them.
5. **The replicas stay in different states and never change:** suggest restarting from frames of these states (adaptive sampling). This is only a suggestion and is not done automatically: replicas that start from chosen frames cannot give equilibrium populations directly.
6. **Not reachable within the limits:** after at least two rounds and after a quarter of the limit is used, the agent estimates from what it saw how long each item that has not passed still needs. State changes follow the mean rate of the replicas (optimistic), and the error falls with the square root of time. If, for all items that have not passed, even this optimistic estimate is above 1.5 times the limit, more simulation cannot reach the target. Then the agent stops early with the reason `not_reachable`, and says in the chat how long it would need and suggests a higher limit or a different method. Without the `stop_early` permission it only sends the message and runs to the limit.

The final suggestions depend on the reason. For items that move between states but too slowly, or whose replicas disagree, the agent suggests adaptive sampling. For side chains that never changed, it suggests enhanced sampling (adaptive sampling can only start from states already seen, so it cannot help).

Enhanced sampling: stop when the target is reached (with `stop_early`). Stop early when all side chains that have not passed cannot be pushed for 3 rounds in a row. Add one enhanced sampling replica when the error comes mainly from differences between replicas. When the budget is used up and nothing is stuck, extend by half with the `extend` permission, up to 2 times the first budget.

## Delivery: the result report (step 6)

When each run ends (`run` or `resume` finishes), the agent writes `report.zh.md` and `report.en.md` in the run folder and says in the chat where the reports are. All content of the reports comes from the run files. The language model does not write it:

- **Question:** the target quantity, the user's own words, the required precision, and what the agent may do by itself.
- **What was simulated:** input, force field, ions, water box, temperature, number of replicas and length. With PDB input, the report says that the ligand was removed.
- **Conclusions:** for each item, the population of each state ± standard error, whether it comes from plain MD or from enhanced sampling (reweighted), and whether it meets the requirement. Results that meet the requirement can be cited as equilibrium populations for this force field and these conditions. Results that do not are only for reference.
- **Why it stopped:** the stop reason of each stage, and each action the agent took or suggested.
- **Reliability and limitations:** including "for an item that never changed state, an error of 0 means it was trapped, not that it is precise".
- **Files and open questions.**

The report also says how much compute was used: the total production time of all replicas, the equilibration time, the sum, and the real hours used.

`python -m mdagent report --run runs/<name>` writes the report again.

**Do two runs give the same answer:**

```bash
python -m mdagent compare --run runs/<name> --reference runs/<reference>
```

The judgement uses an **equivalence test (TOST)**: for each state of each item, the difference plus the half-width of its confidence interval must be within the tolerance (`|difference| + z·combined error ≤ tolerance`; the tolerance defaults to the precision required by the compared run, and you can change it with `--tolerance`).

Before, the test was `|z| ≤ 2`. That means "a difference was not proved", not "the absence of a difference was proved". With less data and larger errors, `|z|` is more easily below 2. So a run that is too short **looks correct exactly because it is imprecise**. A short run must prove the opposite, so the test changed to an equivalence test. z is still printed, because it shows in which direction the difference goes.

## How much compute: comparing strategies (`budget`)

This is the core question of the project. To claim "cheaper", you must compare against other ways to allocate compute. Running real MD for each strategy repeats the whole computation several times. So the program uses **replay** instead:

First, run one high-budget **data pool** (pool): many independent replicas, each longer than any strategy needs. Then each strategy replays on this pool. At each decision point, it sees only its own replicas, and only the frames up to its current length. From these it decides to extend, start one more replica, or stop. It pays only for the part it uses (`cost.py`). The pool is ordinary unbiased MD. So for strategies that only extend and only add independent replicas, the replay is exact. "Restart from selected frames" (adaptive sampling) cannot be replayed: those frames are not in the pool, so they must be really computed.

The four strategies in the comparison:

| Strategy | Decision rule |
|---|---|
| `fixed` | A plan set in advance: fixed number of replicas, fixed length, no look at the data during the run (baseline; use `--fixed-replicas`, `--fixed-ns` to set the plan you would really write) |
| `early_stop` | The same fixed number of replicas, but it stops as soon as it reaches the precision |
| `add_replicas` | Also decides where to spend time: it starts one more replica when the error comes mainly from differences between replicas, and otherwise extends |
| `mdagent` | This program's strategy (`policy.py`): stop early; add replicas only when replicas clearly disagree and less than half the budget is used; continue past the budget when an item still changes but the budget is used up; stop early when an item never changes, because plain MD cannot answer it however long it runs |

At the end, each strategy's answer is compared with an **independent reference**: the pool replicas that the strategy did not use, at full length (with a model of known populations, the comparison is with the true value). A comparison with the whole pool is a comparison with itself, because the strategy's own frames are in it.

`--repeats` sets the number of repeats. Each repeat uses a different set of replicas. One win proves nothing. What matters is the fraction of repeats that reach the correct answer, and the cost of each.

```bash
# No GPU needed: two-state model with known populations, 10 repeats
python -m mdagent budget --synthetic 0.3,0.7 --rate-per-ns 5 --pool-replicas 16 --pool-ns 60 \
    --repeats 10 --fixed-replicas 5 --fixed-ns 40 --parallel 4 --ns-per-day 600

# Use a finished long run as the data pool
python -m mdagent budget --run runs/long_1UBQ --repeats 10 --criterion confidence
```

The results go into `budget.json` and `budget.md`, with one strategy per table row:

```
strategy        compute ns  aggregate    wall  replicas  ns each  precise  accurate  proven  worst err   saved
fixed                203.0      200.0    81.2       5.0     40.0    100%       90%     20%      0.055      0%
early_stop            34.8       33.0    11.6       3.0     11.0     60%       80%      0%      0.097     83%
add_replicas          66.0       60.0    19.8      10.0      6.0    100%      100%      0%      0.041     68%
mdagent               51.9       48.0    16.2       3.0      9.0    100%       90%      0%      0.057     74%
```

- **compute ns** is the main cost: the production time of all replicas, plus the equilibration time of each replica. So "10 replicas of 10 ns each" costs more than "1 replica of 100 ns", although both add up to 100 ns.
- **wall** is the waiting time, scaled by `--parallel` (how many replicas can run at the same time). With enough GPUs, more replicas give a shorter wait at the same cost. So these three numbers are reported separately and are not reduced together as one metric.
- **precise** means the strategy itself judges that "the data are enough". **accurate** means its answer is really within the tolerance. Only accurate shows "cheaper but equally correct". Cheap but not accurate means it stopped too early.
- **proven** is the stricter equivalence test (TOST). It also requires the strategy's own error to be small enough to count as proof.
- The numbers above come from the model with known populations. They are not results for a real protein.

Two findings on this model so far (the table above is this result, tolerance 0.05, 10 repeats):

1. **Early stopping does save compute, but it can stop too early.** `early_stop` saves 83% of the compute, but in 2 of 10 repeats the answer is outside the tolerance (maximum deviation 0.097). `mdagent` saves 74% and is within the tolerance in 9 of 10 repeats (maximum deviation 0.057), the same as the fixed plan. The small extra time it spends buys exactly this.
2. **A stricter criterion is not always more accurate. It first needs enough budget.** With the criterion `confidence`, all strategies run longer (`early_stop` from 35 ns to 62 ns, `mdagent` from 52 ns to 122 ns). But with a budget of 20 ns per replica they cannot reach this criterion. They only stop with "budget exhausted", and the answer does not improve. A stricter criterion is only useful with a larger budget. Whether it is worth it must be measured for each system.

Also note: the 10 repeats on the same data pool are not 10 independent experiments (only the choice of replicas changes). So "2 of 10" shows how sensitive the result is to which replicas it gets. It does not replace running the whole experiment 10 times.

## Experiment A: how many replicas, how long, when to stop (1UBQ)

**Question:** Given an MD task and a precision requirement set in advance, can the agent decide by itself how many replicas to run, how long each runs and when to stop, and get a statistically reliable result with the least compute? This is a benchmark of "how much compute it takes to reach the precision". It is not simple convergence detection.

**Hypothesis A:** An uncertainty-aware agent can allocate replicas and lengths dynamically and decide when to stop. It reaches the accuracy and reliability set in advance with less total compute than fixed plans and conventional sequential stopping.

### Online stopping and offline checking are separate

This is the most important rule of the experiment. The **online stopping criterion** uses only the trajectories that the method itself runs: populations, errors that take time correlation into account, calibrated intervals, transition counts. It never sees the reference answer. The **offline success criterion** is used only after the experiment ends: it compares the result with an independent internal reference and checks whether the method stopped too early. If the online decision looked at the reference, the agent could stop early because it is "close to the correct answer". This is reference leakage.

The reference does not use literature values. It uses a data pool run with the same force field, water model, temperature and workflow: 6 replicas are held out and run at full length, and no method uses them. Literature values come from different force fields and conditions. They would mistake force-field error for sampling error.

### What is measured

The quantity is the backbone basins. Each quantity is judged separately. A method can stop on populations alone, but it must state "converged" or "not converged" for each other quantity. Each wrong statement is counted separately.

| Quantity | Online: the method itself judges it is enough | Offline: really accurate |
|---|---|---|
| Population of each basin (main) | 95% interval half-width ≤ 0.05, and each replica has at least 4 transitions | max \|p − p*\| ≤ 0.05 |
| Free-energy difference between the two main basins | Interval half-width ≤ 0.5 kT | Error ≤ 0.5 kT |
| Transition rates in both directions | Interval of the relative error ≤ 20% | Relative error ≤ 20% |
| Angle distribution map | JS divergence between the two halves of the data ≤ 0.05 bit | JS divergence ≤ 0.05 bit on the grid cells where the reference has enough data |

These thresholds are candidate values that I chose. They are not literature standards. The transition count is only a diagnostic, not an accuracy metric, because it grows with simulation time.

A frame counts as arriving in a basin only when it enters the basin's core (rama default: at least 10° from the partition boundary; β-turn and `--rama-basins centres`: within 50° of the centre). Frames between two cores keep the previous basin. So crossing a boundary back and forth does not count as a transition, and the rates have a meaning.

### Two difficulty levels

- **A-Easy:** fast local φ/ψ conformational changes. A separate pilot run selects the residues: a residue qualifies only if its basin changes at least 4 times in every pilot trajectory. At most 5 are taken, the slowest of them. The data pool's own results must not be used for the selection.
- **A-Hard:** the β-I/β-II turn of D52/G53 in the 50s loop (ψ(D52), φ(G53)). In the literature, this flip happens about 7 times back and forth in about 8 µs of unbiased MD. NMR measures a time scale of about 13–15 µs, and β-II is only about 4% (Arasteh 2021, [doi:10.1021/acs.jpclett.1c00778](https://doi.org/10.1021/acs.jpclett.1c00778); Smith 2016, [doi:10.1073/pnas.1519609113](https://doi.org/10.1073/pnas.1519609113); Mukhopadhyay 2025, [doi:10.1007/s10858-025-00470-1](https://doi.org/10.1007/s10858-025-00470-1)). A 16 × 100 ns data pool sees one or two flips at most and cannot even build a reference. So A-Hard does not test accuracy. It tests only one thing: whether a method honestly says "insufficient evidence" (INSUFFICIENT_EVIDENCE) instead of wrongly declaring convergence. A real accuracy test needs a reference of tens of µs, which takes months at your GPU speed.

### Five methods

| Method | Replicas | Length | How it decides |
|---|---|---|---|
| Fixed plan (Fixed-Long 1 replica, Fixed-Parallel 3, 4, 5, 8 replicas) | Fixed | Fixed, scanned from 5 ns to 100 ns | No online decisions |
| Sequential-CI | 4 or 8, fixed | One look every 5 ns, stops when it reaches the precision | Conventional sequential stopping rule, intervals calibrated for all looks |
| Fixed plan after a pilot (pilot_fixed) | Starts with 3 | After a 5 ns pilot, the plan is set once by projection; after the run, one judgement only | What a careful user does without the agent. It uses the agent's own projection, but looks only once. It separates the value of "projection" from the value of "many looks" |
| Agent | Starts with 3 (the same as the plan in the chat), at most 8 | Jumps by projection to the length it needs | The first look is only for planning and cannot stop. After that, at most 3 more looks, with intervals calibrated for 3. Each time it chooses the cheapest mix of "extend" and "add replicas": it adds replicas only when the difference between replicas is statistically significant, because each new replica needs a new equilibration. When the projected cost is more than 3 times the budget, it says "insufficient evidence" early |

All methods are unbiased MD. Each new replica starts from the same prepared structure with a new random seed. No replica restarts from selected frames, because that would be adaptive sampling (experiment B). All methods use the same estimate: each replica first discards the unequilibrated start automatically (burn-in by the method of Chodera 2016, using only that replica's own data; the discarded part still counts as cost). When a chi-square test shows that the difference between replicas is real, the interval uses the t distribution with n−1 degrees of freedom.

**The agent under test is the agent that users really use.** The Agent in the replay calls `allocation.next_step`, the same code as daily runs (`--allocation projection`). Experiment A is not a separate set of rules written for the agent.

**Sequential-CI is the most important control.** If the agent does not save compute compared with it, the agent is only conventional sequential stopping and adds no value.

### What is reported

**Independent units.** Repeated trials that take different replicas from the same data pool share most trajectories. They are not independent experiments. So the report first states how many "independent units" there are, and all intervals across trials use this number. There are three ways: on synthetic models, each trial generates a new data pool and the reference is the true value (fully independent); `--disjoint` gives each trial a non-overlapping set of replicas (independent, but fewer trials); the default shares one data pool, and the report states that this is only about 1 independent unit.

**Calibration is per quantity.** Intervals are calibrated for each item and each basin. The most important metric is "coverage at stop": at the moment a method stops, does its reported interval contain the reference? It is reported per item (`per_item`). "All items correct at the same time" is reported separately, but the intervals are not calibrated for it.

For each method in each trial: success rate (declares convergence and is accurate), premature stop rate (declares convergence but is not accurate), number of honest "insufficient evidence" results, interval coverage, cost at stop C_stop, cost C_hit at which it first becomes accurate along its own path and stays accurate, GPU hours and waiting time. There is also a cost–accuracy curve (`experiment_A_*_cost_accuracy.svg`, data in the `.csv` of the same name).

**Verdict (written in advance):** Hypothesis A holds only if all three conditions are met.
1. The agent is reliable: success rate ≥ 90%, premature stop rate ≤ 10%.
2. The agent is better than every Sequential-CI and than the "fixed plan after a pilot": when the other method is reliable, the 95% interval of the paired saving on the same replicas is above 0; when the other method is not reliable, the agent must be reliable.
3. The agent costs less than every family of fixed plans at the same reliability. The cost of a fixed plan is the shortest length it needs to be 90% accurate.

When the reference itself is not precise enough, accuracy is always "undecidable". In that case the verdict is on **honesty** instead (the A-Hard verdict, set by the user on 2026-11-23). A trial counts as honest only if it meets all three conditions: (1) it outputs "insufficient evidence" and does not declare convergence; (2) the reason comes from the data (the item changes too rarely, or the projection is far over the budget; in daily runs, such a reason also comes with a suggestion to change the method or raise the limits); (3) the compute used before giving up is not more than half the budget (`Design.honest_share`). The method passes only if the fraction of honest trials reaches the reliability target (default 90%). The report has a separate section "A-Hard: honesty", with one row per method. Conventional rules such as Sequential-CI say insufficient evidence only when the budget is used up. So they fail condition (2). This is expected: they do not explain the reason.

**One thing to know:** the fixed-plan cost in condition 3 can be read only after all trials are done. Nobody can choose that length in advance. An online method needs calibrated intervals to be reliable. So in principle it pays about (its own interval factor ÷ that of the fixed plan)² times more. The report states this. It also lists, as a reference control, the plan an experimenter would write in advance (for example 1 × 100 ns; Huang and MacKerell 2013 validated ubiquitin side chains under CHARMM36 in this way, [doi:10.1002/jcc.23354](https://doi.org/10.1002/jcc.23354)).

### Dry runs on models (not 1UBQ results)

This section describes the old design from before the review revisions (start with 4 replicas, no burn-in, one shared data pool, so "100%" is only about 1 independent unit). For the results of the new design, see the next section.

The rules were first set on models with known answers, then frozen. They are not tuned on 1UBQ data afterwards. On a two-state model with fast transitions (70%/30%): the agent succeeded 100% of the time, never stopped too early, and all intervals covered the true value. It saved about 32% compared with both Sequential-CI variants, and 64% compared with 4 × 100 ns. But it cost more than the shortest reliable plan read afterwards (4 × 15 ns), so condition 3 gives the verdict "does not hold". On a model that imitates A-Hard (β-II 4%, about one flip per 10 µs): the reference itself was fully trapped in β-I, so the verdict was "undecidable". None of the three adaptive methods wrongly declared convergence. The agent gave up after 322 ns, Sequential-CI after 402 ns and 805 ns.

You can run this dry run yourself without a GPU:

```bash
python -m mdagent experiment-a --model "0.7,0.3@8" --trials 20
```

### Revisions after review (2026-11-23): harder synthetic models and sensitivity analysis

To make it possible for the test to fail, four families of synthetic models with known true values were added (`--family`). Each trial generates a new data pool: `equilibrium` (starts at equilibrium, the easiest), `biased_start` (all replicas start from a "crystal" substate that takes about 10 ns to relax), `two_timescale` (two groups of hidden states exchange about every 2 ns), `hidden_slow_start` (exchange about every 10 ns, and all replicas start on the same side; the first few ns look converged at 0.8, but the true value is 0.55).

```bash
python -m mdagent experiment-a --family hidden_slow_start --trials 30
python -m mdagent experiment-a --sensitivity equilibrium,biased_start,two_timescale,hidden_slow_start --trials 20
```

The second command changes one constant at a time (margin, margin before the last look, give-up factor, number of looks, starting number of replicas, minimum number of transitions) and checks whether the conclusion changes. The results go into `runs/experiment_A_model/`.

**Current results (2026-11-23, models only, not real MD):** 30 independent trials per model family, each with a new data pool of 16 replicas × 100 ns, budget 800 ns, tolerance 0.05. The rules are the final version: a burn-in rule with one start point shared by all replicas; add replicas when there are signs of a difference between replicas.

| Model | Agent: success / wrong stop / median compute | Other methods |
|---|---|---|
| equilibrium | 93% / 0% / 196 ns | Sequential-CI with 8 replicas: 100% / 0% / 205 ns |
| biased_start | 70% / 0% / 290 ns | Plan after a pilot: 83% / 0% / 302 ns; Sequential-CI with 4 replicas: 70% / 0% / 332 ns |
| two_timescale | 60% / 0% / 476 ns | Sequential-CI with 8 replicas: 83% / 0% / 565 ns |
| hidden_slow_start | 0% / 0% / 330 ns | All fail; only the "plan after a pilot" stops wrongly (7%) |

Conclusion: in 120 independent trials, the agent never stopped wrongly. When it cannot reach the precision, it says "insufficient evidence". But it saves a little compute only on the simplest model. Hypothesis A, "cheaper than conventional methods", still does not hold. `hidden_slow_start` shows that plain MD from one structure cannot see a hidden state that it never visits. For details and the first-round numbers, see section 3.4 of `docs/architecture.md`.

### How to run it

```bash
bash scripts/experiment_A.sh pilot     # Pilot: 3 replicas × 20 ns (60 ns in total, a few hours)
bash scripts/experiment_A.sh select    # Selects the A-Easy residues and writes them to runs/expA_easy.txt
bash scripts/experiment_A.sh pool      # Data pool: 16 replicas × 100 ns (1600 ns in total, 2 to 5 days depending on GPU speed)
bash scripts/experiment_A.sh analyze   # Replay and verdict, a few minutes; results in runs/expA_pool/experiment_A_easy.md and _hard.md
```

**A paper needs independent trials.** The default 16-replica data pool is only about 1 independent unit. To draw a conclusion, use a larger data pool and analyse it in non-overlapping groups, for example `POOL_REPLICAS=40 HOLDOUT=8 DISJOINT=1` (the other 32 replicas form 4 groups of 8). This takes several days of GPU time. You decide.

**Settings you decide:** the population tolerance 0.05 (`POPULATION`) and the reliability 90% (`RELIABILITY`) are candidate values. They set how strict the conclusion is. Set them **before** you see the results.

## Step 5: validating the decisions with real simulations

The logic tests (`tests/test_stop_reasons.py`, `tests/test_policy.py`) show only that the rules run as designed. Step 5 uses real simulations on a GPU to check whether the agent decides correctly. The plan works for any protein:

| Scenario | Example to look for | Expected |
|---|---|---|
| Reaches the precision normally | A surface side chain that flips fast; a salt bridge that easily forms and breaks | The stop reason is "target reached"; compared with a longer independent reference run, `compare` says they agree |
| All replicas trapped | A slow buried side chain (plain MD only); a very stable salt bridge | "Stalled", must not be judged as target reached; the report explains that an error of 0 means trapped |
| Budget used up before the target | A very high precision requirement and a very small budget | "Budget exhausted" when extension is not allowed; at most 2 times the budget when extension is allowed |
| Enhanced sampling | A side chain that does not flip in plain MD | Switches to enhanced sampling automatically and stops early when it reaches the precision; agrees with a reference run that does not stop early |

`scripts/step5_validation.sh` runs these scenarios one after another. At the end, it writes the stop reason of each run and the comparison results to `runs/val_<protein>_summary.txt`. There are two test proteins now:

- **1UBQ (human ubiquitin, default):** 76 residues, no ligand, about 17,000 atoms in the system. Surface side chains Glu16, Asp32, Lys63; buried Val26; surface salt bridge Lys11–Glu34; internal salt bridge Lys27–Asp52. The UniProt ID P62988 in the file is obsolete (it was split into P0CG47, P0CG48, P62979, P62987), so the script uses `--uniprot P0CG48`. Free ubiquitin is cleaved after Gly76, so the chain ends use `--termini charged`.
- **181L (T4 lysozyme L99A):** surface side chains Arg80, Lys135, Lys162; Val111; C-terminal salt bridge Lys162–Asp159; stable salt bridge Arg145–Glu11.

A full run takes about one night. You can also run only some scenarios (`bash scripts/step5_validation.sh 1UBQ budget_chi1 extend_chi1`). For another protein, pick the matching examples from the table and add a section to the script.

These ten scenarios check "whether the rules run as designed and decide correctly". They are regression tests. The core question of the project (a cheaper allocation) needs the experiment below.

## Core experiment: the same precision for less compute

One high-budget data pool plus replay can compare several allocation methods:

```bash
bash scripts/budget_experiment.sh 1UBQ          # Data pool of 10 replicas × 60 ns, then replay (about one day on one GPU)
POOL_REPLICAS=6 POOL_NS=20 bash scripts/budget_experiment.sh 1UBQ    # Try a small version first
```

The script does two things. First, it runs a data pool that makes no decisions (`--allow none`; it is data, not a strategy). Then `budget` replays each of the four strategies 10 times and lists the three precision criteria (`standard_error`, `confidence`, `sequence`) side by side. The results go into `runs/pool_1UBQ/budget_<criterion>.md`.

The data pool must be large enough in two directions, because all strategies are cut from it. It needs enough replicas (the replicas a strategy does not use serve as the independent reference), and enough length (longer than the budget any strategy can reach). The reference itself also has an error, and this error must count in the tolerance. So the report states its size. If it is close to the tolerance, make the data pool larger instead of trusting the `accurate` column.

A claim such as "64% saved" needs the same target quantity, the same precision requirement and the same error method, and it must hold across many independent repeats. It must not be one chosen win.

## How errors are computed

Neighbouring frames in a simulation are correlated. An error computed from the number of frames is too small. So within each replica, block averaging is used (Flyvbjerg and Petersen 1989): the trajectory is cut into consecutive blocks, the population is computed for each block, and the error comes from the differences between blocks. The block length starts at 1 frame and doubles until fewer than 8 blocks remain. The largest of these errors is taken, which is conservative. In enhanced sampling, frames have weights, and each block uses the weighted population.

When several replicas are combined, the population is the mean. The error is the larger of two values: the combined within-replica errors, or the spread between replicas divided by the square root of the number of replicas. When the difference between two replicas is more than 3 times their combined error (and more than 0.05), it is recorded as "replicas disagree".

Enhanced sampling also checks the weights: when the effective sample size (Kish) is below 50, the result depends on only a few frames and is marked `unreliable`.

**Error floor:** when a state never appears, block averaging gives an error of exactly 0. This is false precision. So the error cannot be smaller than a floor. Plain MD uses the formula for a two-state jump process: after N state transitions, the error of population p is about 2p(1−p)/√N, where N is the number of transitions in all replicas plus the number of replicas. Enhanced sampling uses √(p(1−p)/n), where n is the effective sample size of all replicas. In both cases p first gets a correction of half an observation, so a state that never appears also gets an error that reflects "seen too rarely". For well-sampled states, the floor is close to the block error and does not change the decision.

**Whether two runs agree (`compare`):** they agree when the population difference of each state is not more than twice the combined error (|z| ≤ 2), or when the difference is less than 0.01.

These checks can find undersampling, but they cannot prove that sampling is enough (Grossfield et al. 2018). So the agent also requires each replica to move back and forth between states enough times.

**The cost of repeated decisions.** Stopping as soon as the result "looks precise enough" is a sequential decision. Each round, the agent looks at the same growing data. The more looks, the higher the chance of one look that "seems precise enough" by chance. So the 95% interval at the moment of stopping is really too narrow. `--criterion` sets how to handle this (see "Goal, permissions and stop reasons"): `confidence` splits α over the number of looks known in advance; `sequence` uses a confidence sequence, which holds for any number of looks. The number of looks is known in advance, because it is the budget divided by the length of each round.

## Two modes: the agent decides, or the user specifies

- **The agent decides (default, `--mode auto`):** this mode is used when the user names no method or says "you decide". The simulation time and number of replicas that the user sets are limits. The agent analyses while it runs. It stops early when it reaches the target. It also stops early when more length or more replicas give no gain, and suggests in the chat to change the method (for example adaptive sampling or enhanced sampling). Changing the method, extending past the limit and adding more replicas all need the user's confirmation first.
- **The user specifies (`--mode guided`):** this mode is used when the user names a method (for example "only run plain MD", "use adaptive sampling"). The agent runs only this method, for the length that the user gives. When there is no progress for a long time, the agent sends a message in the chat that suggests a change of method, but it does not pause and does not change by itself. It changes the method only after the user agrees in the chat (for example "Stop X, then continue X with enhanced sampling").

**Messages from runs:** when a run in the background has something to tell the user, it writes a message in Chinese and English to `runs/<name>/messages.jsonl`. The web page checks every 10 seconds and the terminal every 20 seconds. They show new messages in the chat that started the run, with the run name at the start, for example "[181l_dev] The results are ready to use...". The language model also sees these messages the next time it answers. Runs send messages when: plain MD makes no progress for a long time, plain MD ends, the run switches to enhanced sampling, the enhanced sampling result is stable, the results are ready to use, and the run reaches its limit without becoming stable.

## Choosing methods and settings (literature)

At the start, the agent recommends methods and settings for the user's question. The basis is below. "Result" means the quantity the user cares about: populations of pocket side-chain conformations, pocket shape, binding affinity, and so on.

| The user's question | Recommended method | Length and replicas | How to judge it is enough | Basis |
|---|---|---|---|---|
| Side-chain conformations in a pocket (chi1) | Plain MD first; surface side chains flip in ps to ns. Use enhanced sampling for buried side chains that do not flip in tens of ns | At least 3, preferably 5 replicas. Val111 of T4 lysozyme L99A: about 0.04 flips per ns in plain MD; from some starting points, no flip in 60 ns | Populations agree between trajectories; enough flips | Burley 2019, doi:10.1021/acs.jctc.8b01018; Scouras & Daggett 2011, doi:10.1002/pro.565; Mobley 2007, doi:10.1021/ct700032n |
| Pocket shape or conformational ensemble | Many independent plain MD runs; cryptic pockets need enhanced sampling | 5 to 10 replicas, each of the order of 100 ns | Cluster populations are similar between trajectories | Knapp 2018, doi:10.1021/acs.jctc.8b00391; Oleinikovas 2016, doi:10.1021/jacs.6b05425 |
| Many short trajectories or one long trajectory | Many independent replicas | 5 to 10; with fewer than 5, the error estimate is not reliable | Use the spread between replicas as the error | Wan, Sinclair & Coveney 2021, doi:10.1098/rsta.2020.0082; Grossfield 2018, doi:10.33011/livecoms.1.1.5067 |
| Plain MD makes no progress for a long time | Well-tempered metadynamics, then reweighting | Bias factor about 10; stop when replicas agree | Free-energy differences between states no longer change with time; independent replicas agree | Barducci 2008, doi:10.1103/PhysRevLett.100.020603; Tiwary & Parrinello 2015, doi:10.1021/jp504920s |
| Explore rarely visited states | Adaptive sampling (restart from the least visited states) | Many rounds of short trajectories | No new states appear | Bowman 2010, doi:10.1021/ct900620b; Hruska 2018, doi:10.1063/1.5053582 |
| A value of binding affinity | Alchemical free-energy calculation (MDAgent does not do this; it recommends OpenFE or FEP+) | 11 to more than 20 λ windows, 1 to 5 ns each, at least 3 repeats | Enough overlap between neighbouring windows; forward and reverse agree; error about 1 kcal/mol | Mey 2020, doi:10.33011/livecoms.2.1.18378; Wang 2015, doi:10.1021/ja512751q |

Note: the current MDAgent default is 3 replicas, at the lower limit of the literature advice. In binding-affinity calculations, slow side chains (for example Val111) also cause errors. So it is still useful to check the pocket side-chain conformations with MDAgent first (Mobley 2007; Baumann 2021, doi:10.1021/acs.jpcb.0c10263).

## Starting a chat (`start`)

```bash
conda activate mdagent
python -m mdagent start
```

The program asks three questions in order. Press Enter to keep the last choice (stored in `runs/.chats/start.json`):

1. **Interface language:** Chinese or English.
2. **Language model:** Claude (with the Claude Code login) or ChatGPT (with the Codex login). When only one is installed on the computer, it does not ask.
3. **Where to chat:** web page (the browser opens `http://localhost:8765`) or terminal (in this window).

You can also set them directly and skip the questions: `python -m mdagent start --backend codex --ui terminal --lang zh`.

**The program controls the workflow, not the language model.** The language model does only one thing: it fills your words into a fixed settings form (`RunPlan`). Each item of the form matches one option of `mdagent run`. It cannot run commands and cannot see your files. It sees only a summary of the PDB file and the list of runs. MDAgent does every later step itself in a fixed order: it downloads the PDB by ID, checks the structure, makes the command, and starts it in the background only after you confirm. The web page and the terminal use the same program, so the steps are the same and the chat history is shared.

**Use as few tokens as possible:** each sentence you type calls the language model only once (twice when a PDB must be downloaded). The call turns off all Claude Code tools and does not use its long built-in prompt. Codex uses a lower reasoning effort. The list of runs tells the language model only how many result files of each kind there are, and does not list each file name. The terminal commands below do not use the language model at all and cost no tokens:

| Command | What it does |
|---|---|
| `/runs` | Lists all runs of this project: status, latest progress, and the computer and folder of the result files |
| `/last` | Shows the last reply in a window you can scroll up and down; press q to go back to the chat. Use it when a reply is too long for one screen. |
| `/report [run name]` | Shows the result report in the same window. Without a run name, it shows the latest run. |
| `/new` | Starts a new chat. Use a different chat for each case. |
| `/help` | Shows these commands |
| `/quit` | Exits. Running simulations do not stop. |

In the terminal, when the language model proposes to start, stop or continue a run, MDAgent shows the settings, the structure check report and the full command. Then it asks "Start the run? [y/N]". It runs only if you type y. If you type `/runs` or `/last` at this question, it runs the command first and then asks again. On the web page, a button does the same.

**In tmux, the scroll wheel scrolls the input history:** by default tmux does not handle the mouse and turns the wheel into up and down arrow keys. So it scrolls through the commands you typed, not the chat. Use `/last` to see the last reply. Or run `tmux set -g mouse on` so that the wheel scrolls the chat (to keep it, write `set -g mouse on` into `~/.tmux.conf`).

## The chat window (`chat`)

```bash
conda activate mdagent
python -m mdagent chat              # The browser opens http://localhost:8765
```

A local web page. The interface has two languages, Chinese and English; switch at the top right.

- **Language model backend:** `--backend claude` (Claude Code, with your Claude login), `--backend codex` (Codex, with your ChatGPT login), `--backend api` (needs `ANTHROPIC_API_KEY`). The default is `auto`, which selects the first one installed. They only understand your words and make proposals. They cannot run any command: all Claude Code tools are off, and Codex runs in read-only mode.
- **Confirm before execution:** starting, stopping and continuing a run first appear as a card with the settings and the full command. They run only after you click the button.
- **Runs do not depend on the web page:** each run is a separate process with its own process group. Its log goes to `runs/<name>.log`. If you close the web page, or even stop the chat server (Ctrl+C), the simulation continues. It ends only when you say "stop" in the chat window and click to confirm, or when the run finishes.
- **Chats:** the left side has the list of chats. Click "New chat" to start a new case. Each chat has its own history and list of runs, stored in `runs/.chats/`, and they stay after the chat server restarts. Use a different chat for each case so they do not mix.
- **Structure check:** after a run is proposed, the structure check runs first. If there are problems, the page shows the check report instead of the run button. You can say "keep only chain A", "keep the tag", or say clearly "continue despite the problems". When the check passes, you can expand the report on the card.
- **PDB ID:** when you mention a PDB ID in the chat (for example "Use 3HTB and look at the pocket around JZ4"), the agent first downloads it to `structures/3HTB.pdb`, looks at the chains and ligands in the file, and then proposes a run.
- **Restart the chat server:** after a code update, restart the chat server to change the web page. First press Ctrl+C in the original terminal, or run `pkill -f 'mdagent chat'`, then start it again. This stops only the chat server, not the simulations. If you forget to stop the old one, the new one reports that port 8765 is in use.
- **Choose where to save:** name a folder in the chat, for example "Save the results to /data/me/md" or "Save to ~/results". The run is saved in `<run name>/` inside this folder, and the card shows the full path before it starts. The project's `runs/<run name>` is a link (symbolic link) to it, so progress, messages, stop, continue and download in the chat work as usual. The log `runs/<run name>.log` stays in the project folder. If the folder already has files, the program refuses it; choose another one. If you do not name a location, the results are in the project's `runs/<run name>/`. On the command line, write `--out <any path>`; but a run started this way is not in `runs/`, and the chat cannot see it.
- **Where the result files are:** each run card on the right shows the computer name and full path of the result files. Below are "Download all (zip)" and a download link for each file. The zip contains `protein.pdb`, the plain MD and enhanced sampling trajectories, and the chi1 and bias potential data, with the same structure as the run folder. The files exist only on the computer that runs the simulation. If you open the web page on another computer (for example a Mac that opens `localhost:8765` of a server through SSH port forwarding), that computer does not have these files. Use the download links, or copy them from the server with `scp`. You can also ask in the chat "Where are the trajectories?".
- **Progress:** by default the right side lists only the runs started in this chat. You can switch to "All" to see all runs in this project (including runs started from the command line). The list refreshes every 10 seconds. You can also ask "How far has dev5 got?". The language model does not compute results itself. It only passes on the messages from the run.
- **Who can see it:** runs and chats are stored in `runs/` of this project folder on this computer and are not uploaded to GitHub. Anyone who opens the chat window in the same folder sees the same chats and runs. Others who clone the repository on their own computers cannot see yours.

If `ANTHROPIC_API_KEY` is set, Claude Code uses this key instead of your subscription login. The Codex backend has not been tested in a real environment yet.

## Using `/md` in Claude Code (mainly for developers)

This way is not MDAgent's own chat program. Claude Code itself does the work: we wrote instructions in `.claude/commands/`, and Claude Code follows them to run commands, read files and ask for your confirmation. So it uses more Claude Code quota, and the steps depend on Claude Code following the instructions, not on the program. For normal use, use `start` above.

On your own computer, activate the environment first, then open Claude Code in the MDAgent1 folder:

```bash
conda activate mdagent
cd ~/Desktop/MDAgent1
claude
```

Then type in Claude Code:

- `/md Use Amber, look at the pocket around benzene, check protonation first, development test`: Claude Code reads the PDB summary, turns the request into an `mdagent run` command, lists the settings for you to confirm, and after confirmation starts it in the background with `nohup`.
- `/md-status`: shows progress. When a run stopped midway, it asks whether to resume it.

The simulation runs in the background. Closing Claude Code or the terminal does not stop it. The text of the commands is in `.claude/commands/`.

## Describing a run in one sentence (`ask`, one-off command)

```bash
python -m mdagent ask --pdb examples/181L.pdb --out runs/a "Use CHARMM, look at the pocket around benzene, 0.1 M KCl, check protonation first"
```

Like `start`, it uses the Claude Code or Codex login (`--backend claude` or `--backend codex`; the default selects the first one installed). With an Anthropic API key, you can also use `--backend api`; then first run `python -m pip install -e ".[llm]"`. It is not a chat: you give one sentence, the program shows the settings and the command, and it starts after you type `y`. It is suitable for scripts.

The language model only fills a fixed settings form (`RunPlan`). Each item of the form matches one option of `mdagent run`. Items you do not mention stay empty and keep their default values. The program gives Claude a summary of the PDB file: the chains, the residue number ranges and the ligand names. So Claude does not need to guess ligand names. When a ligand you name is not in the file, it asks.

- When the request is not clear, Claude asks a question, and the program does not run.
- When the request is clear, the settings form becomes a normal `mdagent run` command and goes through the same checks as the command line. The program shows the target region, the settings that differ from the defaults and the full command. It starts the run only after you type `y`. Add `--yes` to skip the confirmation.
- If the safety classifier refuses the request, the API retries automatically with another model (server-side fallback). If it is still refused, the program reports an error and does not run.

## License

MIT license, see `LICENSE`. Anyone can use, modify and distribute this program, as long as they keep the copyright notice.

## General decision core (in development)

MDAgent is becoming a general scientific MD agent. The new decision core is in `mdagent/core/`. It reads the success requirement of each estimand in the task, uses competing hypotheses to explain why the requirement is not met yet, chooses the next action within the permissions and the budget, and writes each decision with its evidence to `decisions.jsonl`. The core knows no observable, simulation engine or system. The specific numerical estimates are in `mdagent/estimators.py`. Production runs still use the old workflow. The new core now follows each run in "shadow mode": after each round of plain MD and each enhanced sampling check, it records what it would do with the same data, but does not do it. The record is `decisions.core.jsonl` in the run folder. To turn it off, set `MDAGENT_SHADOW=0` before the run.

Finished runs can also be compared, without new simulations:

```bash
python -m mdagent shadow --run runs/val_1UBQ_reach_chi1 runs/val_1UBQ_trapped_chi1
```

It rebuilds each past decision point from the trajectories and enhanced sampling data in the run folder, asks the new core what it would do, and lists this side by side with what the run really did (same, different stop reason, new core stops earlier, new core continues, different action). It also writes the evidence for each decision to `decisions.core.replay.jsonl`. For newly started runs, use `--live` to compare the shadow decisions recorded during the run. For the architecture assessment, the plan and the evaluation method, see [docs/architecture.md](architecture.md).

## Next steps

**Research direction:** phase one is "when to stop, how many replicas" for plain MD (`budget` can already measure it). Phase two is "which state to restart from, how long each trajectory runs" for adaptive sampling. That cannot be replayed and must be really computed. Also, trajectories that start from selected frames cannot give equilibrium populations directly. Phase three is a benchmark on several proteins of different difficulty, with frozen strategies and independent random seeds. The side chains of 1UBQ flip quite fast, so at least one more system with slow motions and several metastable states is needed.

**Before the repository is made public and more testers are invited**, these validations must be done on a GPU computer:

1. Run 181L once completely with default settings, and confirm that Val111 is found and flips in enhanced sampling. Passed: Val111 did not flip in 3 plain MD replicas of 20 ns and was sent to enhanced sampling. In the first enhanced sampling trajectory, it flipped 373 times in 5 ns and visited all three states.
2. Run a development test with the Codex backend in the chat web page.
3. Run the core experiment once (`scripts/budget_experiment.sh`) and confirm on a real protein that early stopping really saves compute and that the answer is still within the tolerance. Use the result to decide which criterion is the default.


## GPU validation records

These records come from the developer's own GPU computers. On 2026-11-23 they were moved here from the project memory file, without personal information.

- **HP35 core distances (2026-11-24):** Phe47/Phe51/Phe58 CZ-CZ at 0.7 nm, 3 replicas, projection. It stopped with target_reached at 256 ns per replica, but looks 1-8 all said "too few state changes". `diagnose --each-round` showed that the burn-in cut away the active part of a replica (50 changes -> 2 kept), because a stretch without changes counted as many independent samples. Fixed with `uncertainty.indicator_samples` (independent samples at most the number of stays); `RULE_VERSION` 8. Replayed with the fix on the same data: all items ok at look 8 (128 ns per replica), so the run would have stopped there, at half the compute (384 instead of 768 ns).

- **The test machine (from 2026-11-19):** `~/.condarc` had `solver: classic`, and conda stuck at "Collecting package metadata"; `--solver=libmamba` fixed it. Plain `pip` ran the old system pip, so use `python -m pip`. CUDA reported a PTX version error (conda installed 13.5, but driver 610.57 supports only 13.3); after installing `cuda-version=13.3`, CUDA worked (1UBQ about 854 ns/day). RTX 4090 + Intel CPU. CPUs 8 and 9 (physical core 16) are unstable (all 44 segmentation faults in the system log are on these two CPUs). The fast tests had a segmentation fault in `addSolvent`, so run the tests and `mdagent start` with `taskset -c 0-7,10-31` (child processes inherit the limit). For how to check the logs, see `docs/developer.md`; `scripts/find_bad_cores.py` is in the "Troubleshooting" section of the README. On 11-19, step 5 (1UBQ) started on this computer. reach_chi1 finished and decided correctly: ASP32 reached the target in plain MD; LYS63 and GLU16 each had one replica that almost never flipped (rule: at least 4 per replica), so plain MD used the 20 ns limit, switched to enhanced sampling and reached the target. LYS63 is about 89% in g+ (+60°), which matches the main state in 5 plain MD replicas, so the reweighting is correct. LYS63 flips slowly and is not a good example of "plain MD reaches the target". On 11-20, all 10 scenarios of step 5 finished (about 20 hours), and all stop reasons matched reality (trapped_dist judged as target reached is correct: the Lys27–Asp52 distance went back and forth 5, 4 and 8 times in the three replicas, mostly "near", error 0.012; it was expected not to break). Three problems were found: the `diagnose` command diagnosed distance runs as chi1 by mistake (fixed: it now shares `agent.diagnose_run` with runs); a state that never appears has a block error of 0, so `compare` wrongly reported "disagree" (LYS63 g- z=3.5, VAL26 z=inf; an error floor was added, and `compare` also counts a difference below 0.01 as agreement); in `budget_chi1`, the plain MD budget was 2 ns, but after the switch to enhanced sampling it used the default 20 ns again (this conflicts with the rule that the user's values are limits).

- **Backbone φ/ψ gives up by budget (2026-11-23, code 3dd46d8):** A:110–112 of 181L (residues in a helix), 3 replicas, limit 1 µs per replica, 50 ns per round. The three residues stayed in alpha_R and never changed. At 151 ns, the rule judged that it was still worth running (3×849/151≈16.9 ≥ 12); at 201 ns, it judged that it was not (≈11.9 < 12). Plain MD stopped at 201 ns per replica and saved about 80% of the budget, as expected. A simulation confirmed: when such an old run moves to main, `resume --enhanced-sampling` sends the three residues to φ/ψ enhanced sampling (the old cache format is re-read automatically). This record was first only on the old branch `claude/compassionate-lamport-1xn8lc` and was moved here on 2026-11-24.

- **How the user opens the web page (not confirmed yet):** in the screenshot, the browser is Chrome on a Mac and the address is `localhost:8765`, but the run's process ID is 1749823. macOS does not use process IDs this large. So the simulation probably runs on a Linux computer, the web page is opened through SSH port forwarding, and the result files are on that Linux computer.

- **Items that can be validated only on the user's computer:** whether web page download links work through port forwarding (not validated yet); real calls of AdaptivePy (validated, works); Claude Code backend (validated, works); Codex backend (not validated yet; testers will use it, so validate it before going public); `start` in terminal mode with Claude through a development test (validated, works); the full 181L run stopped at about 00:23 on 11-16, at 5.4 ns of the first enhanced sampling trajectory, with no error in the log. A reboot (08:47 on 11-17, unclean shutdown) and an SSH disconnect (it ran in tmux, KillUserProcesses=no) are ruled out. The system log has no entries at all after 00:22:54 on 11-16 until the reboot at 08:47 on 11-17, so the whole machine froze (not a program or memory problem); the cause may be the GPU, driver, power supply or cooling. Now run/resume records in the log the process ID, the start time, and the call stack on a crash or SIGTERM, and progress lines have timestamps; long runs of CHARMM capped systems on a GPU (not validated yet); the first step 5 run (1UBQ reach_chi1, 11-19): the rule for adding replicas was too aggressive, and replicas grew from 3 to 10 (changed: add only when replicas clearly disagree and less than half the budget is used); later the speed fell from 646 to 4 ns/day, while one python process used 23.4 GB of GPU memory at 0% GPU use, which shows that OpenMM silently switched to the CPU (changed: always set the platform explicitly, and report an error when the GPU cannot be used); whether the process that used the GPU memory belongs to MDAgent is still being checked.
