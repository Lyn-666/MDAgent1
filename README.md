# MDAgent: molecular dynamics from one sentence

Tell MDAgent what you want to know, for example *"Which rotamers do the side chains around this ligand take?"*. It fetches and checks the structure, builds the system, runs the simulations, and analyses them as they run. It stops as soon as the answer is reliable and writes a report.

Its main goal is **no wasted simulation**. It starts with 3 short replicas. Then it decides whether to extend them, add replicas, or stop. If plain MD cannot answer within your limits, it says so and suggests another method. It never exceeds your limits. The default limit is 10 replicas × 2 µs.

## What it answers

| Question | You get |
|---|---|
| Side-chain conformations | chi1 populations (g+, t, g−) with errors |
| A distance between two atoms (salt bridge, pocket opening) | Populations of the "near" and "far" states |
| Backbone conformations | Ramachandran basin populations, transition rates, and a φ/ψ free-energy surface |
| Whether a change happens | When and in which replica it is seen. The run stops once it is seen. |
| Structural stability | RMSD and radius of gyration: whether they are steady, their means, an RMSD–Rg free-energy surface, and conformational clusters. "Stable" means the structure was watched long enough: by default, at most 10 departures per µs (about 300 ns in total). Say *"stable for 1 µs"* for a stricter check. |

Plain MD is the default. When a conformation does not change, the agent can switch to enhanced sampling (metadynamics, with the bias removed afterwards). Adaptive sampling is optional. MDAgent does not compute binding free energies.

## Install

You need Linux with an NVIDIA GPU, conda, and [Claude Code](https://claude.com/claude-code) or Codex with your own login. You do not need an API key.

```bash
git clone https://github.com/Lyn-666/MDAgent1.git && cd MDAgent1
```

```bash
conda env create -f environment.yml && conda activate mdagent && python -m pip install -e .
```

```bash
python -m openmm.testInstallation
```

The CUDA line of the last command should say "Successfully computed forces".

## Example

```bash
conda activate mdagent && mdagent start
```

Choose a language, a language model and a web or terminal chat. The web chat is at `http://localhost:8765`. Then type one sentence that names **the system** (a PDB ID or file) and **the question**:

- *"Download 3HTB and look at the side chains within 5 Å of ligand JZ4."*
- *"Use /data/me/protein.pdb. Backbone conformations of A:45 to A:48. At most 3 replicas of 100 ns."*
- *"Download 1YRF and check whether the whole protein stays folded."*

The agent first checks the structure and asks you if something is wrong. Then it shows its plan. Nothing runs until you confirm. The simulation runs in the background, so you can close the chat. Later, ask in the chat: *"How far is run1?"*, *"stop run1"*, *"add 2 replicas to run1"*.

You can also set these in your sentence: precision, limits, a fixed plan ("exactly 3 replicas of 10 ns"), permission to switch method, temperature (default 300 K), ions (0.15 M NaCl), force field (CHARMM36m), pH (7), and the save folder. If something is unclear, the agent asks instead of guessing.

**Tips:**

- For a new system, add *"at most 3 replicas of 100 ns"*. That is less than a day on a recent GPU.
- Add *"development test"* to check the whole workflow in a few minutes.
- Ligands are removed. To keep one, give the paths of a prepared Amber system (`.prmtop` and `.inpcrd`).

## Results

Each run has a folder, `runs/<run name>/`. The report `report.en.md` gives every result with its error, whether it met your precision, each decision the agent made, why it stopped, and the compute it used. The Chinese version is `report.zh.md`. The folder also holds the trajectories (`replica_*.dcd`, with `protein.pdb` as topology), plots and data tables.

"Insufficient evidence" is a valid answer. It means your limits did not give enough data for the precision you asked for, and the report suggests what to do next.

## More

- [TESTING.md](TESTING.md): how to join the beta test and send feedback.
- [docs/developer.md](docs/developer.md): all options, algorithms, and validation.
- License: MIT.
