# MDAgent: Molecular Dynamics from a Single Sentence

**An autonomous, uncertainty-aware agent for molecular dynamics simulations.**

Describe what you want to learn about a molecular system in natural language. MDAgent handles structure retrieval, system preparation, simulation execution, trajectory analysis, and scientific reporting — all within your specified computational budget.

Unlike conventional MD workflows that require predefined simulation lengths and replica counts, MDAgent continuously evaluates simulation progress and statistical uncertainty to determine whether additional sampling is needed.

**The goal is simple: simulate only as much as necessary to answer your scientific question.**

## Key Features

- **Natural language interface** — Describe your scientific question without writing simulation scripts or configuration files.
- **Automated MD workflow** — Retrieve and validate structures, prepare molecular systems, execute simulations, and analyze trajectories.
- **Uncertainty-aware decision-making** — Evaluate statistical convergence and sampling adequacy throughout the simulation.
- **Dynamic resource allocation** — Decide whether to extend existing trajectories, launch additional replicas, or terminate sampling.
- **Intelligent sampling strategy** — Identify sampling bottlenecks and recommend or initiate alternative sampling methods when permitted.
- **Budget-aware execution** — Respect user-defined limits on simulation length and replica count.
- **Transparent scientific reporting** — Document results, uncertainties, methodological decisions, and stopping criteria.
- **Human-in-the-loop control** — Review simulation plans, monitor progress, and intervene at any time.

## How It Works

MDAgent transforms a natural language scientific question into an iterative, evidence-driven simulation workflow:

1. **Understand** — Interpret the scientific question, identify target observables, and establish sampling requirements.
2. **Prepare** — Retrieve or load the molecular structure, validate the system, and prepare simulation inputs.
3. **Plan** — Propose simulation protocols, analysis methods, convergence criteria, and computational limits.
4. **Simulate** — Launch an initial set of independent MD replicas.
5. **Analyze** — Monitor target observables, statistical uncertainty, conformational transitions, and sampling progress.
6. **Decide** — Extend trajectories, allocate additional replicas, recommend alternative sampling methods, or stop.
7. **Report** — Summarize scientific findings, remaining uncertainties, computational costs, and the rationale behind each decision.

By default, MDAgent starts with three short independent replicas and iteratively allocates additional sampling based on the accumulated evidence.

The default computational limit is 10 replicas, each with a maximum simulation length of 2 µs. Users can override these limits according to their scientific objectives and available resources.

## Supported Scientific Questions

| Scientific Question | Analysis and Output |
|---|---|
| Which side-chain conformations are populated? | χ1 rotamer populations (g+, t, g−) with uncertainty estimates. |
| How does an interatomic distance fluctuate? | Distance distributions and populations of near and far states, applicable to salt bridges, pocket opening, and other structural interactions. |
| Which backbone conformations are sampled? | Ramachandran basin populations, transition rate estimates, and φ/ψ free-energy surfaces, where sampling is sufficient. |
| Does a specific conformational transition occur? | The first observed transition time and the corresponding replica. Sampling can terminate once the predefined event is observed. |
| Does the structure remain stable? | RMSD, radius of gyration, structural stability assessment, RMSD–Rg free-energy surfaces, and conformational clustering. |

For structural stability assessments, MDAgent evaluates both structural fluctuations and the amount of accumulated sampling. By default, the stability criterion corresponds to a target upper departure rate of 10 events per µs, requiring approximately 300 ns of aggregate observation when no departures are detected under the assumed event-rate model.

More stringent requirements can be specified directly in natural language, for example:

> Assess structural stability over a 1 µs timescale.

### Supported Sampling Methods

- **Conventional molecular dynamics (MD)** — The default sampling method.
- **Enhanced sampling** — Metadynamics with reweighting for supported equilibrium observables.
- **Adaptive sampling** — Optional sampling strategy.

When conventional MD fails to provide sufficient evidence within the allocated budget, MDAgent identifies the sampling limitation and recommends an appropriate alternative when supported.

MDAgent currently does not support binding free-energy calculations.

## Installation

### Requirements

- Linux
- NVIDIA GPU with CUDA support
- Conda
- [Claude Code](https://claude.com/claude-code) or [Codex](https://chatgpt.com/) with an authenticated account

No separate API key is required when using the supported authenticated CLI backends.

### 1. Clone the Repository

```bash
git clone https://github.com/Lyn-666/MDAgent1.git
cd MDAgent1
```

### 2. Create the Environment

```bash
bash scripts/install.sh
conda activate mdagent
```

The script checks your NVIDIA driver and installs one of two fixed, tested combinations, so everyone runs the same versions:

| NVIDIA driver supports | Installed |
|---|---|
| CUDA 12.9 or newer (driver 575 or newer, including all CUDA 13 drivers) | CUDA 12.9 + OpenMM 8.6.1 |
| CUDA 12.6 to 12.8 (driver 560 to 574) | CUDA 12.6 or 12.8 + OpenMM 8.4.0 |

Older drivers must be updated first. Both combinations support NVIDIA GPUs from Maxwell (2014) to Blackwell (RTX 50 series). To see first what it would install, run `bash scripts/install.sh --check`. Running it again repairs an existing `mdagent` environment.

### 3. Verify the Installation

```bash
python -m openmm.testInstallation
```

Confirm that the CUDA platform reports:

```text
Successfully computed forces
```

## Quick Start

Launch MDAgent:

```bash
conda activate mdagent
mdagent start
```

Select your preferred language, language model, and interaction interface.

MDAgent supports both terminal-based interaction and a local web interface:

http://localhost:8765

To start a simulation, describe your molecular system and scientific question in natural language.

### Example 1: Side-Chain Conformations Near a Binding Site

> Download 3HTB and characterize the side-chain rotamer populations of residues within 5 Å of ligand JZ4.

### Example 2: Backbone Conformational Sampling

> Use /data/me/protein.pdb to investigate the backbone conformations of residues A:45–A:48. Use at most 3 replicas of 100 ns each.

### Example 3: Structural Stability

> Download 1YRF and assess whether the protein remains structurally stable throughout the simulation.

### Example 4: Automated Sampling Strategy

> Investigate the conformational populations of the selected residues using conventional MD. If sampling is insufficient, recommend an alternative sampling strategy before exceeding the computational budget.

MDAgent first validates the molecular structure and requests clarification if necessary. It then presents a proposed simulation plan for your approval.

**No simulation is launched until you confirm the plan.**

Once launched, simulations continue independently of the chat interface. You can return later to monitor or modify the workflow using natural language:

- "What is the current status of run1?"
- "Show the latest convergence analysis for run1."
- "Add two replicas to run1."
- "Stop run1."

## Simulation Configuration

Simulation parameters and scientific requirements can be specified directly in natural language.

| Parameter | Default or Supported Configuration |
|---|---|
| Sampling method | Conventional MD |
| Initial replicas | 3 |
| Maximum replicas | 10 |
| Maximum simulation length per replica | 2 µs |
| Temperature | 300 K |
| Ionic concentration | 0.15 M NaCl |
| Force field | CHARMM36m |
| System preparation pH | 7.0 |
| Sampling precision | Configurable |
| Method switching | Subject to user authorization |
| Output directory | Configurable |

### Fixed Simulation Protocol

You can request a predefined simulation protocol:

> Run exactly 3 independent replicas of 100 ns each using CHARMM36m at 310 K.

### Autonomous Resource Allocation

Alternatively, allow MDAgent to dynamically allocate computational resources:

> Estimate the conformational populations with sufficient statistical precision. Decide how to allocate simulation time and replicas, but do not exceed my computational budget.

### User-Controlled Sampling Strategy

You can explicitly specify whether MDAgent is allowed to change the sampling method:

> Start with conventional MD. If sampling is insufficient, recommend enhanced sampling and ask for my approval before switching methods.

Alternatively:

> Automatically select an appropriate sampling strategy to achieve the requested precision within my computational budget.

When required information is missing or ambiguous, MDAgent requests clarification rather than silently making assumptions.

## When Does MDAgent Stop?

MDAgent terminates sampling when one of the following conditions is met:

- **Target reached:** The requested event or scientific objective has been achieved.
- **Sufficient evidence:** The predefined statistical precision and sampling requirements have been satisfied.
- **Sampling stalled:** A sampling bottleneck is detected, and further conventional MD may be inefficient under the current allocation.
- **Budget exhausted:** The user-defined computational limit has been reached.
- **User intervention:** The user explicitly requests termination.

The stopping decision is accompanied by a documented scientific justification.

Importantly, MDAgent distinguishes between obtaining a sufficiently supported result and simply completing a predefined simulation length.

## Practical Considerations

- For an initial investigation of a new molecular system, consider specifying a conservative computational budget, such as "at most 3 replicas of 100 ns each."
- Include "development test" in your request to perform a short end-to-end workflow test.
- Ligands are removed during default system preparation. Ligand-defined residue selections can be used to identify regions of interest, but simulations that retain ligands require a preparameterized system.
- To simulate a ligand-containing system, provide prepared Amber topology and coordinate files (`.prmtop` and `.inpcrd`).
- Actual simulation throughput depends on system size, hardware, and simulation settings.

## Results and Reproducibility

Each simulation run generates a dedicated output directory:

```text
runs/<run_name>/
├── report.en.md
├── report.zh.md
├── protein.pdb
├── replica_*.dcd
└── ...
```

The scientific report includes:

- Target observables and estimated values.
- Statistical uncertainties and convergence assessments.
- Whether the requested scientific and precision criteria were satisfied.
- Simulation allocation and sampling decisions.
- The rationale for terminating or continuing sampling.
- Total computational resources used.
- Remaining limitations and recommended next steps.

Reports are available in both English and Chinese. Trajectories, structural files, analysis figures, and numerical results are retained for subsequent inspection and independent analysis.

### Insufficient Evidence Is a Valid Result

Not every scientific question can be answered within a finite simulation budget.

If the available trajectories do not provide sufficient statistical or conformational sampling, MDAgent explicitly reports **INSUFFICIENT EVIDENCE** instead of presenting an unsupported conclusion.

The report identifies the unresolved scientific question, explains the sampling limitations, and recommends potential next steps.

## Project Philosophy

Traditional MD workflows often require users to determine simulation lengths and replica counts before knowing how difficult the underlying sampling problem will be.

This can lead to two common problems:

- **Oversampling:** Computational resources are spent on simulations that have already satisfied the scientific objective.
- **Undersampling:** Simulations terminate at a predefined length despite insufficient statistical evidence or inadequate conformational exploration.

MDAgent takes a different approach: computational resources are allocated iteratively according to the scientific objective, accumulated evidence, and remaining uncertainty.

Rather than asking:

> How long should I run this simulation?

MDAgent asks:

> What additional sampling is necessary to answer this scientific question?

The objective is not merely to automate molecular dynamics, but to make simulation workflows more scientifically informed, computationally efficient, transparent, and reproducible.

## Documentation

- [TESTING.md](TESTING.md) — Beta testing, example workflows, and feedback.
- [Developer Documentation](docs/developer.md) — Configuration options, algorithms, decision-making mechanisms, and validation procedures.

## License

MDAgent is released under the MIT License.
