# Testing MDAgent (version 0.1.0b2, closed beta)

Thank you for helping. MDAgent runs molecular dynamics from one sentence. You say what you want to know. The agent builds the system, runs and analyses the simulations, stops when the answer is reliable, and gives you a report. Its goal is to run no unnecessary simulation.

This test asks three questions:

1. Can the agent do what you want **on a system you choose**, from one sentence?
2. Does it stop at sensible times? Does it waste time, or conclude too early?
3. Can you understand its explanations in the chat and its final report?

**Time:** about 30 minutes to install and 10 minutes for the workflow test. A real test on one system usually takes a few hours to two days of GPU time. You do not need to watch it.

**You need:**

- a Linux computer with an NVIDIA GPU and conda;
- Claude Code or Codex, installed and logged in with your own account;
- a GitHub account. Send your username to the person who invited you. You need it to accept the invitation and to send feedback.

**New in 0.1.0b2** (if you tested 0.1.0b1, update as in "Update before each test"):

- **Structural stability:** ask whether a protein or a region stays folded. You get backbone RMSD, radius of gyration, an RMSD–Rg free-energy surface and conformational clusters. "Stable" means the structure was watched long enough: by default the rate of leaving the start structure is at most 10 per µs (about 300 ns in total). You can add clustering to any other question too.
- **Temperature** can be set in your sentence (default 300 K).
- **Fixed:** the agent could discard the part of a trajectory with the most changes and then run much longer than needed. On a real test run this halved the compute.
- **Clearer messages:** when the agent runs longer, it now names the item and the replica that hold it back.

## Step 1: Accept the invitation and install

**1. Accept the invitation.** GitHub sends you an email. Click the link in it, or log in to GitHub and open https://github.com/Lyn-666/MDAgent1/invitations. The repository is private, so you cannot see it until you accept.

**2. Let this computer download the private repository.** GitHub does not accept passwords for downloading code. The easiest way is GitHub's command-line tool `gh` with a one-time browser login:

```bash
conda install -n base -c conda-forge gh -y
```

```bash
gh auth login
```

Choose GitHub.com → HTTPS → Yes (authenticate git with your GitHub credentials) → Login with a web browser. The screen shows an 8-character code and a web address. Open the address in a browser on any computer or phone, log in to GitHub, and enter the code. The server itself does not need a browser.

**3. Download and install:**

```bash
gh repo clone Lyn-666/MDAgent1 && cd MDAgent1
```

```bash
conda env create -f environment.yml && conda activate mdagent && python -m pip install -e .
```

## Step 2: Check the installation

```bash
python -m openmm.testInstallation
```

The CUDA line should say "Successfully computed forces".

```bash
pytest -q tests -m "not slow"
```

This takes about 4 minutes, and all tests should pass. If a step fails, look at "Troubleshooting" at the end of this file first. If that does not help, report it as in step 6. A failed step is a useful test result too.

## Step 3: Workflow test (about 10 minutes)

```bash
conda activate mdagent && mdagent start
```

Choose the interface language, the language model (Claude or ChatGPT), and a web or terminal chat. Then type:

> Download 181L, look at the side chain of A:111, development test, continue despite structure problems.

You should see the following. The agent checks the structure. Then it shows a card with the task and the settings as it understood them. After you confirm, the run finishes within a few minutes, and the chat tells you where the report is. A "development test" is very short, so its results have no scientific meaning. It only shows that the whole workflow runs.

## Step 4: The real test (your own system)

Choose a question **that you care about and whose answer you partly know**. Then you can judge whether the results and the stopping point make sense. Tell the agent in one sentence, for example:

> Download XXXX and look at the side chains within 5 Å of ligand YYY. At most 3 replicas of 100 ns.

> Use my file /path/to/protein.pdb and look at the backbone conformations and φ/ψ free-energy surface of A:45 to A:48. At most 3 replicas of 100 ns.

> Download XXXX and look at the distance between Lys11 NZ and Glu34 CD of chain A; within 0.45 nm counts as formed. At most 3 replicas of 100 ns.

> Download XXXX and check the stability of the whole protein (backbone RMSD, radius of gyration, free-energy surface and clusters). At most 3 replicas of 100 ns.

Tips:

- **For the first test, add "at most 3 replicas of 100 ns".** That is at most 300 ns, which is less than a day on a recent GPU. Without it, the default limit is 10 replicas × 2 µs, and slow questions can run for a long time.
- **Use the residue numbers from the PDB file.** If you are not sure, the agent asks you.
- **The structure check may stop and ask you**, for example "Several identical chains: which one should I keep?". Answer in one sentence. If you get stuck at this step, please report it.
- **Ligands are removed**, and only the protein is simulated. To keep a ligand, prepare an Amber system yourself (`.prmtop` and `.inpcrd`) and give the agent the two file paths in the chat.
- You can close the terminal or web page after the run starts. The simulation keeps running. Run `mdagent start` again to ask about progress in the same chat, for example "How far is run1?".

If you have time, do a second test. For example, give no limits and let the agent decide everything. Or add "switch to enhanced sampling if plain MD fails".

## Step 5: Take notes as you go

- **Translation:** Are the settings on the card what you wanted? Did the agent misunderstand anything?
- **Structure check:** Did it stop you? Was the explanation clear? Did you know how to answer?
- **During the run:** Did the chat explain each decision (extend, add replicas, stop)? Was there a moment when you thought it was wasting time or stopping too early?
- **At the end:** Was the stop reason sensible? Do the results match your experience or the literature? Was the report clear?
- **Cost:** The report states how much compute was used. Was it worth it? How long would you have run yourself?

## Step 6: Send feedback

Open an issue in the GitHub repository:

- If something failed, or did not behave as described, choose "Problem report".
- If nothing failed but you have impressions or suggestions, choose "Test feedback" and answer the questions from step 5.

**Please do not push code to the repository.** As a collaborator you technically can, but during the test please give feedback only through issues.

Please attach:

- the sentence you typed in the chat;
- the log file `runs/<run name>.log` (its first line gives the MDAgent version);
- the report `runs/<run name>/report.en.md`.

The trajectories are large, so do not upload them.

**Your data:** structures, trajectories and results stay on your own computer, and the program uploads nothing. The language model in the chat sees only three things: your sentences, a summary of the PDB file (chains, residue number ranges, ligand names), and a summary of the run list (names, status, progress). It cannot see coordinates or trajectories, and it cannot run commands.

## Update before each test

```bash
cd MDAgent1 && git pull && conda activate mdagent && python -m pip install -e .
```

If `git pull` says `Permission denied` or asks for a password, run `gh auth setup-git` once and try again.

If the chat is still running, stop it with Ctrl+C and run `mdagent start` again. This stops only the chat, not the simulations.

## Known limitations (this is a beta)

- The agent's default planning (extend, add replicas, stop when done) has been checked on synthetic models and in automated tests, but has little experience with real proteins yet. That is exactly where we need your help.
- Ligands are removed unless you provide a prepared Amber system.
- Distances cannot use enhanced sampling yet, and have no free-energy surface yet.
- Replicas run one after another on one GPU. Several GPUs are not used at the same time.
- The main target is Linux with an NVIDIA GPU. macOS is only good for tests of a few minutes. On Windows, use WSL2.
- The chat with a ChatGPT (Codex) login has not been tried in a real setup yet. If you use it, please watch it closely and report what you see.

## Troubleshooting

- **`No module named 'mdagent'`:** reinstall in the current folder with `python -m pip install -e .`.
- **`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`:** your driver is older than the CUDA packages. Run `conda install -c conda-forge cuda-version=X.Y --solver=libmamba`, where X.Y is the CUDA version that `nvidia-smi` shows.
- **conda hangs at "Collecting package metadata":** add `--solver=libmamba`.
- **Random segmentation faults:** some CPU cores may be unstable. `python scripts/find_bad_cores.py` finds them and prints a `taskset` command that avoids them.
