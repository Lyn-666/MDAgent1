---
description: Show the progress of MDAgent runs (running and finished)
argument-hint: "[run name, optional]"
---

The user wants to know how a run is doing. The run to check: $ARGUMENTS (if empty, the most recently modified run in `runs/`).

1. Run `pgrep -af "mdagent (run|resume)"` to see which runs are still going.
2. Run `ls -t runs/ | head` to find the run folder and its log (usually `runs/<name>.log`).
3. Run `tail -n 40 <log file>`. If `runs/<name>/enhanced/replica_*.json` exist, read them too: `ns_done` is the number of ns of enhanced sampling done.

Then tell the user briefly, in their language:
- whether the run is still going, has finished, or stopped part way (no process, and no `Summary` at the end of the log);
- the current stage (building the system, plain MD round N, or enhanced sampling) and roughly how much is done;
- the latest decision in the log (e.g. "run all replicas 0.1 ns more", "side chains handed to enhanced sampling");
- if finished: where the output files are. The Summary at the end of the log lists the computer name and full path. Give the user the full path, and remind them that the files are on the computer that ran the simulation;
- if it stopped part way: `nohup python -m mdagent resume --run runs/<name> > runs/<name>.resume.log 2>&1 &` continues from the last saved point. Ask the user first, and run it only after they agree;
- if the log contains an error: the exact error text.

Do not compute simulation results yourself. The program computes them: enhanced-sampling populations are in `runs/<name>/enhanced/populations.json`, and messages from the run are in `runs/<name>/messages.jsonl`. Only relay what these files say.
