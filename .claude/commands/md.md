---
description: 用一句话描述 MD 运行要求，转成 mdagent run 命令，确认后在后台启动
argument-hint: "[要求，例如：用 Amber，看苯周围的口袋，先检查质子化]"
---

用户的要求：$ARGUMENTS

你是 MDAgent 的操作窗口。把用户的要求转成一条 `python -m mdagent run` 命令，得到用户确认后在后台启动。按下面的步骤做，不要跳步。

1. **检查环境。** 运行 `python -c "import mdagent, openmm; print(openmm.__version__)"`。如果失败，告诉用户：先退出 Claude Code，运行 `conda activate mdagent`，再在 MDAgent1 文件夹里重新运行 `claude`。然后停下。

2. **确定输入文件。** 有两种输入：
   - 蛋白 PDB 文件（`--pdb`）：MDAgent 自己搭体系，配体会被删掉。
   - 用户自己准备好的 Amber 体系（`.parm7`/`.prmtop` 加 `.rst7`/`.inpcrd`，用 `--prmtop` 和 `--inpcrd`）：原样使用，配体保留。用户提到这类文件，或者要求保留配体时用这种。
   - 用户给出 PDB 编号（例如 3HTB）时，先运行 `python -m mdagent fetch <编号>`，它下载到 `structures/<编号>.pdb`；命令里用 `--pdb-id <编号>`，或者 `--pdb structures/<编号>.pdb`。
   用户没有提到文件时用 `examples/181L.pdb`，并在第 5 步告诉用户。文件不存在时问用户，然后停下。要保留配体但用户只给了 PDB 文件时，告诉用户需要先用 tleap 等工具准备好 Amber 体系，然后停下。

3. **读取 PDB 概况。** 运行：
   `python -m mdagent info --pdb <PDB 文件>`（Amber 体系用 `--prmtop <文件>`；Amber 文件里残基从 1 开始编号，没有链名，`--residues` 写成 `3-5` 这样不带链名的形式）
   它列出链、残基编号范围和配体名称。只能使用这里出现的配体名称和残基编号。用户说的配体或残基在这里找不到时（例如用户说 ATP，文件里没有），向用户提问，然后停下。

4. **转成命令。** 运行 `python -m mdagent run --help` 查看所有选项和默认值。规则：
   - 只设置用户明确要求的选项，其余保持默认。不要自作主张改参数。
   - 目标区域用 `--around-ligand <配体名>`（可加 `--cutoff-angstrom`），或 `--residues A:99-120`。
   - 用户说"开发测试"或"短一点"时，用很短的参数：`--replicas 2 --first-ns 0.1 --extend-ns 0.1 --max-ns 0.3 --no-flip-patience-ns 0.25 --enhanced-ns 0.1 --equil-ps 60`。
   - 输出目录用 `runs/<简短英文名>`，必须是不存在的目录（先用 `ls runs` 检查）。
   - 要求里有这些选项做不到的事情时，告诉用户，然后停下。

5. **请用户确认。** 用中文简短列出：PDB 文件、目标区域、哪些设置改了默认值、预计的模拟总长度，以及完整命令。问用户"要开始吗？"，然后停下，等用户回答。用户没有明确同意之前，不要运行。

6. **在后台启动。** 用户同意后运行：
   `mkdir -p runs && nohup python -m mdagent run <选项> --out runs/<名字> > runs/<名字>.log 2>&1 &`
   等 20 秒，运行 `tail -n 15 runs/<名字>.log`，确认已开始（看到 `Building the system` 或 `Round 1`）。如果日志里有 `error`，把错误告诉用户。

7. **告诉用户：** 运行已在后台开始，关掉 Claude Code 也不会停；日志在 `runs/<名字>.log`；以后用 `/md-status` 查看进度。

不要分析模拟结果（不算状态比例、不判断收敛）。这是项目的规则，见 CLAUDE.md。
