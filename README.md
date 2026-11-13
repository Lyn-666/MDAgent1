# MDAgent1

一个 MD agent：用户给出蛋白 PDB 和关心的区域，agent 自动搭体系、跑 MD，找出这个区域里没有采样好的侧链。agent 自己决定跑多久：每轮结束后诊断，再决定续跑还是停止。之后的版本会对卡住的侧链自动做增强采样。

## 工作流程

1. **读入目标区域。** 可以直接给出残基（如 `A:99-120`），或者给出"某个配体周围 N Å 内的残基"。
2. **搭体系。** 每项设置都有默认值，用户可以改成别的。见下面的"搭体系的选项"。
3. **分轮跑 MD。** 几条独立的轨迹同时推进，它们只有随机种子不同。第 1 轮默认每条 1 ns，以后每轮每条加 5 ns。
4. **每轮结束后诊断。** 统计每个目标残基的 chi1 二面角在 g+、t、g- 三个状态之间翻转了几次，然后按"卡住程度"排序。
5. **每轮结束后决定。** agent 对每个残基做出以下决定之一，并写下理由：

| 决定 | 条件 |
|---|---|
| `done` | 每条轨迹都翻转了足够多次，采样够了。 |
| `extend` | 普通 MD 还有用：已经看到翻转但次数不够，或者还没跑到放弃的时长。 |
| `enhanced_sampling` | 跑了很久（默认每条 10 ns）仍然一次都没翻转，或者总预算用完了。普通 MD 回答不了这个问题。 |
| `skip` | Gly 或 Ala，没有 chi1。 |

只要还有残基是 `extend`，并且没超出预算（默认每条 20 ns），agent 就再跑一轮。否则停止。

**诊断结论的含义：**

| 结论 | 含义 |
|---|---|
| `replicas_disagree` | 不同轨迹停在不同状态，结果取决于看哪一条，说明没有收敛。 |
| `no_flip` | 所有轨迹都没有翻转。侧链可能被卡住了，也可能本来就只有一个稳定状态，普通 MD 分辨不了这两种情况。 |
| `rare_flips` | 翻转过，但次数太少，还不能估计各状态的比例。 |
| `ok` | 每条轨迹都翻转了足够多次。 |

**排序方法：** 按"平均多久翻转一次"排序，越慢越靠前。这个时间等于总模拟时间除以翻转次数。没有翻转时，只能知道它大于总模拟时间，表里写成 `>`。时间相同时，轨迹之间越不一致的越靠前。

## 完整流程（按顺序）

1. **输入：** PDB 编号（下载到 `structures/<编号>.pdb`）、PDB 文件，或自己准备好的 Amber 体系（`--prmtop`/`--inpcrd`，跳过第 2、3 步，原样使用）。
2. **结构检查：** 和 UniProt 参考序列比对，找标签、多拷贝、金属、未知残基、长缺口、断链。有问题就停，除非 `--accept-problems`。见下面"结构检查"。
3. **处理结构（PDBFixer），依次：**
   1. 只保留选定的链（`--chains`）；删掉链两端有坐标的标签。
   2. 删掉所有非蛋白分子：晶体水、离子、配体、辅因子、末端帽（ACE、NH2、NME）。
   3. 链中间缺的残基建模补上；链两端缺的不补。
   4. 非标准残基换成标准氨基酸（例如 MSE → MET）。
   5. 补缺失的重原子。
   6. 删掉原有的氢，按 pH 7（或 PROPKA、或手动指定）重新加氢；His 的氢加在 ND1 还是 NE2 由氢键决定。
   7. **链的两端（`--termini`，默认 `auto`）：** 结构检查把每条链第一个和最后一个有坐标的残基对到 UniProt 全长序列上。如果真实蛋白在这一端还没结束（例如只解析了一个结构域，或者两端缺了残基），这一端加中性帽：N 端 ACE，C 端 NME（N-甲基酰胺）。如果这一端就是真实蛋白的起点或终点，保留带电末端：N 端 NH₃⁺，C 端 COO⁻。没有参考序列时，按结构两端是否缺残基判断。`--termini charged` 一律带电，`--termini capped` 一律加帽。用 CHARMM 时，帽的原子并入末端残基（CHARMM 的 ACE、CT3 补丁）；用 Amber 时，ACE、NME 是单独的残基。
   8. 加水盒子（默认立方体，蛋白到每个面至少 1.0 nm，盒子最窄处至少为两倍截断加 10%），先中和电荷，再加 0.15 M NaCl。
   9. 力场：默认 CHARMM36m + CHARMM TIP3P（截断 1.2 nm，从 1.0 nm 开始平滑减小，无长程色散修正）；也可选 Amber ff14SB + TIP3P 或 ff19SB + OPC（截断 1.0 nm，有长程色散修正）。静电用 PME，含氢的键用刚性约束。
4. **平衡（每条轨迹 600 ps）：** 骨架 N、CA、C 加 1 kcal/mol/Å² 的约束 → 能量最小化 → NVT 从 5 K 升温到 300 K（100 ps，分 20 步）→ 带约束的 NPT（200 ps）→ 分 20 步撤掉约束（200 ps）→ 无约束的 NPT（100 ps）。积分器 Langevin，步长 2 fs，摩擦 1/ps；压强 1 bar。
5. **普通 MD，分轮：** 3 条轨迹，第 1 轮每条 1 ns，以后每轮加 5 ns，每条最多 20 ns；每 5 ps 存一帧蛋白坐标。可选自适应采样（`--sampling adaptive`）。
6. **每轮结束后诊断：** 统计目标区域每个侧链 chi1 的翻转次数，决定"采样够了""续跑"或"交给增强采样"（跑满 10 ns 仍没翻转）。
7. **增强采样：** 对需要的侧链，在 chi1 上做 well-tempered metadynamics（bias factor 10，高斯峰 1.2 kJ/mol，宽 12°，每 1 ps 加一次），2 条 × 20 ns，从普通 MD 的最终状态开始。只保存数据，不做分析。
8. **输出：** 都在 `runs/<名字>/` 里，见下面"运行目录里的文件"。

## 两种输入

- **PDB 编号（`--pdb-id 3HTB`）：** MDAgent 从 RCSB PDB 下载结构，存为项目文件夹里的 `structures/3HTB.pdb`，然后像 `--pdb` 一样使用。已经下载过的不会重复下载。只下载不运行：`python -m mdagent fetch 3HTB 181L`，会打印每个文件的位置和内容概况。只有 mmCIF 格式的大结构暂时不支持。`structures/` 不会上传到 GitHub。
- **蛋白 PDB 文件（`--pdb`）：** MDAgent 自己搭体系，可用下面"搭体系的选项"调整。配体会被删掉，因为 OpenMM 自带的力场没有配体参数。
- **自己准备好的 Amber 体系（`--prmtop X.parm7 --inpcrd X.rst7`）：** 例如用 tleap 加 GAFF 参数准备的蛋白-配体复合物。MDAgent 原样使用，不增删任何原子，配体和所有参数都保留。文件里必须有周期性盒子（已加水）。非键截断 1.0 nm，PME。Amber 文件里的残基从 1 开始重新编号；用 `python -m mdagent info --prmtop X.parm7` 查看编号，`--residues` 写成 `3-5` 这样不带链名的形式。"搭体系的选项"对这种输入不起作用，用了会给出提示。

## 结构检查（用 PDB 文件或 PDB 编号时）

搭体系之前，MDAgent 先检查结构。只有结构"干净"，才会进入下一步；发现问题时，运行停下（退出码 4），并说明怎么处理。单独检查、不搭体系：`python -m mdagent check --pdb-id 4OBE`。

检查的内容：

- **和参考序列比对：** PDB 文件里记录了每条链对应的 UniProt 条目和残基范围（DBREF），以及每一处已知差异和原因（SEQADV，例如表达标签、人为突变）。MDAgent 下载 UniProt 序列（包括所有异构体），存在 `structures/uniprot/<编号>.fasta`，逐位比对。文件里说明了原因的差异只记录；没有说明的差异算问题。如果文件里的编号对应明显有误，就按序列重新对齐。
- **标签：** 标为表达标签等、或者不在参考序列范围内的残基，算作标签。链两端有坐标的标签默认删掉（`--keep-tags` 保留）；夹在链中间的标签算问题。
- **多条链：** 同一个蛋白的多个拷贝算问题，用 `--chains A` 选一条，或用 `--chains A,B` 明确保留全部。
- **其他分子：** 水、离子、配体照常删掉。结构性金属（Zn、Fe、Cu 等）算问题，因为删掉可能破坏结构；需要保留时，请自己准备 Amber 体系后用 `--prmtop`。
- **完整性：** 未知残基、链中间连续缺 5 个以上残基、文件里没有说明的断链，都算问题。短缺口会被建模补上；链末端缺失、交替构象、多个模型只记录。
- **没有参考序列的文件**（例如自己做的模型）：算问题，可以用 `--uniprot <编号>` 指定参考序列。

确认问题无关紧要时，加 `--accept-problems` 继续。检查报告保存在运行目录的 `structure_check.txt`。

注意：`examples/181L.pdb` 和 UniProt P00720 有两处没有说明的差异（R12G、I137R），所以用它测试时要加 `--accept-problems`。

## 平衡

每条轨迹的第一段正式模拟之前，先做平衡，默认共 600 ps（`--equil-ps` 可改）：

1. 在蛋白骨架原子（N、CA、C）上加约束（1 kcal/mol/Å²），做能量最小化；
2. NVT 下从 5 K 升温到目标温度（默认 300 K），占 1/6 的时间；
3. NPT 下保持约束，占 2/6；
4. NPT 下分 20 步把约束减到零，占 2/6；
5. NPT 下无约束，占 1/6。

约束力只存在于平衡用的模拟里。正式模拟是一个新的模拟，只接收平衡结束时的坐标、速度和盒子，所以三者来自同一时刻，正式模拟里也没有残留的约束力。

## 搭体系的选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--forcefield` | `charmm36m` | `charmm36m`：CHARMM36m + CHARMM TIP3P，范德华截断 1.2 nm，从 1.0 nm 开始平滑减小，不加长程色散修正。`amber14`：ff14SB + TIP3P。`amber19`：ff19SB + OPC。每个选项都是一整套匹配好的力场、水模型和截断设置，不能拆开混用。 |
| `--ph` | 7 | 决定可滴定残基的默认质子化状态。 |
| `--check-protonation` | 关 | 打开后，先用 PROPKA 根据蛋白环境预测每个 Asp、Glu、Lys、His 的 pKa，再按预测结果设定质子化状态，然后才开始 MD。报告里会列出所有被改变的残基，并标出哪些在目标区域里。 |
| `--protonation` | 无 | 手动指定，例如 `A:31=HIP,A:70=ASH`。在 PROPKA 之后生效，所以会覆盖 PROPKA 的结果。链两端的残基不能改。 |
| `--ion-conc` | 0.15 | 盐浓度（M）。体系总是先被中和。 |
| `--cation` | `Na+` | `Na+` 或 `K+`。阴离子是 Cl⁻。 |
| `--padding-nm` | 1.0 | 蛋白到水盒子每个面的最小距离。盒子是周期性的，所以蛋白和相邻盒子里的镜像至少隔两倍这个距离。 |
| `--box-shape` | `cube` | `cube`、`dodecahedron` 或 `octahedron`。菱形十二面体盒子的水比立方体少约 30%，模拟更快。 |
| `--setup-only` | 关 | 只搭体系，然后停下，方便先检查报告。确认后用 `resume` 开始 MD。 |

**缺失残基：** 程序会逐段列出缺了哪些残基。链两端缺的不补，因为通常是无序区；链中间缺的由 PDBFixer 补上。补的这一段超过 5 个残基、或者离目标区域 4 个残基以内时，会给出警告。如果 PDB 文件没有序列记录（SEQRES），PDBFixer 发现不了缺失；这时程序会检查相邻残基的肽键长度，发现断链时给出警告。

搭体系的完整报告会显示在屏幕上，并保存为 `setup_report.txt`。所有选项也记录在 `run.json` 里。

**关于速度：** CHARMM 的部分范德华参数在 OpenMM 里要用较慢的方式计算，所以 CHARMM36m 比 Amber 慢一些。

## 用法

```bash
git clone https://github.com/Lyn-666/MDAgent1.git && cd MDAgent1
conda create -n mdagent -c conda-forge python=3.11 openmm pdbfixer mdtraj -y
conda activate mdagent
pip install -e ".[test]"

# 开始。agent 自己决定跑多久，每条轨迹最多 20 ns。
python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/181L

# 先用 PROPKA 检查质子化状态，改用 Amber ff19SB + OPC 和 0.1 M KCl，只搭体系不跑 MD。
python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/181L_b \
    --check-protonation --forcefield amber19 --ion-conc 0.1 --cation K+ --setup-only
python -m mdagent resume --run runs/181L_b   # 看过报告后，开始 MD。

# 接着跑：可以继续一个中断的运行，也可以给已结束的运行更大的预算。
python -m mdagent resume --run runs/181L --max-ns 40

# 查看最新的排序诊断。
python -m mdagent diagnose --run runs/181L

pytest -q tests                # 全部测试，约 1 分钟
pytest -q tests -m "not slow"  # 不跑 MD 的测试，约 1 秒
```

常用参数：`--replicas`（轨迹条数，默认 3）、`--first-ns`（第 1 轮长度，默认 1）、`--extend-ns`（以后每轮长度，默认 5）、`--max-ns`（每条轨迹的总预算，默认 20）、`--no-flip-patience-ns`（多久没翻转就改用增强采样，默认 10）。

速度参考：181L 体系约 3.5 万个原子。在一块消费级 GPU 上约每天 470 ns，按默认设置全部跑完约 3 小时。在 4 核 CPU 上约每天 3 ns，只适合测试。

## 运行目录里的文件

| 文件 | 内容 |
|---|---|
| `run.json` | 搭体系的选项、MD 设置、目标残基、每轮的长度。续跑时从这里读取。 |
| `setup_report.txt` | 搭体系的报告：力场、缺失残基、质子化、水盒子、离子和警告。 |
| `decisions.log` | 每轮的决定和理由，纯文本。 |
| `status.json` | 最新的诊断和决定，供其他程序读取。 |
| `replica_<i>_part<k>.dcd` | 第 i 条轨迹第 k 轮的轨迹，只含蛋白原子。 |
| `replica_<i>_part<k>.state.xml` | 第 k 轮结束时的状态。只有这一轮完整跑完后才会出现。 |
| `solvated.pdb`、`system.xml`、`protein.pdb` | 搭好的体系和分析用的蛋白结构。 |

如果某一轮中途停止，`resume` 会先补完这一轮。已经跑完的轨迹段不会重跑。

## 测试体系

`examples/181L.pdb` 是 T4 溶菌酶 L99A 与苯的复合物。口袋里的 Val111 侧链在常规 MD 中很难翻转，这个现象已有文献报道。

## 当前限制

- 用 `--pdb` 输入时，配体会被去掉，只模拟蛋白本身。要保留配体，先准备好 Amber 体系，再用 `--prmtop`/`--inpcrd` 输入。
- PROPKA 只在用户要求时运行，预测的是去掉配体后的蛋白。
- 只看 chi1，还没有看 chi2。

## 增强采样

普通 MD 停下后，agent 自动对被标为 `enhanced_sampling` 的侧链做增强采样。方法是 well-tempered metadynamics，直接在 OpenMM 里实现，不需要 PLUMED。每个侧链的 chi1 有一个独立的偏置势，所有侧链在同一次模拟里处理。每条增强采样轨迹从普通 MD 的最终状态开始，默认 2 条，每条 20 ns。用 `resume --enhanced-ns 30` 可以加长。

agent 只负责跑增强采样和保存数据，不做分析。输出在运行目录的 `enhanced/` 里：

| 文件 | 内容 |
|---|---|
| `replica_<i>_chunk<k>.dcd` | 第 i 条增强采样轨迹第 k 个 1 ns 片段的蛋白轨迹。拓扑用运行目录里的 `protein.pdb`。 |
| `replica_<i>_chi.npy` | 每个被偏置残基的 chi1（弧度），每 1 ps 一行。 |
| `replica_<i>_bias.npy` | 每个被偏置残基在 chi1 网格上的最终偏置势（kJ/mol），网格从 -180° 到 180°，361 个点。 |
| `replica_<i>.state.xml`、`replica_<i>.json` | 续跑用的状态和进度。 |

残基在 `.npy` 文件里的顺序，与 `run.json` 中 `enhanced.residues` 的顺序相同。

## 自适应采样（可选）

加上 `--sampling adaptive` 后，普通 MD 阶段从第 2 轮起，每条轨迹不再接着自己的上一段跑，而是从 AdaptivePy（Shukla 课题组的自适应采样包）挑出的帧重新起跑。默认策略 `least_counts` 会挑"访问最少"区域里的帧。需要先安装：`pip install adaptivepy-sampling`。

- **特征：** 目标区域里各侧链 chi1 的正弦和余弦。
- **起点要分散：** 聚类数按每 5 帧一类设定（最多 `--adaptive-clusters` 个）。AdaptivePy 先给出 3 倍于所需的候选帧，agent 再从中选出彼此差别最大的几个作为起点，避免几条轨迹从几乎相同的结构出发。
- **完整坐标：** 这个模式下每段轨迹每 10 ps 额外保存一帧完整体系（含水），文件名是 `replica_<i>_part<k>.full.dcd`。起点只从这些帧里挑。重新起跑时，按设定温度重新随机分配速度。
- **判断规则：** 各段轨迹的起点是故意挑得不同的，所以这个模式下 agent 只看所有轨迹段加起来的翻转次数，不再比较各条轨迹是否停在同一个状态。
- **记录：** 每一轮挑出的起点记在 `run.json` 的 `starts` 里；AdaptivePy 的输入和输出在 `adaptive/round_<k>/` 里。程序会核对 AdaptivePy 给出的帧编号，对不上就停下报错。

选项：`--adaptive-policy`（`least_counts` 或 `random`）、`--adaptive-clusters`（聚类数，默认 20）、`--full-save-ps`（完整坐标的保存间隔，默认 10 ps）。

## 聊天窗口（`chat`）

```bash
conda activate mdagent
pip install -e ".[chat]"
python -m mdagent chat              # 浏览器打开 http://localhost:8765
```

一个本地网页，界面有中文和 English 两种语言，右上角可以切换。

- **语言模型后台：** `--backend claude`（Claude Code，用你的 Claude 登录）、`--backend codex`（Codex，用你的 ChatGPT 登录）、`--backend api`（需要 `ANTHROPIC_API_KEY`）。默认 `auto`，选第一个已安装的。它们只负责理解你的话并提出建议，不能运行任何命令：Claude Code 的工具全部关闭，Codex 用只读模式。
- **先确认后执行：** 开始、停止、继续一个运行，都会先显示成一张卡片，上面有设置和完整命令。你点击按钮后才执行。
- **运行和网页无关：** 每个运行都是独立的进程，有自己的进程组，日志写在 `runs/<名字>.log`。关掉网页、甚至停掉聊天服务（Ctrl+C），模拟都会继续。只有在聊天窗口里说"停止"并点击确认，或者运行完成，它才会结束。
- **对话：** 左侧是对话列表，点"新对话"开始一个新案例。每个对话有自己的聊天记录和运行列表，保存在 `runs/.chats/` 里，重启聊天服务后仍在。不同的案例用不同的对话，就不会混在一起。
- **结构检查：** 提出运行方案后，先做结构检查。有问题时不显示运行按钮，而是把检查报告给你看，你可以说"只保留 A 链""保留标签"，或者明确说"有问题也继续"。检查通过时，卡片里可以展开查看报告。
- **PDB 编号：** 在对话里提到一个 PDB 编号（比如"用 3HTB，看 JZ4 周围的口袋"），agent 会先下载到 `structures/3HTB.pdb`，看过文件里的链和配体，再提出运行方案。
- **进度：** 右侧默认只列出本对话启动的运行，可以切换到"全部"，看这个项目里的所有运行（包括用命令行启动的）。列表每 10 秒刷新一次。也可以直接问"dev5 跑到哪了"。聊天窗口不分析模拟结果。
- **谁能看到：** 运行和对话都保存在这台电脑上这个项目文件夹的 `runs/` 里，不会上传到 GitHub。在同一个文件夹里打开聊天窗口的人看到的是同一份对话和运行；别人在自己电脑上克隆仓库，看不到你的。

如果设置了 `ANTHROPIC_API_KEY`，Claude Code 会优先用这个密钥，而不是你的订阅登录。Codex 后台还没有在真实环境里测试过。

## 用 Claude Code 当 agent 窗口（不需要 API 密钥）

在自己的电脑上，先激活环境，再在 MDAgent1 文件夹里打开 Claude Code：

```bash
conda activate mdagent
cd ~/Desktop/MDAgent1
claude
```

然后在 Claude Code 里输入：

- `/md 用 Amber，看苯周围的口袋，先检查质子化，开发测试`：Claude Code 读取 PDB 概况，把要求转成 `mdagent run` 命令，列出设置请你确认，确认后用 `nohup` 在后台启动。
- `/md-status`：查看进度。运行中途停了时，它会问你要不要续跑。

模拟在后台运行，关掉 Claude Code 或终端都不会停。Claude Code 不分析模拟结果。命令的内容在 `.claude/commands/` 里。

## 用一句话描述要求（`ask`，需要 API 密钥）

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...        # 或者用 ant auth login 登录
python -m mdagent ask --pdb examples/181L.pdb --out runs/a "用 CHARMM，看苯周围的口袋，0.1 M KCl，先检查质子化"
```

Claude（`claude-opus-5-5`）只负责填一张固定的设置表，表里每一项都对应 `mdagent run` 的一个选项。你没提到的项留空，保持默认值。程序会把 PDB 文件的概况告诉 Claude：有哪些链、残基编号范围和配体名称。所以 Claude 不需要猜配体名称；你说的配体在文件里找不到时，它会提问。

- 要求不清楚时，Claude 会提出问题，程序不运行。
- 要求清楚时，设置表被转成一条普通的 `mdagent run` 命令，经过和命令行完全相同的检查。程序显示目标区域、哪些设置改了默认值、完整的命令，等你输入 `y` 才开始运行。加 `--yes` 可以跳过确认。
- 如果安全分类器拒绝了请求，API 会自动换一个模型重试（server-side fallback）。仍被拒绝时，程序会报错，不会运行。

## 下一步

1. 在用户电脑上试用聊天窗口（`python -m mdagent chat`），分别试 Claude Code 和 Codex 后台。
