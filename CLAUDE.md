# CLAUDE.md

这个文件是本项目的记忆。Claude Code 每次在这个仓库里开始对话时都会读取它。

**更新规则：** 用户提出新的要求或偏好、做出设计决定、发现只能在用户电脑上验证的事项，或者项目状态有重要变化时，在同一次工作里更新本文件对应的栏目，和代码一起提交推送。删掉过时的内容，不要只往后追加。保持简短；细节写进 README。

## 与用户沟通的要求（每次都必须遵守）

开始前，用一句话说明你准备做什么；工作中简短更新进展，帮助用户跟上。结束时给出一段可以独立理解的简短回顾，说明发现了什么、做了什么，以及接下来如何，让只看到最后一条消息的读者也能了解全貌。

清楚沟通；面向聪明、但不了解当前上下文的读者写作。读者没有看过你的工具调用、推理过程或打开的文件，也可能刚从其他工作切换过来。消息通常应能独立理解：读者无需翻找其他上下文，就应明白你的意思。把关键点放在开头几句话里。通过句子顺序逐步展开信息，不要在第一行就使用要到第十二行才能看懂、且尚未定义的指代或概念。

用平实词语，写清楚、好懂的完整句子。不要使用工作过程中自己造出的术语或缩写，以任何读者都能理解的方式说明它是什么。写作总体上参考 ASD-STE100 简化技术英语的风格：使用短句，每句表达一个意思，采用主动语态，对同一件事始终使用同一个词。观点彼此相连时，用连贯自然的段落；只有列举独立项目时才使用项目符号。除非读者要求，否则省去不必要的细节，尽量简洁，但要确保保留关键背景。努力让面向用户的文字尽可能清晰易读。

用户使用中文交流，回复用中文。给用户的命令要能直接复制运行，并包含 `cd ~/Desktop/MDAgent1 && git pull`。

## 项目目标（用户最初的要求）

用户输入蛋白（PDB 编号、PDB 文件或准备好的 Amber 体系）和目标区域。agent 自动处理蛋白、搭水盒子、加离子、跑 MD；跑一段时间后自动判断目标区域里哪些侧链没采样到或各条轨迹太相似，再对这些侧链做增强采样。整个过程在一个 agent 窗口里完成。

**用户后来说明的目的（2026-11-17）：** 用户描述任务后，agent 按文献建议方法和设置（普通 MD、自适应采样、增强采样；alchemical 只推荐外部工具），但跑什么由用户决定。默认 agent 自己判断（`--mode auto`）：边跑边分析，自己决定何时换方法、何时停、何时交付。用户指定方法时（`--mode guided`）照做，没有进展就在对话里发消息建议换方法，不暂停、不自己换。核心：再加重复或加长对结果没有增益时，agent 停下。

**用户说明的核心目的（2026-11-20，修正）：** 目的是完成整个模拟、得到用户想观测的结果，但要边跑边分析。用户开跑前定好模拟时间和重复数（上限）；agent 看到用户指定的结果（例如某个结构变化）就提前停；如果再加长、加重复没有收益，也停下，并建议换方法（例如自适应采样）。自适应采样也按同样规则跑。**用户的决定（2026-11-20）：换方法只建议，用户确认后才做；不能超过用户定的时间和重复数，但可以发消息提醒。**已做：默认模式权限只有 stop_early；policy 第 6 条“上限内达不到”（乐观估计超过上限 1.5 倍就停，原因 not_reachable）；加长、加重复、换方法的建议都写明怎么确认（`resume --max-ns/--replicas/--enhanced-sampling`，对话里也能说）；误差下限。已做：目标类型“观察到某个变化”（`--observe "A:111=t"` 或 `"<距离>=far"`，从别的状态进入并停留 `--hold-ps`（默认 100 ps）才算，`--seen-in` 条重复看到就停；goal quantity `event_seen`）。已做：自适应采样的比例用马尔可夫状态模型算（`msm.py`：核心集状态、细致平衡的最大似然转变矩阵、对轨迹段 bootstrap 的误差）。

**通用化方向（2026-11-20，用户要求）：** MDAgent 要成为通用的科学 MD agent：理解任务目标、执行合适的模拟、评估证据、在预算和权限内做有依据的决定、少花不必要的计算。1UBQ（和 181L）只是开发和评估用例，不是设计目标；不要把任何体系、观测量、转变时间、重复数或模拟长度的假设写进通用决策。通用推理和具体的数值估计要分开；不能对不同的观测量用同一个收敛阈值；启发式论证不能当作统计结论。评估分开发、校准、保留测试三组体系，并和固定预算、常规逐轮停止规则比较。架构评估、目标架构、分阶段计划和评估计划在 `docs/architecture.md`。第 1 阶段（通用决策核心 `mdagent/core/`）和第 2 阶段（OpenMM 适配器、影子模式：新核心每轮记下自己的决定 `decisions.core.jsonl`，不执行；`mdagent shadow` 用已有文件重放已跑完的运行并对比）已完成，正式运行路径的决定还没改。下一步：用户在 GPU 电脑上对第 5 步的运行跑 `mdagent shadow`，逐条看分歧，再做第 3 阶段（在开关后面切换到新核心）。第 3 阶段要先解决：换方法后，只有旧方法能采样的估计量怎么收尾。**骨架 φ/ψ（2026-11-21，用户要求）：** 不为某个固定的蛋白，而是通用的：用户提出要求时，agent 给出方案（方法、开始的重复数、上限、停止条件）并在确认后执行。已做：`--quantity backbone`（任何残基的 Ramachandran basin 比例、basin 之间的转变统计、φ/ψ 自由能面，各有自己的精度要求，三样都达到才算达标）；basin 定义是数据，可用 `--basins` 换；对话里“你来决定重复数和长度，最多 N 条、每条最多 M ns”会转成：先开 3 条、`--max-replicas N --max-ns M --allow stop_early,add_replicas`，回复里写明方案。骨架的增强采样已做（2026-11-21，用户要求）：φ 和 ψ 上各一个一维 well-tempered metadynamics 偏置势（`enhanced.dihedral_quads(kind="backbone")`，列顺序 φ1、ψ1、φ2…），重新加权给 basin 比例和自由能面（`backbone/*_fes_reweighted.*`），停下要比例和自由能面都达到精度；转变速率和停留时间只来自普通 MD（偏置改变动力学）。已知答案的模型测试通过；OpenMM 里真跑 10 个二面角的慢速测试通过。2026-11-21 起：一直不变化的项目什么时候放弃，默认不再用固定 10 ns，而是按剩余预算判断（95% 上限速率 3/已跑时间，剩余预算也看不到足够的转变才放弃，和新核心同一规则；`--no-flip-patience-ns` 仍可指定；旧运行续跑时保留它们存下的 10 ns）；`--extend-ns` 默认 max(5, max_ns/20)；骨架 φ/ψ 的已完成轨迹段缓存在 `backbone/cache/`。还没在用户电脑上用真实语言模型试过这句话的转换（第一次在 masi-pc 上启动时忘了 `conda activate mdagent`，现在程序会提示）。用户已同意（2026-11-20）：1UBQ 和 181L 算校准体系（已经用来调过规则）；保留测试体系要另选（最好由别人选）并先封存名单。

**用户的开发计划（2026-11-17）：** 1 保存任务目标和权限（目标量、精度、预算；能否延长、加重复、换方法、提前停）；2 区分停止原因（达到目标、采样停滞、预算用完、运行失败；比例不再变化只是诊断，不是停止理由）；3 让分析可信（考虑时间相关性的不确定度、检查权重退化、验证多个耦合 chi1 的重新加权）；4 每轮后选择下一步动作（延长、加重复、重新播种、建议换方法、停止），执行前查权限；5 针对性验证（正常达标、所有重复同时困住、预算用完未达标）；6 完善交付，再扩展方法。**1–3 已完成（2026-11-17）。2026-11-18 用户同意先把"结果量"做成可替换的一层（`observables.py`），已完成：侧链构象和原子间距离两种量；增强采样只支持侧链构象。第 4 步已完成（`policy.py`），第 6 步的报告已完成（`report.py`），第 5 步的验证脚本已写好（`scripts/step5_validation.sh`），等用户在 GPU 上跑。** 5 里"三个场景"的逻辑测试已有（`tests/test_stop_reasons.py`），还缺用真实模拟检查停下时的误差。

## 用户定下的规则和决定

- **要做结果分析（2026-11-17 改）：** 以前的规则"不做增强采样之后的分析"已取消。增强采样用 Tiwary–Parrinello 重新加权算各状态比例（`reweight.py`），结果和停下的理由要交付给用户。语言模型自己不算结果，只转述运行发来的消息。
- **只有序列和参考序列一致、结构完好干净，才开始正式的下一步。** 结构检查有问题就停，除非用户明确说继续（`--accept-problems`）。181L 和 UniProt P00720 有两处未说明差异（R12G、I137R），用 181L 测试要加 `--accept-problems`。
- **每项设置都有默认值。** 影响科学结论的选择（力场、离子、pH、质子化、末端）由用户决定；用户提出非默认要求时，agent 先执行（例如先跑 PROPKA），再开始 MD。所有搭体系选项都在 `SetupOptions` 里，聊天窗口也填同一张表。
- **默认值：** 温度 300 K（用户不要改）；力场 CHARMM36m；0.15 M NaCl；pH 7；立方盒子 1.0 nm；平衡 600 ps（按用户自己的 OpenMM 流程：骨架约束、最小化、5 K 升温、约束 NPT、逐步撤约束、无约束 NPT）；末端 `--termini auto`（真实蛋白在这一端还没结束时加 ACE/NME 中性帽，按 UniProt 全长判断；否则保留带电末端）。
- **启动的运行要在后台一直跑**，关网页、关聊天服务都不停；只有用户确认停止或运行完成才结束。
- **界面文字要有中文和英文两种语言。**
- **路径要让用户一看就懂**，例如 `structures/3HTB.pdb`、`runs/<名字>/`。
- **对话里可以指定保存位置（2026-11-18）：** `ChatAction.save_dir`；运行实际存在 `<save_dir>/<运行名>/`，`runs/<运行名>` 是指向它的符号链接（确认时才建），对话的进度、消息、下载照常。已有文件的文件夹会被拒绝。
- **告诉用户结果文件在哪里时，写电脑名和完整路径**，并说明文件只在跑模拟的那台电脑上。网页的运行卡片上有下载链接（单个文件或 zip）。起因：用户跑完测试后在自己电脑上找不到轨迹。
- **测试者在自己的电脑上运行**，用自己的 Claude 或 Codex 登录。所以聊天网页不需要访问密码（只接受本机连接就够了）。**主要面向 Linux + NVIDIA GPU。** macOS 能运行但没有 CUDA，只适合短测试，不专门适配；Windows 只能通过 WSL2（程序用到 `os.killpg`、`pgrep` 和 Unix 的 C 库）。
- **语言模型只做"翻译"**：把用户的话填进固定的设置表，不能运行命令；开始、停止、继续都要用户确认（网页点按钮，终端输入 y）。
- **运行发消息到对话：** 运行写 `runs/<名字>/messages.jsonl`（中英文），网页和终端把新消息显示在启动它的对话里。建议换方法时只发消息，不暂停运行，不用按钮。
- **使用方式（用户的设计）：** 装好程序后运行 `mdagent start`，依次选界面语言、Claude 还是 ChatGPT（Codex）、网页还是终端。流程由我们的程序控制，不由语言模型控制；网页和终端共用 `chat.Assistant`。**尽量少用 Claude Code 和 Codex 的 token**：每句话只调用一次模型，提示里不放长列表，`/runs` 等本地命令不调用模型。Claude Code 的 `/md` 是 Claude Code 自己按说明工作，费额度，只给开发者用。
- **公开计划：** 先保持私有，邀请少数同学测试；公开前要完成：科学修复（已完成：跳过脯氨酸和二硫键半胱氨酸、增强采样最多 10 个侧链）、测试者材料（已完成：`environment.yml`、README 快速上手、Issue 模板、GitHub Actions 自动测试）、许可证（已完成：MIT，版权人 Lyn-666）、用户在 GPU 上的两项验证（见 README“下一步”；181L 完整运行的普通 MD 部分已通过：VAL111 在 3×20 ns 里没有翻转，被送进增强采样，10 个侧链的上限生效；增强采样里 VAL111 第 1 条轨迹 5 ns 翻转 373 次，三种状态都去过，也已通过）、删掉 CLAUDE.md 里只和用户个人电脑有关的记录。
- **开发阶段测试运行要很短（几分钟）**：`--replicas 2 --first-ns 0.1 --extend-ns 0.1 --max-ns 0.3 --no-flip-patience-ns 0.25 --enhanced-ns 0.1 --equil-ps 60`。
- **长上限也可以（2026-11-21，用户说明）：** 用户可以把上限定得很长（例如每条 1 µs），agent 在上限内自己决定重复数、长度和何时停；用户允许时也可以换方法。

## 代码结构

代码在 `mdagent/`，完整流程写在 README 的“完整流程”。

- `fetch.py`：按 PDB 编号从 RCSB 下载到 `structures/<ID>.pdb`。
- `check.py`：搭体系前的结构检查。拿不到参考序列（下载失败或 UniProt 条目已失效）时不判断标签、不删任何残基；失效条目会列出替代编号；`--uniprot` 可以替换文件里的编号。 按 DBREF/SEQADV 和 UniProt（含异构体，缓存在 `structures/uniprot/`）逐位比对；找标签、多拷贝、金属、末端帽、未知残基、长缺口、断链；判断每条链两端是否要加帽。
- `build.py`：搭体系（PDBFixer）。选链、去标签、加帽（CHARMM 下帽并入末端残基作为补丁，Amber 下是单独残基）、质子化、加水和离子。`load_amber` 原样读入用户准备好的 Amber 体系（配体保留）。
- `protonation.py`：PROPKA 和手动质子化。
- `simulate.py`：分段跑 MD，可续跑；`equilibrate` 是分阶段平衡。
- `diagnose.py`：侧链 chi1 诊断和排序。脯氨酸和形成二硫键的半胱氨酸（按 SG-SG 距离判断）判为 `fixed_chi1`，跳过。`agent.py`：多轮循环，决定续跑或停止；`run_all` 在普通 MD 后自动做增强采样。
- `enhanced.py`：二面角上的 well-tempered metadynamics（chi1；骨架 φ/ψ 时每个残基两个），在 OpenMM 里实现。一次最多偏置 `max_residues`（默认 10，`--enhanced-max-residues`）个最慢的侧链。各重复轮流跑，每轮 1 ns。
- `reweight.py`：重放高斯峰重建偏置势，Tiwary–Parrinello 重新加权，算状态比例和误差；`assess` 给每个侧链判断 reached / stalled / unreliable / not_yet；`populations.json`。用已知比例的模型和两个耦合侧链的模型测试过。
- `uncertainty.py`：分块平均的误差（考虑时间相关）、合并重复、有效样本数、误差下限（普通 MD 按转变次数，增强采样按有效样本数）。普通 MD 和增强采样都用。
- 目标和权限在 `agent.Goal`（`run.json` 的 `goal`）；停下的原因 target_reached / stalled / budget_exhausted / failed 记在 `run.json`（`notes.plain_md_outcome`、`enhanced.outcome`、失败时 `notes.outcome`）。
- `messages.py`：运行发给对话的消息。
- `policy.py`：第 4 步，每轮后选下一步动作（停、加长、加重复并补跑、超出预算延长、建议重新播种、上限内达不到就提前停），先查权限；没有权限就在对话里建议，并写明怎么确认。记录在 `run.json` 的 `notes.actions`。增强采样的同类规则写在 `agent.run_all` 的循环里。
- `report.py`：第 6 步，运行结束时写 `report.zh.md`、`report.en.md`（结论、误差、停下原因、动作、限制、文件、遗留问题）；`compare` 判断两个运行是否在误差内一致（|z| ≤ 2）。命令 `report`、`compare`。
- `scripts/step5_validation.sh`：第 5 步在 GPU 上跑的场景（达标、困住、预算用完、增强采样，加参考运行），汇总在 `runs/val_<蛋白>_summary.txt`。默认用 1UBQ（用户要求换掉 181L 作为主例子；`--uniprot P0CG48 --termini charged`），也可用 181L。
- `observables.py`：结果量层。每个量换成逐帧的状态，再用同一套诊断、误差和停下判断。还有“看到某个变化”的目标（`EventSpec`、`first_entry`、`diagnose_events`）。`chi1`（g+/t/g-）和 `distance`（近/远，`A:88:OH-A:111:CG1@0.6`，阈值 ±0.05 nm 的缓冲）。`--quantity`、`--distances`；对话里 `target_kind` 为 `distance`。
- `adaptive.py`：可选的 AdaptivePy 自适应采样（`--sampling adaptive`）。`msm.py`：自适应采样的比例（马尔可夫状态模型）。
- `envcheck.py`：启动时检查平台和力场文件；用选中的平台先算一个两原子体系，算不了就停下并说明修法（例如 CUDA 组件比驱动新时的 PTX 错误，改装匹配的 `cuda-version`）。
- `backends.py`：语言模型后台（`claude -p`、`codex exec`、Anthropic API），统一返回结构化结果。claude 后台会去掉环境里的 `ANTHROPIC_API_KEY`，以使用 Claude Code 的订阅登录。
- `chat.py`、`chat_page.html`：`chat` 命令，本地网页（`http://localhost:8765`），多个对话，存在 `runs/.chats/`。运行卡片显示电脑名、完整路径、文件列表和下载链接（`/api/file`、`/api/download`，只提供 `outputs.py` 列出的文件）。
- `outputs.py`：列出一个运行的结果文件（`protein.pdb`、轨迹、chi1 和偏置势数据），以及电脑名和完整路径。命令行的 Summary 和网页都用它。
- `terminal.py`：终端对话（`chat --terminal`）和 `start` 菜单（选择记在 `runs/.chats/start.json`）。
- `llm.py`：`ask` 命令和设置表 `RunPlan`。`ask` 也用 `backends.py`，不再只能用 API 密钥。
- `cli.py`：命令行 `start`、`run`、`resume`、`diagnose`、`check`、`info`、`fetch`、`ask`、`chat`。
- `core/`：通用决策核心，只用 Python 标准库，不提任何观测量、引擎或体系（`tests/test_core_architecture.py` 检查）。`task.py`（任务、每个估计量自己的要求、预算、权限）、`evidence.py`（估计结果、有效性检查、共用信号）、`registry.py`（估计器按“种类 × 数据类型”查找；工具接口）、`reasoner.py`（相互竞争的假设，都标为启发式）、`controller.py`（候选动作、价值、成本、权限和预算检查、选择）、`ledger.py`（`decisions.jsonl`）、`loop.py`（循环、执行前后的检查、续跑）。
- `backbone.py`：任何残基的骨架 φ/ψ：basin 定义（默认 alpha_R、beta 含 PPII、alpha_L、other；核心区边缘 10°）、逐帧 basin、转变统计（速率 = 次数 / 停留时间，泊松误差；平均停留时间 = 总时间 / 离开次数）、自由能面（15° 一格，误差来自重复之间的差别加误差下限），结果写到运行目录的 `backbone/`。
- `tools_openmm.py`：给新核心用的 OpenMM 工具（`PlainMDTool` 包装 `run_segment`，`MetadTool` 包装 `run_replica` 并重建偏置势给出权重）和特征化（chi1、距离、事件；`event_series` 和 `agent._diagnose_events` 共用）。工具可以只看过去某一时刻的数据（重放用）。`shadow.py`：影子模式（每轮记下新核心的决定，出错不影响运行，`MDAGENT_SHADOW=0` 关闭）、重放、和原策略对比；命令 `shadow`。
- `estimators.py`：具体的估计器（无偏比例、重新加权比例、马尔可夫模型比例、首次进入事件），数值来自 `uncertainty.py`、`msm.py`、`observables.py`。`synthetic.py`：已知答案的合成马尔可夫链工具，给开发和测试用。
- `.claude/commands/`：Claude Code 项目命令 `/md` 和 `/md-status`。

测试：`pytest -q tests`（约 4 分钟，含在 CPU 上跑小肽段的测试）；`pytest -q tests -m "not slow"` 只跑快速测试（208 个，约 2.5 分钟）。骨架增强采样的测试在 `tests/test_backbone_enhanced.py`。骨架 φ/ψ 的测试在 `tests/test_backbone.py`（速率和停留时间用 30 个随机种子检查误差是否校准）。通用核心的测试在 `tests/test_core_*.py` 和 `tests/test_estimators.py`（约 7 秒，全在合成体系上）；适配器和影子模式在 `tests/test_shadow.py`（用设定好 chi1 的假运行，不跑 MD）。测试离线运行，UniProt 序列在 `tests/data/uniprot/`。测试体系 `examples/181L.pdb`（T4 溶菌酶 L99A 与苯，口袋里的 Val111 侧链在常规 MD 中很难翻转）。

## 环境

- **用户的电脑：** 项目在 `~/Desktop/MDAgent1`，conda 环境 `mdagent`，OpenMM 8.6.1 带 CUDA，GPU 上约每天 300 到 600 ns。用户有 Claude 订阅，但没有 API 密钥。
- **不要在用户的 conda 环境里装 openmm-plumed：** conda-forge 上的版本会把 OpenMM 降到 8.2 并丢掉 CUDA，而且其中的 PLUMED 不带 OPES。
- **云端会话**（Claude Code 网页版的临时容器）：没有 GPU，只适合很短的流程测试；装不了 AdaptivePy；没有 API 密钥。新容器里没有依赖：`pip install openmm mdtraj "pydantic>=2" propka pytest legacy-cgi`；pdbfixer 用 pip 装会失败，要 `git clone https://github.com/openmm/pdbfixer` 后把其中的 `pdbfixer/` 文件夹复制到 site-packages。改动直接推送到 GitHub `Lyn-666/MDAgent1` 的 `main` 分支。最早的 `zcorn2017/binding-lang` 仓库已恢复原样，不再使用。
- **第一位测试者（2026-11-19，masi-pc，项目在 `~/MDAgent1`，不是 `~/Desktop/MDAgent1`）：** `~/.condarc` 里有 `solver: classic`，conda 卡在 “Collecting package metadata”，加 `--solver=libmamba` 解决；直接用 `pip` 会跑到系统旧 pip，要用 `python -m pip`；CUDA 报 PTX 版本错误（conda 装了 13.5，驱动 610.57 只支持 13.3），装 `cuda-version=13.3` 后 CUDA 可用（1UBQ 约 854 ns/day）；RTX 4090 + Intel CPU，CPU 8 和 9（物理核 16）不稳定（系统日志里 44 次段错误都在这两个 CPU 上），快速测试在 `addSolvent` 里段错误，要用 `taskset -c 0-7,10-31` 运行测试和 `mdagent start`（子进程继承限制）。日志查法和 `scripts/find_bad_cores.py` 见 README。11-19 在这台电脑上开始跑第 5 步（1UBQ）：reach_chi1 已完成且判断正确：ASP32 普通 MD 达标；LYS63、GLU16 各有一条重复几乎不翻转（每条至少 4 次的规则），普通 MD 用到 20 ns 上限后改用增强采样并达标；LYS63 约 89% 在 g+（+60°），和普通 MD 5 条重复的主要状态一致，说明重新加权正确。LYS63 翻转慢，不适合当“普通 MD 就能达标”的例子。11-20 第 5 步 10 个场景全部跑完（约 20 小时），停下原因都和实际一致（trapped_dist 判为达标是对的：Lys27–Asp52 距离在三条重复里来回 5、4、8 次，主要为“近”，误差 0.012；原以为它不会断开）；发现三个问题：`diagnose` 命令对距离运行误按 chi1 诊断（已修：和运行共用 `agent.diagnose_run`）；从没出现过的状态分块误差为 0，`compare` 因此误报“不一致”（LYS63 g- z=3.5，VAL26 z=inf；已加误差下限，`compare` 差别不到 0.01 也算一致）；`budget_chi1` 普通 MD 预算 2 ns，换增强采样后又用了默认 20 ns（和“用户定的是上限”不符，见上面“核心目的”）。
- **用户怎样打开网页（还没确认）：** 截图里浏览器是 Mac 上的 Chrome，地址是 `localhost:8765`，但运行的进程号是 1749823，macOS 上不会出现这么大的进程号。所以模拟很可能跑在一台 Linux 电脑上，网页经 SSH 端口转发打开，结果文件在那台 Linux 电脑上。
- **只能在用户电脑上验证的事项：** 网页下载链接经端口转发是否可用（还没验证）；AdaptivePy 的真实调用（已验证可用）；Claude Code 后台（已验证可用）；Codex 后台（还没验证；测试者会用到，公开前要验证）；`start` 加终端模式用 Claude 跑完开发测试（已验证可用）；181L 完整运行在 11-16 00:23 左右、增强采样第 1 条轨迹 5.4 ns 处停了，日志没有报错。已排除重启（11-17 08:47，非正常关机）和 SSH 断开（在 tmux 里跑，KillUserProcesses=no）；系统日志显示整台电脑在 11-16 00:22:54 之后就没有任何记录，直到 11-17 08:47 重启，所以是整机死机（不是程序或内存的问题），原因可能是 GPU、驱动、电源或散热。现在 run/resume 会在日志里记下进程号、开始时间、崩溃或 SIGTERM 时的调用栈，进度行带时间；CHARMM 加帽体系在 GPU 上的长时间运行（还没验证）；第 5 步第一次跑（1UBQ reach_chi1，11-19）：加重复的规则太激进，重复从 3 条加到 10 条（已改：只在重复明显不一致、且预算用了不到一半时加）；后来速度从 646 掉到 4 ns/day，同时一个 python 进程占了 23.4 GB 显存、GPU 使用率 0%，说明 OpenMM 悄悄改用了 CPU（已改：总是明确指定平台，GPU 不能用就报错）；占显存的是不是 MDAgent 自己的进程，还在确认。
