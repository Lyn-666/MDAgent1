# CLAUDE.md

这个文件是本项目的记忆。Claude Code 每次在这个仓库里开始对话时都会读取它。

**更新规则：** 用户提出新的要求或偏好、做出设计决定、发现只能在用户电脑上验证的事项，或者项目状态有重要变化时，在同一次工作里更新本文件对应的栏目，和代码一起提交推送。删掉过时的内容，不要只往后追加。保持简短；细节写进 README。

## 与用户沟通的要求（每次都必须遵守）

开始前，用一句话说明你准备做什么；工作中简短更新进展，帮助用户跟上。结束时给出一段可以独立理解的简短回顾，说明发现了什么、做了什么，以及接下来如何，让只看到最后一条消息的读者也能了解全貌。

清楚沟通；面向聪明、但不了解当前上下文的读者写作。读者没有看过你的工具调用、推理过程或打开的文件，也可能刚从其他工作切换过来。消息通常应能独立理解：读者无需翻找其他上下文，就应明白你的意思。把关键点放在开头几句话里。通过句子顺序逐步展开信息，不要在第一行就使用要到第十二行才能看懂、且尚未定义的指代或概念。

用平实词语，写清楚、好懂的完整句子。不要使用工作过程中自己造出的术语或缩写，以任何读者都能理解的方式说明它是什么。写作总体上参考 ASD-STE100 简化技术英语的风格：使用短句，每句表达一个意思，采用主动语态，对同一件事始终使用同一个词。观点彼此相连时，用连贯自然的段落；只有列举独立项目时才使用项目符号。除非读者要求，否则省去不必要的细节，尽量简洁，但要确保保留关键背景。努力让面向用户的文字尽可能清晰易读。

用户使用中文交流，回复用中文。给用户的命令要能直接复制运行，并包含 `cd ~/Desktop/MDAgent1 && git pull`。**用户在 masi-pc 上运行时（提示符是 `diego@masi-pc`），项目在 `~/MDAgent1`，命令用 `cd ~/MDAgent1 && git pull`，跑模拟和测试要加 `taskset -c 0-7,10-31`（CPU 8、9 不稳定）。**

**测试要和用户使用时一样（用户的要求，2026-11-20）：** 让用户验证功能时，不给脚本和文件路径，而是让用户打开 agent（`mdagent start`），在对话里用一句话把任务交给它，由 agent 自己下载结构、搭体系、决定怎么跑、交付报告。离线基准（`experiment-a`、`scripts/experiment_A.sh`）是开发者用来评估 agent 的工具，不是给用户的测试方法。

## 项目目标（用户最初的要求）

用户输入蛋白（PDB 编号、PDB 文件或准备好的 Amber 体系）和目标区域。agent 自动处理蛋白、搭水盒子、加离子、跑 MD；跑一段时间后自动判断目标区域里哪些侧链没采样到或各条轨迹太相似，再对这些侧链做增强采样。整个过程在一个 agent 窗口里完成。

**用户后来说明的目的（2026-11-17）：** 用户描述任务后，agent 按文献建议方法和设置（普通 MD、自适应采样、增强采样；alchemical 只推荐外部工具），但跑什么由用户决定。默认 agent 自己判断（`--mode auto`）：边跑边分析，自己决定何时换方法、何时停、何时交付。用户指定方法时（`--mode guided`）照做，没有进展就在对话里发消息建议换方法，不暂停、不自己换。核心：再加重复或加长对结果没有增益时，agent 停下。

**用户说明的核心目的（2026-11-20，修正）：** 目的是完成整个模拟、得到用户想观测的结果，但要边跑边分析。用户开跑前定好模拟时间和重复数（上限）；agent 看到用户指定的结果（例如某个结构变化）就提前停；如果再加长、加重复没有收益，也停下，并建议换方法（例如自适应采样）。自适应采样也按同样规则跑。**用户的决定（2026-11-20）：换方法只建议，用户确认后才做；不能超过用户定的时间和重复数，但可以发消息提醒。**已做：默认模式权限只有 stop_early；policy 第 6 条“上限内达不到”（乐观估计超过上限 1.5 倍就停，原因 not_reachable）；加长、加重复、换方法的建议都写明怎么确认（`resume --max-ns/--replicas/--enhanced-sampling`，对话里也能说）；误差下限。已做：目标类型“观察到某个变化”（`--observe "A:111=t"` 或 `"<距离>=far"`，从别的状态进入并停留 `--hold-ps`（默认 100 ps）才算，`--seen-in` 条重复看到就停；goal quantity `event_seen`）。已做：自适应采样的比例用马尔可夫状态模型算（`msm.py`：核心集状态、细致平衡的最大似然转变矩阵、对轨迹段 bootstrap 的误差）。

**用户重新定位的研究目标（2026-11-20，最重要）：** 项目不是"达到采样目标就停"的工作流，而是**在给定精度下用最少计算量得到和高预算参考一致的结果**（accuracy-constrained simulation budget optimization）。要回答的问题：给定体系、要测的量和预算，agent 能不能自己分配模拟时间、重复数和采样策略，用最小成本得到统计上可靠的结果。应用在两类任务上：(1) 普通 MD——跑多少 ns、几条重复、什么时候可以提前停；(2) 自适应采样——何时开新重复、从哪个构象重新起跑、每条跑多久、剩下的预算怎么分。注意自适应采样不等于增强采样；先做无偏 MD 的分配和重新起跑，更容易验证平衡量是否正确。

用户当时明确说：**先不要再加新的采样方法。** 先把现有的决策框架改成"精度约束下的预算优化"，再把同一套决策接口扩展到自适应采样。用户的三阶段路线：1 普通 MD 的提前停和重复分配；2 自适应采样的智能重新起跑（和 uniform restart、least-counts 比）；3 在难度不同的体系上做基准测试（独立种子、冻结策略，报告误差、失败率、GPU 小时、累计 ns、重复数、达到精度的时间）。

用户强调的几点：成本要以 total compute cost 为主要目标，replica 数、累计 ns、wall-clock 分开报告，不要把"最少时间"和"最少重复数"当成两个要同时最小化的指标（10×10 ns 和 1×100 ns 累计相同，但初始化开销和对慢跃迁的探索完全不同；GPU 够多时多开重复反而等得更短）；不同的量（rotamer 比例、自由能差、跃迁速率、MFPT）不能用同一套收敛判据；在线停止不能用高预算参考来指导，参考只能用于离线验证；`|z| ≤ 2` 不足以证明一致，要用等效性检验和校准过的序贯置信区间。

**用户的开发计划（2026-11-17，在上面这个新目标之前写的）：** 1 保存任务目标和权限（目标量、精度、预算；能否延长、加重复、换方法、提前停）；2 区分停止原因（达到目标、采样停滞、预算用完、运行失败；比例不再变化只是诊断，不是停止理由）；3 让分析可信（考虑时间相关性的不确定度、检查权重退化、验证多个耦合 chi1 的重新加权）；4 每轮后选择下一步动作（延长、加重复、重新播种、建议换方法、停止），执行前查权限；5 针对性验证（正常达标、所有重复同时困住、预算用完未达标）；6 完善交付，再扩展方法。**1–3 已完成（2026-11-17）。2026-11-18 用户同意先把"结果量"做成可替换的一层（`observables.py`），已完成：侧链构象和原子间距离两种量；增强采样只支持侧链构象。第 4 步已完成（`policy.py`），第 6 步的报告已完成（`report.py`），第 5 步的验证脚本已写好（`scripts/step5_validation.sh`），等用户在 GPU 上跑。** 5 里"三个场景"的逻辑测试已有（`tests/test_stop_reasons.py`），还缺用真实模拟检查停下时的误差。

**按新目标做了什么（2026-11-20）：** 加了三件事，都在 GPU 上没验证过。(1) 成本核算 `cost.py`：aggregate_ns（所有重复相加的生产时间）、compute_ns（加上每条重复各自的平衡时间，主要指标）、wall_ns（按同时能跑几条折算的等待时间）分开报告；运行每轮记下实测秒数（`run.json` 的 `notes.rounds`）；结果报告里写明用了多少计算量。(2) 精度判据 `uncertainty.Precision`：`--criterion standard_error`（默认，含义和以前一样）/`confidence`（α 平摊到预先知道的判断次数）/`sequence`（Robbins 置信序列）。默认没改，因为改成 `confidence` 会让运行长好几倍；先用 `budget` 量出代价再决定。(3) 策略对照 `budget.py` + `budget` 命令：一次高预算数据池，离线回放 fixed / early_stop / add_replicas / mdagent 四种策略，比成本和正确率；对"只加长、只加独立重复"的策略回放是精确的，"从挑出的帧重新起跑"回放不了。`compare` 从 `|z| ≤ 2` 换成等效性检验（TOST）。在已知比例的模型上看到（容差 0.05，10 次重复）：`early_stop` 省 83% 但 10 次里 2 次超出容差（最大偏差 0.097），`mdagent` 省 74%、9 次在容差内（最大偏差 0.057，和固定方案一样）；把判据换成 `confidence` 后各策略会跑更久，但在每条 20 ns 的预算下达不到这个判据，只是以"预算用完"停下，答案没变好——更严的判据要配更大的预算。

**实验 A（2026-11-20，按用户的设计重做）：** 问题：给定 MD 任务和事先定好的精度，agent 能否自己决定几条重复、每条多长、何时停，以最少计算量得到统计上可靠的结果（task-conditioned, uncertainty-aware budget optimization，不是收敛检测）。用户的要求：**在线停止判据只用自己的轨迹，离线成功判据才用独立参考，防止参考泄漏**；参考用同一力场流程的内部数据池（留出的重复），不用文献数值；测主链构象区（比例为主，另报自由能差、速率、角度图谱，各自单独判断，不能一并宣布收敛）；四个对照：固定方案（1 条和 3/4/5/8 条，按长度扫描）、Sequential-CI（最重要：agent 必须比它省）、Agent；报告成功率、过早停止率、区间覆盖率、C_stop 与 C_hit、GPU 小时、等待时间、成本–准确度曲线；预算不够时输出 INSUFFICIENT_EVIDENCE 也是正确行为；固定方案的成本取它自己达到同样可靠性所需的成本。A-Easy：预实验挑出的快速 φ/ψ 残基；A-Hard：50 环 D52/G53 β-I/β-II 转角。文献（PubMed 核对）：这个翻转约 8 µs 来回 7 次、NMR 约 13–15 µs、β-II 约 4%（Arasteh 2021 doi:10.1021/acs.jpclett.1c00778；Smith 2016 doi:10.1073/pnas.1519609113），所以 16×100 ns 的数据池建不起 A-Hard 的参考，A-Hard 只检验"会不会诚实地说证据不足"。代码：`basins.py`（rama:A:73、turn:A:52，核心半径 50°、过渡式分配）、`experiment_a.py`（全部逻辑，命令 `experiment-a`，`--model` 不需要 GPU，`--select` 用于预实验挑残基）、`scripts/experiment_A.sh`（pilot/select/pool/analyze）。agent 规则：一次只规划的查看（不停、不消耗 α），之后最多 3 次可停的查看（Bonferroni 按 3 次）；状态变化太少、根本不可能停的查看不算进这 3 次（2026-11-20 改，起因：1UBQ 运行的第 2 次查看只是因为变化太少而加长，却占了名额，会让运行在每条约 16 ns 时以"查看次数用完"停下）；`allocate()` 在加长和加重复之间选成本最低的，只有重复间差别显著（卡方检验）才加重复；最后一次查看前余量更大；用掉四分之一预算后，投影超过预算 3 倍才放弃。普通 MD 回答不了的单个项目（耐心时间后一次没变化，或它自己的投影远超预算）**单独**标为证据不足、搁置，其余项目照常跑到精度；只有全部都回答不了时整个运行才说证据不足（2026-11-20，起因：1UBQ 任务里 52/53 转角约 10 µs 才翻转，会连累残基 8–10）。规则只在模型上调过，现已冻结。模型干跑：快模型上 agent 100% 成功、0 过早停止、比 Sequential-CI 省约 32%，但比事后读出的最短可靠固定方案贵（原理上躲不开，约为区间系数平方比）。旧的 `budget.py`/`hypothesis_a` 保留作通用策略对照，不再是实验 A 的主体。默认容差 0.05、可靠性 90%，由用户在看结果前确认。**还没在 GPU 上跑。**

**用户的要求（2026-11-20）：agent 不能是专门为实验 A 写的。** 用户在对话里打出"使用 1UBQ，通过 standard unbiased MD 估计指定残基的 Ramachandran basin populations、transition statistics 和 φ/ψ free-energy landscape，你可以自行决定 replicas 数量、每条模拟长度和停止时间，在保证准确度的前提下最小化总 simulation cost"时，agent 自己就要完成实验并交付预期结果；换成别的实验也一样。已做：决策规则在 `allocation.py`（`next_step`），日常运行（`--allocation projection`，`policy.allocation_next`）和实验 A 回放（`experiment_a.run_agent`）调用同一份代码；主链构象区是通用测量量（`--quantity backbone --backbone "rama:A:8-10;turn:A:52"`，`observables.diagnose_backbone`，比例/自由能差/速率/图谱的计算在 `basins.summarize`，两边共用）；对话设置表有 `target_kind` backbone、`backbone`、`allocation`、`budget_ns`、`max_replicas`；交付报告有"每个量单独判断""agent 的决策"和成本章节，图谱在 `landscape.py`（`landscape_*.svg/.csv`）。用真实的 `claude -p` 测过用户这句话：没给残基编号时它先追问；给了编号就填出 backbone + projection + 允许 extend,add_replicas,stop_early（不换增强采样）。默认分配仍是 `rounds`，没改旧行为；要不要把 projection 设为自动模式的默认，等实验 A 在 GPU 上验证后由用户决定。

## 用户定下的规则和决定

- **"更省"要能验证（2026-11-20）：** 一个策略只有既更便宜、答案又仍然在容差以内，才算更好。所以：成本三个数分开报；参考必须独立（策略没用到的那些重复，或者已知真值），不能和整个数据池比，那等于和自己比；一致性用等效性检验，不用 `|z| ≤ 2`（数据越少越容易通过，短跑会正因为不精确而显得正确）；一次跑赢不算，要多次重复（`--repeats`）看正确率。同一个数据池的多次重复不是完全独立的（换的是取哪几条重复），报告里说明了这一点。
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

## 代码结构

代码在 `mdagent/`，完整流程写在 README 的“完整流程”。

- `fetch.py`：按 PDB 编号从 RCSB 下载到 `structures/<ID>.pdb`。
- `check.py`：搭体系前的结构检查。拿不到参考序列（下载失败或 UniProt 条目已失效）时不判断标签、不删任何残基；失效条目会列出替代编号；`--uniprot` 可以替换文件里的编号。 按 DBREF/SEQADV 和 UniProt（含异构体，缓存在 `structures/uniprot/`）逐位比对；找标签、多拷贝、金属、末端帽、未知残基、长缺口、断链；判断每条链两端是否要加帽。
- `build.py`：搭体系（PDBFixer）。选链、去标签、加帽（CHARMM 下帽并入末端残基作为补丁，Amber 下是单独残基）、质子化、加水和离子。`load_amber` 原样读入用户准备好的 Amber 体系（配体保留）。
- `protonation.py`：PROPKA 和手动质子化。
- `simulate.py`：分段跑 MD，可续跑；`equilibrate` 是分阶段平衡。
- `diagnose.py`：侧链 chi1 诊断和排序。脯氨酸和形成二硫键的半胱氨酸（按 SG-SG 距离判断）判为 `fixed_chi1`，跳过。`agent.py`：多轮循环，决定续跑或停止；`run_all` 在普通 MD 后自动做增强采样。
- `enhanced.py`：chi1 上的 well-tempered metadynamics，在 OpenMM 里实现。一次最多偏置 `max_residues`（默认 10，`--enhanced-max-residues`）个最慢的侧链。各重复轮流跑，每轮 1 ns。
- `reweight.py`：重放高斯峰重建偏置势，Tiwary–Parrinello 重新加权，算状态比例和误差；`assess` 给每个侧链判断 reached / stalled / unreliable / not_yet；`populations.json`。用已知比例的模型和两个耦合侧链的模型测试过。
- `uncertainty.py`：分块平均的误差（考虑时间相关）、合并重复、有效样本数、误差下限（普通 MD 按转变次数，增强采样按有效样本数）。普通 MD 和增强采样都用。还有 `Precision`（容差 + 判据 + α + 判断次数，`met()` 判断够不够精确）和 `planned_looks()`（判断次数 = 预算 ÷ 每轮长度）。`judge`、`reweight.assess` 都接受 `Precision` 或一个浮点数（旧行为）。
- `cost.py`：计算成本。`of()` 按每条重复的长度算，`of_run()` 从 `run.json` 算（增强采样接着普通 MD 跑，不再付平衡时间）；`reduction()` 比两个成本。
- `budget.py`：策略对照（回放）。`Pool`（每条重复逐帧的状态）、`pool_from_run()`（从跑完的运行读，chi1 和距离都支持）、`markov_pool()`（已知比例的模型，不需要 GPU）、`replay()`（一种策略跑一遍）、`reference()`（独立参考：真值或没用到的重复）、`check_against()`（accurate = 真的在容差内；equivalent = TOST 证明了）、`compare_strategies()`（多次重复）、`write()`。四种策略：fixed / early_stop / add_replicas / mdagent。**mdagent 策略的规则是 `policy.py` 规则的镜像**，改 `policy.py` 的规则时要同步改 `budget._decide`；两边共用 `policy.spread_labels` 和 `policy.stalled_labels`。
- `scripts/budget_experiment.sh`：通用的策略对照（数据池 + 回放四种策略、三种判据）。
- `allocation.py`：agent 的分配和停止规则（`next_step`、`allocate`），日常运行和实验 A 共用。`landscape.py`：φ/ψ 自由能图。
- `basins.py`、`experiment_a.py`、`scripts/experiment_A.sh`：主链构象区和实验 A（见上面"实验 A"）。`budget` 的 `--fixed-grid`、`--baseline`、`--holdout`、`--items`、`--target-rate` 和 `hypothesis_a()` 是上一版实验 A 的做法，仍可用于通用对照。
- 目标、精度判据（`criterion`、`alpha`）和权限在 `agent.Goal`（`run.json` 的 `goal`）；`RunState.precision()` 组装成 `Precision`；停下的原因 target_reached / stalled / budget_exhausted / failed 记在 `run.json`（`notes.plain_md_outcome`、`enhanced.outcome`、失败时 `notes.outcome`）。
- `messages.py`：运行发给对话的消息。
- `policy.py`：第 4 步，每轮后选下一步动作（停、加长、加重复并补跑、超出预算延长、建议重新播种、上限内达不到就提前停），先查权限；没有权限就在对话里建议，并写明怎么确认。记录在 `run.json` 的 `notes.actions`。增强采样的同类规则写在 `agent.run_all` 的循环里。 按成本分配时用 `allocation_next`（`allocation.py`）。
- `report.py`：第 6 步，运行结束时写 `report.zh.md`、`report.en.md`（结论、误差、成本、停下原因、动作、限制、文件、遗留问题）；`equivalent()` 是等效性检验（TOST），`compare` 用它判断两个运行是不是给出同一个答案（容差默认取被比较的运行要求的精度，`--tolerance` 可改）。命令 `report`、`compare`、`budget`。
- `scripts/step5_validation.sh`：第 5 步在 GPU 上跑的场景（达标、困住、预算用完、增强采样，加参考运行），汇总在 `runs/val_<蛋白>_summary.txt`。默认用 1UBQ（用户要求换掉 181L 作为主例子；`--uniprot P0CG48 --termini charged`），也可用 181L。
- `observables.py`：结果量层。每个量换成逐帧的状态，再用同一套诊断、误差和停下判断。还有“看到某个变化”的目标（`EventSpec`、`first_entry`、`diagnose_events`）。`chi1`（g+/t/g-）和 `distance`（近/远，`A:88:OH-A:111:CG1@0.6`，阈值 ±0.05 nm 的缓冲）。`--quantity`、`--distances`；对话里 `target_kind` 为 `distance`。
- `adaptive.py`：可选的 AdaptivePy 自适应采样（`--sampling adaptive`）。`msm.py`：自适应采样的比例（马尔可夫状态模型）。
- `envcheck.py`：启动时检查平台和力场文件；用选中的平台先算一个两原子体系，算不了就停下并说明修法（例如 CUDA 组件比驱动新时的 PTX 错误，改装匹配的 `cuda-version`）。
- `backends.py`：语言模型后台（`claude -p`、`codex exec`、Anthropic API），统一返回结构化结果。claude 后台会去掉环境里的 `ANTHROPIC_API_KEY`，以使用 Claude Code 的订阅登录。
- `chat.py`、`chat_page.html`：`chat` 命令，本地网页（`http://localhost:8765`），多个对话，存在 `runs/.chats/`。运行卡片显示电脑名、完整路径、文件列表和下载链接（`/api/file`、`/api/download`，只提供 `outputs.py` 列出的文件）。
- `outputs.py`：列出一个运行的结果文件（`protein.pdb`、轨迹、chi1 和偏置势数据），以及电脑名和完整路径。命令行的 Summary 和网页都用它。
- `terminal.py`：终端对话（`chat --terminal`）和 `start` 菜单（选择记在 `runs/.chats/start.json`）。
- `llm.py`：`ask` 命令和设置表 `RunPlan`。`ask` 也用 `backends.py`，不再只能用 API 密钥。
- `cli.py`：命令行 `start`、`run`、`resume`、`diagnose`、`check`、`info`、`fetch`、`ask`、`chat`、`report`、`compare`、`budget`、`experiment-a`。
- `.claude/commands/`：Claude Code 项目命令 `/md` 和 `/md-status`。

测试：`pytest -q tests`（约 4 分钟，含在 CPU 上跑小肽段的测试）；`pytest -q tests -m "not slow"` 只跑快速测试。测试离线运行，UniProt 序列在 `tests/data/uniprot/`。测试体系 `examples/181L.pdb`（T4 溶菌酶 L99A 与苯，口袋里的 Val111 侧链在常规 MD 中很难翻转）。

## 环境

- **用户的电脑：** 项目在 `~/Desktop/MDAgent1`，conda 环境 `mdagent`，OpenMM 8.6.1 带 CUDA，GPU 上约每天 300 到 600 ns。用户有 Claude 订阅，但没有 API 密钥。
- **不要在用户的 conda 环境里装 openmm-plumed：** conda-forge 上的版本会把 OpenMM 降到 8.2 并丢掉 CUDA，而且其中的 PLUMED 不带 OPES。
- **云端会话**（Claude Code 网页版的临时容器）：没有 GPU，只适合很短的流程测试；装不了 AdaptivePy；没有 API 密钥。改动推送到 GitHub `Lyn-666/MDAgent1` 的 `main` 分支，用户和测试者都从 `main` 拉代码。有的会话在开发分支（例如 `claude/admiring-noether-6mlzki`）上工作：推送前先把 `main` 合并进来，再快进推到 `main`，不然用户 `git pull` 拿不到（2026-11-20 就因此在 masi-pc 上报 `invalid choice: 'experiment-a'`）。两个会话可能同时改代码，合并时两边功能都保留。最早的 `zcorn2017/binding-lang` 仓库已恢复原样，不再使用。
- **第一位测试者（2026-11-19，masi-pc，项目在 `~/MDAgent1`，不是 `~/Desktop/MDAgent1`）：** `~/.condarc` 里有 `solver: classic`，conda 卡在 “Collecting package metadata”，加 `--solver=libmamba` 解决；直接用 `pip` 会跑到系统旧 pip，要用 `python -m pip`；CUDA 报 PTX 版本错误（conda 装了 13.5，驱动 610.57 只支持 13.3），装 `cuda-version=13.3` 后 CUDA 可用（1UBQ 约 854 ns/day）；RTX 4090 + Intel CPU，CPU 8 和 9（物理核 16）不稳定（系统日志里 44 次段错误都在这两个 CPU 上），快速测试在 `addSolvent` 里段错误，要用 `taskset -c 0-7,10-31` 运行测试和 `mdagent start`（子进程继承限制）。日志查法和 `scripts/find_bad_cores.py` 见 README。11-19 在这台电脑上开始跑第 5 步（1UBQ）：reach_chi1 已完成且判断正确：ASP32 普通 MD 达标；LYS63、GLU16 各有一条重复几乎不翻转（每条至少 4 次的规则），普通 MD 用到 20 ns 上限后改用增强采样并达标；LYS63 约 89% 在 g+（+60°），和普通 MD 5 条重复的主要状态一致，说明重新加权正确。LYS63 翻转慢，不适合当“普通 MD 就能达标”的例子。11-20 第 5 步 10 个场景全部跑完（约 20 小时），停下原因都和实际一致（trapped_dist 判为达标是对的：Lys27–Asp52 距离在三条重复里来回 5、4、8 次，主要为“近”，误差 0.012；原以为它不会断开）；发现三个问题：`diagnose` 命令对距离运行误按 chi1 诊断（已修：和运行共用 `agent.diagnose_run`）；从没出现过的状态分块误差为 0，`compare` 因此误报“不一致”（LYS63 g- z=3.5，VAL26 z=inf；已加误差下限，`compare` 差别不到 0.01 也算一致）；`budget_chi1` 普通 MD 预算 2 ns，换增强采样后又用了默认 20 ns（和“用户定的是上限”不符，见上面“核心目的”）。
- **用户怎样打开网页（还没确认）：** 截图里浏览器是 Mac 上的 Chrome，地址是 `localhost:8765`，但运行的进程号是 1749823，macOS 上不会出现这么大的进程号。所以模拟很可能跑在一台 Linux 电脑上，网页经 SSH 端口转发打开，结果文件在那台 Linux 电脑上。
- **只能在用户电脑上验证的事项：** 网页下载链接经端口转发是否可用（还没验证）；AdaptivePy 的真实调用（已验证可用）；Claude Code 后台（已验证可用）；Codex 后台（还没验证；测试者会用到，公开前要验证）；`start` 加终端模式用 Claude 跑完开发测试（已验证可用）；181L 完整运行在 11-16 00:23 左右、增强采样第 1 条轨迹 5.4 ns 处停了，日志没有报错。已排除重启（11-17 08:47，非正常关机）和 SSH 断开（在 tmux 里跑，KillUserProcesses=no）；系统日志显示整台电脑在 11-16 00:22:54 之后就没有任何记录，直到 11-17 08:47 重启，所以是整机死机（不是程序或内存的问题），原因可能是 GPU、驱动、电源或散热。现在 run/resume 会在日志里记下进程号、开始时间、崩溃或 SIGTERM 时的调用栈，进度行带时间；CHARMM 加帽体系在 GPU 上的长时间运行（还没验证）；第 5 步第一次跑（1UBQ reach_chi1，11-19）：加重复的规则太激进，重复从 3 条加到 10 条（已改：只在重复明显不一致、且预算用了不到一半时加）；后来速度从 646 掉到 4 ns/day，同时一个 python 进程占了 23.4 GB 显存、GPU 使用率 0%，说明 OpenMM 悄悄改用了 CPU（已改：总是明确指定平台，GPU 不能用就报错）；占显存的是不是 MDAgent 自己的进程，还在确认。
