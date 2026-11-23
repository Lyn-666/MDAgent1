# CLAUDE.md

这个文件是本项目的记忆。Claude Code 每次在这个仓库里开始对话时都会读取它。

**更新规则：** 用户提出新的要求或偏好、做出设计决定、发现只能在用户电脑上验证的事项，或者项目状态有重要变化时，在同一次工作里更新本文件对应的栏目，和代码一起提交推送。删掉过时的内容，不要只往后追加。保持简短；细节写进 README。

## 与用户沟通的要求（每次都必须遵守）

开始前，用一句话说明你准备做什么；工作中简短更新进展，帮助用户跟上。结束时给出一段可以独立理解的简短回顾，说明发现了什么、做了什么，以及接下来如何，让只看到最后一条消息的读者也能了解全貌。

清楚沟通；面向聪明、但不了解当前上下文的读者写作。读者没有看过你的工具调用、推理过程或打开的文件，也可能刚从其他工作切换过来。消息通常应能独立理解：读者无需翻找其他上下文，就应明白你的意思。把关键点放在开头几句话里。通过句子顺序逐步展开信息，不要在第一行就使用要到第十二行才能看懂、且尚未定义的指代或概念。

用平实词语，写清楚、好懂的完整句子。不要使用工作过程中自己造出的术语或缩写，以任何读者都能理解的方式说明它是什么。写作总体上参考 ASD-STE100 简化技术英语的风格：使用短句，每句表达一个意思，采用主动语态，对同一件事始终使用同一个词。观点彼此相连时，用连贯自然的段落；只有列举独立项目时才使用项目符号。除非读者要求，否则省去不必要的细节，尽量简洁，但要确保保留关键背景。努力让面向用户的文字尽可能清晰易读。

用户使用中文交流，回复用中文。给用户的命令要能直接复制运行，并包含 `cd ~/Desktop/MDAgent1 && git pull`。**用户在测试机上运行时（项目在 `~/lynn/MDAgent1`，2026-11-23 起；以前在 `~/MDAgent1`），命令用 `cd ~/lynn/MDAgent1 && git pull`，跑模拟和测试要加 `taskset -c 0-7,10-31`。**

**测试要和用户使用时一样（用户的要求，2026-11-20）：** 让用户验证功能时，不给脚本和文件路径，而是让用户打开 agent（`mdagent start`），在对话里用一句话把任务交给它，由 agent 自己下载结构、搭体系、决定怎么跑、交付报告。离线基准（`experiment-a`、`scripts/experiment_A.sh`）是开发者用来评估 agent 的工具，不是给用户的测试方法。

**提交署名（用户的要求，2026-11-23）：** 贡献者要显示为用户本人，不要显示 Claude。每次会话开始先运行 `git config user.name "Lin (Lynn) Zhu"` 和 `git config user.email "73653622+Lyn-666@users.noreply.github.com"`（GitHub 隐私邮箱）；提交说明里不加 `Co-Authored-By: Claude` 或 `Claude-Session` 这类行。2026-11-23 用户同意后改写了 `main` 的全部历史（作者全改为用户、去掉 Claude 署名行，代码内容不变），强制推送了 `main` 和 `claude/brave-allen-dwswih`；改写前的完整备份是会话临时目录里的 git bundle（会话结束会消失）。远端另外四个旧的 `claude/*` 分支还是旧历史。

## 项目定位（用户的决定，2026-11-23，最新）

**实用第一，文章第二。** 主要目的是真的减少不必要的 MD 模拟；帮不懂 MD 的用户完成普通 MD、自适应采样、增强采样；用户只用一句话说想做什么，agent 用最省的方式完成。**默认由 agent 规划（用户的决定，2026-11-23）：** 用户没有把方案说死时，`mdagent run` 和对话默认 `--allocation projection`，上限 10 条、每条 2 µs（`cli.resolve_defaults`、`PLANNED_MAX_NS`），权限 `stop_early,add_replicas`（加重复在上限内，不算越界；超过上限 `extend` 和换方法 `switch_method` 仍要用户同意），`--burn-in auto`；结果够了就停。固定方案（按轮，默认 3 条、每条最多 20 ns）只用于：用户说死数字（“正好 3 条、每条 10 ns”，对话填 `mode guided` + `allocation rounds`）、自适应采样、“看到某个变化就停”。开发测试加 `max_replicas 2`。**其他都好了就不空等（用户的决定，2026-11-23）：** 上限放大后，一个从不变化的项目按预算规则要等很久（约所有重复合计 2.5 µs）。现在只要变化过的项目都已达到精度、只剩从不变化的项目，agent 就停下普通 MD（`allocation` 的 `others_done`，`RULE_VERSION` 7）；这些项目单独标为证据不足，有 `switch_method` 权限时转去增强采样，没有时在对话里建议“用增强采样继续”。只剩从不变化的项目、没有别的项目时，仍按预算规则。开发者脚本显式写 `--allocation rounds`。用真实 `claude` 测过四句话的翻译（只说想看什么 → 全默认；说死数字 → guided + rounds；只限普通 MD → 不加多余权限；给上限并允许换方法 → stop_early,add_replicas,switch_method）。文章投会议的 workshop。之后邀请别人测试，**体系由测试者自己选**。面向用户的说明在 `README.md`（只写用户需要的：能做什么、安装、一句话怎么说、agent 怎样省计算、结果和常见问题）；开发和研究内容（所有选项、算法、预算对照、实验 A、第 5 步验证、决策核心）在 `docs/developer.md`，架构和评估在 `docs/architecture.md`。改用户可见的行为时，同步改 README。

## 项目目标（用户最初的要求）

用户输入蛋白（PDB 编号、PDB 文件或准备好的 Amber 体系）和目标区域。agent 自动处理蛋白、搭水盒子、加离子、跑 MD；跑一段时间后自动判断目标区域里哪些侧链没采样到或各条轨迹太相似，再对这些侧链做增强采样。整个过程在一个 agent 窗口里完成。

**用户后来说明的目的（2026-11-17）：** 用户描述任务后，agent 按文献建议方法和设置（普通 MD、自适应采样、增强采样；alchemical 只推荐外部工具），但跑什么由用户决定。默认 agent 自己判断（`--mode auto`）：边跑边分析，自己决定何时换方法、何时停、何时交付。用户指定方法时（`--mode guided`）照做，没有进展就在对话里发消息建议换方法，不暂停、不自己换。核心：再加重复或加长对结果没有增益时，agent 停下。

**用户说明的核心目的（2026-11-20，修正）：** 目的是完成整个模拟、得到用户想观测的结果，但要边跑边分析。用户开跑前定好模拟时间和重复数（上限）；agent 看到用户指定的结果（例如某个结构变化）就提前停；如果再加长、加重复没有收益，也停下，并建议换方法（例如自适应采样）。自适应采样也按同样规则跑。**用户的决定（2026-11-20）：换方法只建议，用户确认后才做；不能超过用户定的时间和重复数，但可以发消息提醒。**已做：默认模式权限只有 stop_early；policy 第 6 条“上限内达不到”（乐观估计超过上限 1.5 倍就停，原因 not_reachable）；加长、加重复、换方法的建议都写明怎么确认（`resume --max-ns/--replicas/--enhanced-sampling`，对话里也能说）；误差下限。已做：目标类型“观察到某个变化”（`--observe "A:111=t"` 或 `"<距离>=far"`，从别的状态进入并停留 `--hold-ps`（默认 100 ps）才算，`--seen-in` 条重复看到就停；goal quantity `event_seen`）。已做：自适应采样的比例用马尔可夫状态模型算（`msm.py`：核心集状态、细致平衡的最大似然转变矩阵、对轨迹段 bootstrap 的误差）。

**通用化方向（2026-11-20，用户要求）：** MDAgent 要成为通用的科学 MD agent：理解任务目标、执行合适的模拟、评估证据、在预算和权限内做有依据的决定、少花不必要的计算。1UBQ（和 181L）只是开发和评估用例，不是设计目标；不要把任何体系、观测量、转变时间、重复数或模拟长度的假设写进通用决策。通用推理和具体的数值估计要分开；不能对不同的观测量用同一个收敛阈值；启发式论证不能当作统计结论。评估分开发、校准、保留测试三组体系，并和固定预算、常规逐轮停止规则比较。架构评估、目标架构、分阶段计划和评估计划在 `docs/architecture.md`。第 1 阶段（通用决策核心 `mdagent/core/`）和第 2 阶段（OpenMM 适配器、影子模式：新核心每轮记下自己的决定 `decisions.core.jsonl`，不执行；`mdagent shadow` 用已有文件重放已跑完的运行并对比）已完成，正式运行路径的决定还没改。下一步：用户在 GPU 电脑上对第 5 步的运行跑 `mdagent shadow`，逐条看分歧，再做第 3 阶段（在开关后面切换到新核心）。第 3 阶段要先解决：换方法后，只有旧方法能采样的估计量怎么收尾。**骨架 φ/ψ（2026-11-21，用户要求）：** 不为某个固定的蛋白，而是通用的：用户提出要求时，agent 给出方案（方法、开始的重复数、上限、停止条件）并在确认后执行。已做：`--quantity backbone`（任何残基的 Ramachandran basin 比例、basin 之间的转变统计、φ/ψ 自由能面，各有自己的精度要求，三样都达到才算达标）；basin 定义是数据，可用 `--basins` 换；对话里“你来决定重复数和长度，最多 N 条、每条最多 M ns”会转成：先开 3 条、`--max-replicas N --max-ns M --allow stop_early,add_replicas`，回复里写明方案。骨架的增强采样已做（2026-11-21，用户要求）：φ 和 ψ 上各一个一维 well-tempered metadynamics 偏置势（`enhanced.dihedral_quads(kind="backbone")`，列顺序 φ1、ψ1、φ2…），重新加权给 basin 比例和自由能面（`backbone/*_fes_reweighted.*`），停下要比例和自由能面都达到精度；转变速率和停留时间只来自普通 MD（偏置改变动力学）。已知答案的模型测试通过；OpenMM 里真跑 10 个二面角的慢速测试通过。2026-11-21 起：一直不变化的项目什么时候放弃，默认不再用固定 10 ns，而是按剩余预算判断（95% 上限速率 3/已跑时间，剩余预算也看不到足够的转变才放弃，和新核心同一规则；`--no-flip-patience-ns` 仍可指定；旧运行续跑时保留它们存下的 10 ns）；`--extend-ns` 默认 max(5, max_ns/20)；骨架 φ/ψ 的已完成轨迹段缓存在 `backbone/cache/`。还没在用户电脑上用真实语言模型试过这句话的转换（第一次在测试机上启动时忘了 `conda activate mdagent`，现在程序会提示）。用户已同意（2026-11-20）：1UBQ 和 181L 算校准体系（已经用来调过规则）；保留测试体系要另选（最好由别人选）并先封存名单。

**用户重新定位的研究目标（2026-11-20，最重要，在另一个会话里说的）：** 项目不是"达到采样目标就停"的工作流，而是**在给定精度下用最少计算量得到和高预算参考一致的结果**（accuracy-constrained simulation budget optimization）。要回答的问题：给定体系、要测的量和预算，agent 能不能自己分配模拟时间、重复数和采样策略，用最小成本得到统计上可靠的结果。应用在两类任务上：(1) 普通 MD——跑多少 ns、几条重复、什么时候可以提前停；(2) 自适应采样——何时开新重复、从哪个构象重新起跑、每条跑多久、剩下的预算怎么分。注意自适应采样不等于增强采样；先做无偏 MD 的分配和重新起跑，更容易验证平衡量是否正确。

用户当时明确说：**先不要再加新的采样方法。** 先把现有的决策框架改成"精度约束下的预算优化"，再把同一套决策接口扩展到自适应采样。用户的三阶段路线：1 普通 MD 的提前停和重复分配；2 自适应采样的智能重新起跑（和 uniform restart、least-counts 比）；3 在难度不同的体系上做基准测试（独立种子、冻结策略，报告误差、失败率、GPU 小时、累计 ns、重复数、达到精度的时间）。

用户强调的几点：成本要以 total compute cost 为主要目标，replica 数、累计 ns、wall-clock 分开报告，不要把"最少时间"和"最少重复数"当成两个要同时最小化的指标（10×10 ns 和 1×100 ns 累计相同，但初始化开销和对慢跃迁的探索完全不同；GPU 够多时多开重复反而等得更短）；不同的量（rotamer 比例、自由能差、跃迁速率、MFPT）不能用同一套收敛判据；在线停止不能用高预算参考来指导，参考只能用于离线验证；`|z| ≤ 2` 不足以证明一致，要用等效性检验和校准过的序贯置信区间。

**用户的开发计划（2026-11-17，在上面这个新目标之前写的）：** 1 保存任务目标和权限（目标量、精度、预算；能否延长、加重复、换方法、提前停）；2 区分停止原因（达到目标、采样停滞、预算用完、运行失败；比例不再变化只是诊断，不是停止理由）；3 让分析可信（考虑时间相关性的不确定度、检查权重退化、验证多个耦合 chi1 的重新加权）；4 每轮后选择下一步动作（延长、加重复、重新播种、建议换方法、停止），执行前查权限；5 针对性验证（正常达标、所有重复同时困住、预算用完未达标）；6 完善交付，再扩展方法。**1–3 已完成（2026-11-17）。2026-11-18 用户同意先把"结果量"做成可替换的一层（`observables.py`），已完成：侧链构象和原子间距离两种量；增强采样只支持侧链构象。第 4 步已完成（`policy.py`），第 6 步的报告已完成（`report.py`），第 5 步的验证脚本已写好（`scripts/step5_validation.sh`），等用户在 GPU 上跑。** 5 里"三个场景"的逻辑测试已有（`tests/test_stop_reasons.py`），还缺用真实模拟检查停下时的误差。

**按新目标做了什么（2026-11-20）：** 加了三件事，都在 GPU 上没验证过。(1) 成本核算 `cost.py`：aggregate_ns（所有重复相加的生产时间）、compute_ns（加上每条重复各自的平衡时间，主要指标）、wall_ns（按同时能跑几条折算的等待时间）分开报告；运行每轮记下实测秒数（`run.json` 的 `notes.rounds`）；结果报告里写明用了多少计算量。(2) 精度判据 `uncertainty.Precision`：`--criterion standard_error`（默认，含义和以前一样）/`confidence`（α 平摊到预先知道的判断次数）/`sequence`（Robbins 置信序列）。默认没改，因为改成 `confidence` 会让运行长好几倍；先用 `budget` 量出代价再决定。(3) 策略对照 `budget.py` + `budget` 命令：一次高预算数据池，离线回放 fixed / early_stop / add_replicas / mdagent 四种策略，比成本和正确率；对"只加长、只加独立重复"的策略回放是精确的，"从挑出的帧重新起跑"回放不了。`compare` 从 `|z| ≤ 2` 换成等效性检验（TOST）。在已知比例的模型上看到（容差 0.05，10 次重复）：`early_stop` 省 83% 但 10 次里 2 次超出容差（最大偏差 0.097），`mdagent` 省 74%、9 次在容差内（最大偏差 0.057，和固定方案一样）；把判据换成 `confidence` 后各策略会跑更久，但在每条 20 ns 的预算下达不到这个判据，只是以"预算用完"停下，答案没变好——更严的判据要配更大的预算。

**实验 A（2026-11-20，按用户的设计重做）：** 问题：给定 MD 任务和事先定好的精度，agent 能否自己决定几条重复、每条多长、何时停，以最少计算量得到统计上可靠的结果（task-conditioned, uncertainty-aware budget optimization，不是收敛检测）。用户的要求：**在线停止判据只用自己的轨迹，离线成功判据才用独立参考，防止参考泄漏**；参考用同一力场流程的内部数据池（留出的重复），不用文献数值；测主链构象区（比例为主，另报自由能差、速率、角度图谱，各自单独判断，不能一并宣布收敛）；四个对照：固定方案（1 条和 3/4/5/8 条，按长度扫描）、Sequential-CI（最重要：agent 必须比它省）、Agent；报告成功率、过早停止率、区间覆盖率、C_stop 与 C_hit、GPU 小时、等待时间、成本–准确度曲线；预算不够时输出 INSUFFICIENT_EVIDENCE 也是正确行为；固定方案的成本取它自己达到同样可靠性所需的成本。A-Easy：预实验挑出的快速 φ/ψ 残基；A-Hard：50 环 D52/G53 β-I/β-II 转角。文献（PubMed 核对）：这个翻转约 8 µs 来回 7 次、NMR 约 13–15 µs、β-II 约 4%（Arasteh 2021 doi:10.1021/acs.jpclett.1c00778；Smith 2016 doi:10.1073/pnas.1519609113），所以 16×100 ns 的数据池建不起 A-Hard 的参考，A-Hard 只检验"会不会诚实地说证据不足"。代码：`basins.py`（rama:A:73、turn:A:52，核心半径 50°、过渡式分配）、`experiment_a.py`（全部逻辑，命令 `experiment-a`，`--model` 不需要 GPU，`--select` 用于预实验挑残基）、`scripts/experiment_A.sh`（pilot/select/pool/analyze）。agent 规则：一次只规划的查看（不停、不消耗 α），之后最多 3 次可停的查看（Bonferroni 按 3 次）；状态变化太少、根本不可能停的查看不算进这 3 次（2026-11-20 改，起因：1UBQ 运行的第 2 次查看只是因为变化太少而加长，却占了名额，会让运行在每条约 16 ns 时以"查看次数用完"停下；继续旧运行时按记录重新数一遍，`policy.recount_tests`）；`allocate()` 在加长和加重复之间选成本最低的，只有重复间差别显著（卡方检验）才加重复；最后一次查看前余量更大；用掉四分之一预算后，投影超过预算 3 倍才放弃。普通 MD 回答不了的单个项目（一次没变化、而且按 95% 置信上限的最快速率也要超出预算才能达到精度——`allocation.stuck_projection`，2026-11-20 起取代固定的耐心时间：1UBQ 运行在每条 16 ns 就放弃了，而最乐观只需约 336 ns、预算 800 ns；或它自己的投影远超预算）**单独**标为证据不足、搁置，其余项目照常跑到精度；只有全部都回答不了时整个运行才说证据不足；规则改动时提高 `allocation.RULE_VERSION`，旧运行"继续"时会按新规则重新决定（2026-11-20，起因：1UBQ 任务里 52/53 转角约 10 µs 才翻转，会连累残基 8–10）。规则只在模型上调过，现已冻结。旧设计（4 条起步、不丢弃开头、共用一个数据池，只相当于约 1 个独立单位）的模型干跑：快模型上 agent 100% 成功、0 过早停止、比 Sequential-CI 省约 32%，但比事后读出的最短可靠固定方案贵（原理上躲不开，约为区间系数平方比）。旧的 `budget.py`/`hypothesis_a` 保留作通用策略对照，不再是实验 A 的主体。默认容差 0.05、可靠性 90%，由用户在看结果前确认。**还没在 GPU 上跑。**

**审稿意见和修改（2026-11-23，用户要求按审稿意见修改）：** 我按顶会审稿人的角度审了项目：主要结论还没有真实 MD 的证据；共用一个数据池的试验不独立，成功率区间被算窄；只有 3–4 条重复却用正态分位数；合成模型从平衡态出发，测试不会失败；初始构象偏差没有处理；对照太弱；四套决策逻辑并存；保留测试集还没选。原则是“先修尺子，再花 GPU”。已做（都不需要 GPU，`docs/architecture.md` 3.4 节）：(1) 论文评估的 agent 是 `allocation.next_step`（投影分配）；论文前打 git 标签冻结。(2) 保证的对象是每个量：停下那一刻每个项目的区间包住参考的比例（`per_item`、`coverage_at_stop`）；“所有项目同时对”另报。(3) 重复之间的差别经卡方检验确实存在时（`uncertainty.replica_spread_real`，和分配规则同一个检验）用 t 分布（自由度 n−1）：实验 A 的判断和覆盖率、`allocate(t_dist)`、日常运行 confidence 判据下的判断（`diagnose.judge`）、骨架自由能面误差；standard_error 判据不变。第一版只要差别偏大就用 t，计划和判断用的系数不一致，简单模型上 23% 的试验用完查看次数，已改。`RULE_VERSION` 升到 5。(4) 自动丢弃开头没平衡的部分（Chodera 2016，`uncertainty.joint_equilibration_start`、`basins.trim`、`LoopSettings.burn_in`）：所有重复共用一个起点，有效样本数至少增加 1.5 倍、构象变化中位数至少 20 次才丢（第一版每条重复各选起点，审查发现会让平衡数据偏 0.014–0.023，已改，现在偏差小于 0.001）；判断“一直不变化”和投影只用保留下来的时间（`cut_ns`、`burn_in_ns`）；新的 projection 运行默认 `auto`，按轮方式和旧运行保持 `none`；实验 A 所有方法都用。(5) 实验 A：每次试验新数据池（`--family`，参考 = 真值）、互不重叠的组（`--disjoint`）、共用数据池（报告写明独立单位约 1）；四类合成模型 equilibrium、biased_start、two_timescale、hidden_slow_start；新对照“预实验后定固定方案”（`pilot_fixed`，进入判定第 2 条）；agent 起步改为 3 条（和对话方案一致）；`--sensitivity` 一次只改一个常数。推迟：速率误差在来回跳时的覆盖率、Kish 有效样本数考虑时间相关、构象区边界的敏感性、图谱 JS 阈值的依据。**模型上的当前结果（最终规则，每类 30 次独立试验）：** agent 成功 / 错误停止 / 计算量：equilibrium 93%/0%/196 ns（8 条 Sequential-CI 100%/205 ns）、biased_start 70%/0%/290 ns（各方法都降了：共用起点切得保守，剩一点偏差，就说证据不足）、two_timescale 60%/0%/476 ns（第一轮 47%、错误停止 3%；8 条 Sequential-CI 83%）、hidden_slow_start 0%/0%（第一轮错误停止 7%）；120 次里没有一次错误停止；只在最简单的模型上略省，假设 A 仍不成立。端到端测试（云端 CPU，小肽段）：`ask` 用 `claude` 把一句中文请求翻成 backbone + projection + 允许 extend,add_replicas,stop_early，按这个命令真跑、诊断、决定、写中英文报告都正常；CPU 太慢，肽段没翻转，agent 正确地说证据不足；加长、加重复的路径要在 GPU 上看。**用户的决定（2026-11-23）：** 合并到 `main`（已做）；two_timescale 的弱点按“差别有迹象就加重复”修：`Limits.spread_level` 0.8，只有连乐观的差别（95% 置信下限，`allocation.optimistic`）也达不到时才放弃，否则先加重复测准（`explore_replicas`，同时每条多跑一段）；`RULE_VERSION` 6。独立代码审查找出的 7 个问题都已修（见 `docs/architecture.md` 3.4）。合成模型族改成满足细致平衡。**A-Hard 怎样算对（用户的决定，2026-11-23）：** 参考建不起来时不判准确度，判诚实性（`experiment_a.honesty`，`verdict["a_hard"]`）：一次试验要同时做到 (1) 输出证据不足、不宣布收敛，(2) 理由来自数据（`Trial.kind` 为 stuck、hopeless 或 others_done），(3) 放弃前用掉的计算量不超过预算的一半（`Design.honest_share`）；诚实的试验占比达到可靠性目标（90%）才通过。默认上限改成 10×2 µs 后，日常运行跑 1UBQ 时可能真的看到这个转角翻转几次；这只影响用户的运行，不影响实验 A（它用自己的数据池和预算）。**等用户决定：**要不要让重复从不同结构出发（能发现隐藏慢状态，但要马尔可夫模型校正，等于加方法）；谁选保留测试体系、何时封存；GPU 数据池多大（论文需要独立试验，例如 40×100 ns、留 8 条、分 4 组，要几天）。

**用户的要求（2026-11-20）：agent 不能是专门为实验 A 写的。** 用户在对话里打出"使用 1UBQ，通过 standard unbiased MD 估计指定残基的 Ramachandran basin populations、transition statistics 和 φ/ψ free-energy landscape，你可以自行决定 replicas 数量、每条模拟长度和停止时间，在保证准确度的前提下最小化总 simulation cost"时，agent 自己就要完成实验并交付预期结果；换成别的实验也一样。已做：决策规则在 `allocation.py`（`next_step`），日常运行（`--allocation projection`，`policy.allocation_next`）和实验 A 回放（`experiment_a.run_agent`）调用同一份代码；主链构象区是通用测量量（`--quantity backbone --backbone "rama:A:8-10;turn:A:52"`，`backbone.diagnose_backbone`（2026-11-22 合并后；自由能差和主转变仍用 `basins.summarize`，和实验 A 共用））；对话设置表有 `target_kind` backbone、`backbone`、`allocation`、`budget_ns`、`max_replicas`；交付报告有"每个量单独判断""agent 的决策"和成本章节，图谱在 `landscape.py`（`landscape_*.svg/.csv`）。用真实的 `claude -p` 测过用户这句话：没给残基编号时它先追问；给了编号就填出 backbone + projection + 允许 extend,add_replicas,stop_early（不换增强采样）。2026-11-23 起 projection 是自动模式的默认（见“项目定位”）。

**两个分支合并（2026-11-22，用户选的方案）：** 11-20 起有两个会话各自开发：`claude/admiring-noether-6mlzki`（上面的研究目标、`allocation`/`cost`/`budget`/实验 A、`rama:`/`turn:` 写法）和 `claude/compassionate-lamport-1xn8lc`（通用决策核心、影子模式、φ/ψ 转变统计和增强采样）。合并在后者上，两边的提交都保留；2026-11-22 用户同意后快进合并到 `main`（`bb52379`），测试者从 `main` 拉代码。用户的三个决定：(1) rama 的 basin 默认按**分区覆盖全图**（alpha_R、beta 含 PPII、alpha_L、other，核心区边缘 10°），三个中心加 50° 半径保留为 `--rama-basins centres`；合并前开始的 `rama:` 运行续跑时自动用 centres，不中途换定义；β-turn 用中心。(2) 用户把重复数、长度、停止交给 agent 时，用 `--allocation projection`（对话会这样填）。(3) 精度标准按方式分开：projection 用按查看次数校正的置信区间，按轮方式仍用标准误差。骨架只有一套计算：`backbone.diagnose_backbone`（输入 `basins.BasinSet`，分配用 `basins.assign`，自由能差和主转变用 `basins.summarize`，所有速率、停留时间、带误差的自由能面、缓存、增强采样用 `backbone.py`）；`observables.diagnose_backbone` 已删。项目名统一为 `rama:A:110`；我这边合并前的骨架运行（`run.json` 的 `backbone` 是字典）续跑时自动换成 rama 项目，设置移到 `backbone_settings`。另一个分支里写死的“泛素 50 环约 10 µs”的建议已删（违反通用化规则）。

**关于“先不要加新采样方法”：** 用户 11-20 在另一个会话里这样说；11-21 又在这个会话里明确要求加 φ/ψ 的增强采样（已做）。以较新的要求为准；做新方法前先问用户。

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
- **公开计划：** 先保持私有，邀请少数同学测试（流程在 `TESTING.md`，版本 `0.1.0b1`，对应提交 `3a37da9`（2026-11-23 改写历史后的编号）；云端会话不能推送 git 标签，标签 `v0.1.0b1` 要用户在自己电脑上打）；公开前要完成：科学修复（已完成：跳过脯氨酸和二硫键半胱氨酸、增强采样最多 10 个侧链）、测试者材料（已完成：`environment.yml`、README 快速上手、Issue 模板、GitHub Actions 自动测试）、许可证（已完成：MIT，版权人 Lyn-666）、用户在 GPU 上的两项验证（见 `docs/developer.md`“下一步”；181L 完整运行的普通 MD 部分已通过：VAL111 在 3×20 ns 里没有翻转，被送进增强采样，10 个侧链的上限生效；增强采样里 VAL111 第 1 条轨迹 5 ns 翻转 373 次，三种状态都去过，也已通过）、删掉 CLAUDE.md 里只和用户个人电脑有关的记录（2026-11-23 已做：用户名、硬件故障和死机记录移到 `docs/developer.md`，去掉了身份信息）。
- **开发阶段测试运行要很短（几分钟）**：`--replicas 2 --first-ns 0.1 --extend-ns 0.1 --max-ns 0.3 --no-flip-patience-ns 0.25 --enhanced-ns 0.1 --equil-ps 60`。
- **长上限也可以（2026-11-21，用户说明）：** 用户可以把上限定得很长（例如每条 1 µs），agent 在上限内自己决定重复数、长度和何时停；用户允许时也可以换方法。

## 代码结构

代码在 `mdagent/`，完整流程写在 `docs/developer.md` 的“完整流程”。

- `fetch.py`：按 PDB 编号从 RCSB 下载到 `structures/<ID>.pdb`。
- `check.py`：搭体系前的结构检查。拿不到参考序列（下载失败或 UniProt 条目已失效）时不判断标签、不删任何残基；失效条目会列出替代编号；`--uniprot` 可以替换文件里的编号。 按 DBREF/SEQADV 和 UniProt（含异构体，缓存在 `structures/uniprot/`）逐位比对；找标签、多拷贝、金属、末端帽、未知残基、长缺口、断链；判断每条链两端是否要加帽。
- `build.py`：搭体系（PDBFixer）。选链、去标签、加帽（CHARMM 下帽并入末端残基作为补丁，Amber 下是单独残基）、质子化、加水和离子。`load_amber` 原样读入用户准备好的 Amber 体系（配体保留）。
- `protonation.py`：PROPKA 和手动质子化。
- `simulate.py`：分段跑 MD，可续跑；`equilibrate` 是分阶段平衡。
- `diagnose.py`：侧链 chi1 诊断和排序。脯氨酸和形成二硫键的半胱氨酸（按 SG-SG 距离判断）判为 `fixed_chi1`，跳过。`agent.py`：多轮循环，决定续跑或停止；`run_all` 在普通 MD 后自动做增强采样。
- `enhanced.py`：二面角上的 well-tempered metadynamics（chi1；骨架 φ/ψ 时每个残基两个），在 OpenMM 里实现。一次最多偏置 `max_residues`（默认 10，`--enhanced-max-residues`）个最慢的侧链。各重复轮流跑，每轮 1 ns。
- `reweight.py`：重放高斯峰重建偏置势，Tiwary–Parrinello 重新加权，算状态比例和误差；`assess` 给每个侧链判断 reached / stalled / unreliable / not_yet；`populations.json`。用已知比例的模型和两个耦合侧链的模型测试过。
- `uncertainty.py`：分块平均的误差（考虑时间相关）、合并重复、有效样本数、t 分布系数（`t_factor`、`Precision.factor(df)`）、自动平衡检测（`statistical_inefficiency`、`equilibration_start`）、误差下限（普通 MD 按转变次数，增强采样按有效样本数）。普通 MD 和增强采样都用。还有 `Precision`（容差 + 判据 + α + 判断次数，`met()` 判断够不够精确）和 `planned_looks()`（判断次数 = 预算 ÷ 每轮长度）。`judge`、`reweight.assess` 都接受 `Precision` 或一个浮点数（旧行为）。
- `cost.py`：计算成本。`of()` 按每条重复的长度算，`of_run()` 从 `run.json` 算（增强采样接着普通 MD 跑，不再付平衡时间）；`reduction()` 比两个成本。
- `budget.py`：策略对照（回放）。`Pool`（每条重复逐帧的状态）、`pool_from_run()`（从跑完的运行读，chi1 和距离都支持）、`markov_pool()`（已知比例的模型，不需要 GPU）、`replay()`（一种策略跑一遍）、`reference()`（独立参考：真值或没用到的重复）、`check_against()`（accurate = 真的在容差内；equivalent = TOST 证明了）、`compare_strategies()`（多次重复）、`write()`。四种策略：fixed / early_stop / add_replicas / mdagent。**mdagent 策略的规则是 `policy.py` 规则的镜像**，改 `policy.py` 的规则时要同步改 `budget._decide`；两边共用 `policy.spread_labels` 和 `policy.stalled_labels`。
- `scripts/budget_experiment.sh`：通用的策略对照（数据池 + 回放四种策略、三种判据）。
- `allocation.py`：agent 的分配和停止规则（`next_step`、`allocate`），日常运行和实验 A 共用。`landscape.py`：φ/ψ 自由能图。
- `basins.py`、`experiment_a.py`、`scripts/experiment_A.sh`：主链构象区和实验 A（见上面"实验 A"和"审稿意见和修改"）。`basins.trim`（丢弃开头）、`basins.state_factors`（每个构象区用 z 还是 t）；`experiment_a.FAMILIES`/`family_pool`（四类合成模型）、`run_pilot_fixed`、`sensitivity`。`budget` 的 `--fixed-grid`、`--baseline`、`--holdout`、`--items`、`--target-rate` 和 `hypothesis_a()` 是上一版实验 A 的做法，仍可用于通用对照。
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
- `cli.py`：命令行 `start`、`run`、`resume`、`diagnose`、`check`、`info`、`fetch`、`ask`、`chat`、`report`、`compare`、`budget`、`experiment-a`、`shadow`。
- `core/`：通用决策核心，只用 Python 标准库，不提任何观测量、引擎或体系（`tests/test_core_architecture.py` 检查）。`task.py`（任务、每个估计量自己的要求、预算、权限）、`evidence.py`（估计结果、有效性检查、共用信号）、`registry.py`（估计器按“种类 × 数据类型”查找；工具接口）、`reasoner.py`（相互竞争的假设，都标为启发式）、`controller.py`（候选动作、价值、成本、权限和预算检查、选择）、`ledger.py`（`decisions.jsonl`）、`loop.py`（循环、执行前后的检查、续跑）。
- `backbone.py`：任何残基的骨架 φ/ψ：basin 定义（默认 alpha_R、beta 含 PPII、alpha_L、other；核心区边缘 10°）、逐帧 basin、转变统计（速率 = 次数 / 停留时间，泊松误差；平均停留时间 = 总时间 / 离开次数）、自由能面（15° 一格，误差来自重复之间的差别加误差下限），结果写到运行目录的 `backbone/`。
- `tools_openmm.py`：给新核心用的 OpenMM 工具（`PlainMDTool` 包装 `run_segment`，`MetadTool` 包装 `run_replica` 并重建偏置势给出权重）和特征化（chi1、距离、事件；`event_series` 和 `agent._diagnose_events` 共用）。工具可以只看过去某一时刻的数据（重放用）。`shadow.py`：影子模式（每轮记下新核心的决定，出错不影响运行，`MDAGENT_SHADOW=0` 关闭）、重放、和原策略对比；命令 `shadow`。
- `estimators.py`：具体的估计器（无偏比例、重新加权比例、马尔可夫模型比例、首次进入事件），数值来自 `uncertainty.py`、`msm.py`、`observables.py`。`synthetic.py`：已知答案的合成马尔可夫链工具，给开发和测试用。
- `.claude/commands/`：Claude Code 项目命令 `/md` 和 `/md-status`。

测试：`pytest -q tests`（约 4 分钟，含在 CPU 上跑小肽段的测试）；`pytest -q tests -m "not slow"` 只跑快速测试（298 个，约 4 分钟；全部 309 个约 15 分钟）。审稿后的修改和独立审查修复的测试在 `tests/test_review_revisions.py`。骨架增强采样的测试在 `tests/test_backbone_enhanced.py`。骨架 φ/ψ 的测试在 `tests/test_backbone.py`（速率和停留时间用 30 个随机种子检查误差是否校准）。通用核心的测试在 `tests/test_core_*.py` 和 `tests/test_estimators.py`（约 7 秒，全在合成体系上）；适配器和影子模式在 `tests/test_shadow.py`（用设定好 chi1 的假运行，不跑 MD）。测试离线运行，UniProt 序列在 `tests/data/uniprot/`。测试体系 `examples/181L.pdb`（T4 溶菌酶 L99A 与苯，口袋里的 Val111 侧链在常规 MD 中很难翻转）。

## 环境

- **用户的电脑：** 项目在 `~/Desktop/MDAgent1`，conda 环境 `mdagent`，OpenMM 8.6.1 带 CUDA，GPU 上约每天 300 到 600 ns。用户有 Claude 订阅，但没有 API 密钥。
- **不要在用户的 conda 环境里装 openmm-plumed：** conda-forge 上的版本会把 OpenMM 降到 8.2 并丢掉 CUDA，而且其中的 PLUMED 不带 OPES。
- **云端会话**（Claude Code 网页版的临时容器）：没有 GPU，只适合很短的流程测试；装不了 AdaptivePy；没有 API 密钥。新容器里没有依赖：`pip install openmm mdtraj "pydantic>=2" propka pytest legacy-cgi`；pdbfixer 用 pip 装会失败，要 `git clone https://github.com/openmm/pdbfixer` 后把其中的 `pdbfixer/` 文件夹复制到 site-packages。改动推送到 GitHub `Lyn-666/MDAgent1` 的 `main` 分支（会话被指定了开发分支时推到那个分支，合并到 `main` 由用户决定），用户和测试者都从 `main` 拉代码。有的会话在开发分支（例如 `claude/admiring-noether-6mlzki`）上工作：推送前先把 `main` 合并进来，再快进推到 `main`，不然用户 `git pull` 拿不到（2026-11-20 就因此在测试机上报 `invalid choice: 'experiment-a'`）。两个会话可能同时改代码，合并时两边功能都保留。最早的 `zcorn2017/binding-lang` 仓库已恢复原样，不再使用。
- **测试机（项目在 `~/lynn/MDAgent1`）：** 用户的输出里项目路径是 `~/lynn/MDAgent1` 时就是这台。2026-11-23 换到这个文件夹后，`mdagent start` 报 `No module named 'mdagent'`（可编辑安装还指向旧文件夹），在新文件夹里 `python -m pip install -e .` 解决。运行测试和 `mdagent start` 要加 `taskset -c 0-7,10-31`（有两个 CPU 核不稳定）；conda 用 `--solver=libmamba`；装包用 `python -m pip`；CUDA 组件版本要和驱动匹配（`cuda-version=13.3`）。第 5 步的 10 个场景在这台机器上跑完过，结论和发现的问题见 `docs/developer.md` 的“GPU 上的验证记录”。
- **网页可能经 SSH 端口转发打开：** 模拟在 Linux 电脑上，浏览器在另一台电脑上；结果文件只在跑模拟的那台电脑上。
- **还要在 GPU 电脑上验证的事项：** 新的默认路径（agent 规划、加长、加轨迹、停下，2026-11-23）；Codex 后台；网页下载链接经端口转发是否可用；CHARMM 加帽体系的长时间运行。已验证可用：AdaptivePy、Claude Code 后台、`start` 终端模式。整机死机和 GPU 悄悄改用 CPU 的排查记录见 `docs/developer.md`。
