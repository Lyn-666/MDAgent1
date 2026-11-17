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

## 用户定下的规则和决定

- **不做增强采样之后的分析**（不算状态比例、不判断收敛、不据此自动加长）。用户明确要求去掉了这部分。只保存数据，并告诉用户文件在哪里。
- **只有序列和参考序列一致、结构完好干净，才开始正式的下一步。** 结构检查有问题就停，除非用户明确说继续（`--accept-problems`）。181L 和 UniProt P00720 有两处未说明差异（R12G、I137R），用 181L 测试要加 `--accept-problems`。
- **每项设置都有默认值。** 影响科学结论的选择（力场、离子、pH、质子化、末端）由用户决定；用户提出非默认要求时，agent 先执行（例如先跑 PROPKA），再开始 MD。所有搭体系选项都在 `SetupOptions` 里，聊天窗口也填同一张表。
- **默认值：** 温度 300 K（用户不要改）；力场 CHARMM36m；0.15 M NaCl；pH 7；立方盒子 1.0 nm；平衡 600 ps（按用户自己的 OpenMM 流程：骨架约束、最小化、5 K 升温、约束 NPT、逐步撤约束、无约束 NPT）；末端 `--termini auto`（真实蛋白在这一端还没结束时加 ACE/NME 中性帽，按 UniProt 全长判断；否则保留带电末端）。
- **启动的运行要在后台一直跑**，关网页、关聊天服务都不停；只有用户确认停止或运行完成才结束。
- **界面文字要有中文和英文两种语言。**
- **路径要让用户一看就懂**，例如 `structures/3HTB.pdb`、`runs/<名字>/`。
- **告诉用户结果文件在哪里时，写电脑名和完整路径**，并说明文件只在跑模拟的那台电脑上。网页的运行卡片上有下载链接（单个文件或 zip）。起因：用户跑完测试后在自己电脑上找不到轨迹。
- **测试者在自己的电脑上运行**，用自己的 Claude 或 Codex 登录。所以聊天网页不需要访问密码（只接受本机连接就够了）。**主要面向 Linux + NVIDIA GPU。** macOS 能运行但没有 CUDA，只适合短测试，不专门适配；Windows 只能通过 WSL2（程序用到 `os.killpg`、`pgrep` 和 Unix 的 C 库）。
- **语言模型只做"翻译"**：把用户的话填进固定的设置表，不能运行命令；开始、停止、继续都要用户确认（网页点按钮，终端输入 y）。
- **使用方式（用户的设计）：** 装好程序后运行 `mdagent start`，依次选界面语言、Claude 还是 ChatGPT（Codex）、网页还是终端。流程由我们的程序控制，不由语言模型控制；网页和终端共用 `chat.Assistant`。**尽量少用 Claude Code 和 Codex 的 token**：每句话只调用一次模型，提示里不放长列表，`/runs` 等本地命令不调用模型。Claude Code 的 `/md` 是 Claude Code 自己按说明工作，费额度，只给开发者用。
- **公开计划：** 先保持私有，邀请少数同学测试；公开前要完成：科学修复（已完成：跳过脯氨酸和二硫键半胱氨酸、增强采样最多 10 个侧链）、测试者材料（已完成：`environment.yml`、README 快速上手、Issue 模板、GitHub Actions 自动测试）、许可证（已完成：MIT，版权人 Lyn-666）、用户在 GPU 上的两项验证（见 README“下一步”；181L 完整运行的普通 MD 部分已通过：VAL111 在 3×20 ns 里没有翻转，被送进增强采样，10 个侧链的上限生效；增强采样里 VAL111 第 1 条轨迹 5 ns 翻转 373 次，三种状态都去过，也已通过）、删掉 CLAUDE.md 里只和用户个人电脑有关的记录。
- **开发阶段测试运行要很短（几分钟）**：`--replicas 2 --first-ns 0.1 --extend-ns 0.1 --max-ns 0.3 --no-flip-patience-ns 0.25 --enhanced-ns 0.1 --equil-ps 60`。

## 代码结构

代码在 `mdagent/`，完整流程写在 README 的“完整流程”。

- `fetch.py`：按 PDB 编号从 RCSB 下载到 `structures/<ID>.pdb`。
- `check.py`：搭体系前的结构检查。按 DBREF/SEQADV 和 UniProt（含异构体，缓存在 `structures/uniprot/`）逐位比对；找标签、多拷贝、金属、末端帽、未知残基、长缺口、断链；判断每条链两端是否要加帽。
- `build.py`：搭体系（PDBFixer）。选链、去标签、加帽（CHARMM 下帽并入末端残基作为补丁，Amber 下是单独残基）、质子化、加水和离子。`load_amber` 原样读入用户准备好的 Amber 体系（配体保留）。
- `protonation.py`：PROPKA 和手动质子化。
- `simulate.py`：分段跑 MD，可续跑；`equilibrate` 是分阶段平衡。
- `diagnose.py`：侧链 chi1 诊断和排序。脯氨酸和形成二硫键的半胱氨酸（按 SG-SG 距离判断）判为 `fixed_chi1`，跳过。`agent.py`：多轮循环，决定续跑或停止；`run_all` 在普通 MD 后自动做增强采样。
- `enhanced.py`：chi1 上的 well-tempered metadynamics，在 OpenMM 里实现，只跑和存数据。一次最多偏置 `max_residues`（默认 10，`--enhanced-max-residues`）个最慢的侧链。
- `adaptive.py`：可选的 AdaptivePy 自适应采样（`--sampling adaptive`）。
- `envcheck.py`：启动时检查平台和力场文件。
- `backends.py`：语言模型后台（`claude -p`、`codex exec`、Anthropic API），统一返回结构化结果。claude 后台会去掉环境里的 `ANTHROPIC_API_KEY`，以使用 Claude Code 的订阅登录。
- `chat.py`、`chat_page.html`：`chat` 命令，本地网页（`http://localhost:8765`），多个对话，存在 `runs/.chats/`。运行卡片显示电脑名、完整路径、文件列表和下载链接（`/api/file`、`/api/download`，只提供 `outputs.py` 列出的文件）。
- `outputs.py`：列出一个运行的结果文件（`protein.pdb`、轨迹、chi1 和偏置势数据），以及电脑名和完整路径。命令行的 Summary 和网页都用它。
- `terminal.py`：终端对话（`chat --terminal`）和 `start` 菜单（选择记在 `runs/.chats/start.json`）。
- `llm.py`：`ask` 命令和设置表 `RunPlan`。`ask` 也用 `backends.py`，不再只能用 API 密钥。
- `cli.py`：命令行 `start`、`run`、`resume`、`diagnose`、`check`、`info`、`fetch`、`ask`、`chat`。
- `.claude/commands/`：Claude Code 项目命令 `/md` 和 `/md-status`。

测试：`pytest -q tests`（约 4 分钟，含在 CPU 上跑小肽段的测试）；`pytest -q tests -m "not slow"` 只跑快速测试。测试离线运行，UniProt 序列在 `tests/data/uniprot/`。测试体系 `examples/181L.pdb`（T4 溶菌酶 L99A 与苯，口袋里的 Val111 侧链在常规 MD 中很难翻转）。

## 环境

- **用户的电脑：** 项目在 `~/Desktop/MDAgent1`，conda 环境 `mdagent`，OpenMM 8.6.1 带 CUDA，GPU 上约每天 300 到 600 ns。用户有 Claude 订阅，但没有 API 密钥。
- **不要在用户的 conda 环境里装 openmm-plumed：** conda-forge 上的版本会把 OpenMM 降到 8.2 并丢掉 CUDA，而且其中的 PLUMED 不带 OPES。
- **云端会话**（Claude Code 网页版的临时容器）：没有 GPU，只适合很短的流程测试；装不了 AdaptivePy；没有 API 密钥。改动直接推送到 GitHub `Lyn-666/MDAgent1` 的 `main` 分支。最早的 `zcorn2017/binding-lang` 仓库已恢复原样，不再使用。
- **用户怎样打开网页（还没确认）：** 截图里浏览器是 Mac 上的 Chrome，地址是 `localhost:8765`，但运行的进程号是 1749823，macOS 上不会出现这么大的进程号。所以模拟很可能跑在一台 Linux 电脑上，网页经 SSH 端口转发打开，结果文件在那台 Linux 电脑上。
- **只能在用户电脑上验证的事项：** 网页下载链接经端口转发是否可用（还没验证）；AdaptivePy 的真实调用（已验证可用）；Claude Code 后台（已验证可用）；Codex 后台（还没验证；测试者会用到，公开前要验证）；`start` 加终端模式用 Claude 跑完开发测试（已验证可用）；181L 完整运行在增强采样第 1 条轨迹 5.4 ns 处停了，日志没有报错，原因还在查（可能是 SSH 断开后系统结束了进程）；CHARMM 加帽体系在 GPU 上的长时间运行（还没验证）。
