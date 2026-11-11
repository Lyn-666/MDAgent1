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

- 配体会被去掉，只模拟蛋白本身。
- PROPKA 只在用户要求时运行，预测的是去掉配体后的蛋白。
- 平衡阶段没有对蛋白加位置约束。
- 只看 chi1，还没有看 chi2。

## 增强采样

普通 MD 停下后，agent 自动对被标为 `enhanced_sampling` 的侧链做增强采样。方法是 well-tempered metadynamics，直接在 OpenMM 里实现，不需要 PLUMED。每个侧链的 chi1 有一个独立的偏置势，所有侧链在同一次模拟里处理。每条增强采样轨迹从普通 MD 的最终状态开始，默认 2 条，每条 20 ns。

每个残基得到以下结论之一：

| 结论 | 含义 |
|---|---|
| `single_state` | chi1 只有一个稳定状态（主要状态占 95% 以上）。普通 MD 没有被卡住。 |
| `multiple_states` | chi1 有多个状态。普通 MD 停在一个状态，说明它被卡住了，算出的比例不对。 |
| `not_converged` | 两条轨迹的结果不一致，后半段比例还在变化，或者在偏置下翻转仍然太少。可以用 `resume --enhanced-ns` 加长。 |

结果保存在 `enhanced/results.txt` 和 `enhanced/results.json` 里，包括每个残基的状态比例和沿 chi1 的自由能曲线。

## 下一步

1. 在 181L 上验证增强采样：看 Val111 等侧链得到什么结论，并检查偏置参数是否合适。
2. 把 AdaptivePy（Shukla 课题组的自适应采样包）加成普通 MD 阶段的可选模式：每轮从它挑出的帧重新起跑，而不是接着原轨迹跑。这需要：保存包含水的完整坐标；把诊断方法改为马尔可夫状态模型，因为从挑选的帧起跑后，直接数翻转次数和算比例会有偏差。它只能帮助翻转过但次数太少的侧链；从没翻转过的侧链仍然需要增强采样。
3. 加一个 LLM 层：把用户的自然语言要求转换成 `SetupOptions` 和其他设置，用通俗的话汇报结果。
4. 做成异步的 agent 窗口：MD 在后台运行，定期汇报进度。
