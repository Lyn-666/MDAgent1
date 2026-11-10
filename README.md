# MDAgent1

一个 MD agent：用户给出蛋白 PDB 和关心的区域，agent 自动搭体系、跑 MD，找出这个区域里没有采样好的侧链。agent 自己决定跑多久：每轮结束后诊断，再决定续跑还是停止。之后的版本会对卡住的侧链自动做增强采样。

## 工作流程

1. **读入目标区域。** 可以直接给出残基（如 `A:99-120`），或者给出"某个配体周围 N Å 内的残基"。
2. **搭体系。** 用 PDBFixer 去掉配体和晶体水，补缺失原子，按 pH 加氢，再加水盒子和 0.15 M NaCl。力场为 Amber14 + TIP3P。
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

## 用法

```bash
git clone https://github.com/Lyn-666/MDAgent1.git && cd MDAgent1
conda create -n mdagent -c conda-forge python=3.11 openmm pdbfixer mdtraj -y
conda activate mdagent
pip install -e ".[test]"

# 开始。agent 自己决定跑多久，每条轨迹最多 20 ns。
python -m mdagent run --pdb examples/181L.pdb --around-ligand BNZ --out runs/181L

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
| `run.json` | 设置、目标残基、每轮的长度。续跑时从这里读取。 |
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
- 质子化使用 PDBFixer 的默认规则，不预测蛋白内部的 pKa 变化。
- 平衡阶段没有对蛋白加位置约束。
- 只看 chi1，还没有看 chi2。

## 下一步

1. 对 `enhanced_sampling` 的侧链的 chi1 做增强采样（通过 openmm-plumed 使用 OPES），输出各状态的比例和自由能曲线。
2. 加一个 LLM 层：把用户的自然语言目标转换成残基列表，用通俗的话汇报诊断结果，并在执行增强采样之前请用户确认。
3. 做成异步的 agent 窗口：MD 在后台运行，定期汇报进度。
