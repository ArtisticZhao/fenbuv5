# 带时滞反应-扩散系统 Hopf 分岔仿真程序

本项目用于围绕带时滞参数的反应-扩散系统做两步数值实验：

1. 扫描并拟合 Hopf 分支，寻找候选分岔参数；
2. 以候选 Hopf 点的 `epsilon` 和空间模态 `n` 为参数，运行时滞反应-扩散数值仿真。

项目不是通用 PDE 求解器，主要工作流是“先定位 Hopf 候选点，再在分岔点附近仿真”。

## 环境

默认使用 conda 环境 `note`：

```bash
conda activate note
```

主要依赖包括：

- `numpy`
- `scipy`
- `matplotlib`
- `fipy`

如果不进入环境，也可以用：

```bash
conda run -n note python main.py --help
```

## 快速运行

使用示例配置 `test_b4.json` 运行完整流程：

```bash
conda run -n note python main.py --config test_b4.json
```

这会：

1. 根据 `model` 参数生成 Hopf 搜索所需的理论参数；
2. 在 `hopf.n_list` 和 `hopf.k_list` 指定的分支上扫描候选 Hopf 点；
3. 保存 Hopf 分支图；
4. 如果 `simulation.enabled=true`，对选中的 Hopf 点运行仿真；
5. 将结果保存到 `output.dir` 指定目录。

只寻找 Hopf 候选点，不运行仿真：

```bash
conda run -n note python main.py --config test_b4.json --simulate false
```

只仿真排序后的第 0 个和第 2 个 Hopf 候选点：

```bash
conda run -n note python main.py --config test_b4.json --hopf-indices 0,2
```

改输出目录：

```bash
conda run -n note python main.py --config test_b4.json --output-dir output1
```

## 生成配置模板

可以先生成默认模板，再按需要修改：

```bash
conda run -n note python main.py --write-template config_template.json
```

配置文件是 JSON 格式，顶层包含：

- `model`：反应-扩散-时滞模型与数值网格参数；
- `hopf`：Hopf 分支扫描与局部多项式拟合参数；
- `simulation`：是否仿真、仿真哪些 Hopf 候选点、初值扰动幅值；
- `plots`：是否显示或保存图；
- `output`：输出目录和数据保存开关。

## 参数说明

### `model`

这些参数会同时影响 Hopf 理论计算和最终数值仿真。

| 参数 | 含义 |
| --- | --- |
| `d1` | 扩散项系数。 |
| `d2` | 时滞梯度驱动项系数。 |
| `r1` | 反应项系数。Hopf 搜索中使用 `fu = r1 * (1 - 2 * u_bar)`。 |
| `u_bar` | 平衡态或背景常数值。 |
| `tau` | 时滞核的最大时滞。必须为正数。 |
| `epsilon` | 仿真默认时滞核参数；完整工作流中会被 Hopf 候选点的 `epsilon` 覆盖。需满足 `0 < epsilon < tau`。 |
| `L` | 一维空间区间长度。Hopf 理论按 `L = l * pi` 使用 `l = L / pi`。 |
| `Nx` | 空间网格数，至少为 3。 |
| `dt` | 时间步长。`tau / dt` 必须接近整数，否则时滞历史无法对齐。 |
| `Tend` | 仿真终止时间。 |
| `sweep_tol` | FiPy 每步 sweep 的残差容忍度。 |
| `stagnation_tol` | 残差停滞判断阈值。 |
| `max_sweeps` | 每个时间步最多 sweep 次数。 |
| `cfl_warning_threshold` | 对流 CFL 警告阈值。 |

常用调整：

- 想提高空间精度：增大 `Nx`；
- 想提高时间精度或改善时滞核离散：减小 `dt`，同时确保 `tau / dt` 是整数；
- 想缩短测试时间：减小 `Tend`、`Nx` 或 Hopf 扫描点数。

### `hopf`

这些参数控制 Hopf 候选点搜索。程序先在 `epsilon`-`omega` 网格上寻找 `G(omega, epsilon)=0` 的根分支，再计算每条分支上的 `S` 值，并用局部多项式拟合 `S=0` 附近的候选点。

| 参数 | 含义 |
| --- | --- |
| `n_list` | 要扫描的空间模态列表。对应 `mu = n^2 / l^2`。 |
| `k_list` | 要扫描的相位分支编号列表。 |
| `eps_min` | `epsilon` 搜索下界。 |
| `eps_max_margin` | `epsilon` 搜索上界为 `tau - eps_max_margin`。 |
| `omega_min` | `omega` 搜索下界。 |
| `omega_max` | `omega` 搜索上界。 |
| `num_eps` | `epsilon` 网格点数。越大越精细，也越慢。 |
| `num_omega` | `omega` 网格点数。越大越容易捕捉根，也越慢。 |
| `min_hopf_omega` | 小于该值的根会被过滤。 |
| `max_roots_per_eps` | 每个 `epsilon` 截面最多保留的根数量。 |
| `max_total_branches` | 每个 `n` 最多追踪的根分支数量。 |
| `branch_gap_abs` | 分支追踪时允许的绝对频率跳变。 |
| `branch_gap_rel` | 分支追踪时允许的相对频率跳变。 |
| `branch_max_fill` | 对短缺口做线性填补的最大连续缺失点数。 |
| `s_tol` | 识别 `S` 近零区域的阈值。 |
| `polynomial_degree` | 局部多项式拟合阶数。 |
| `polynomial_window` | 局部拟合窗口点数。 |
| `polynomial_plot_points` | 绘制局部拟合曲线时的采样点数。 |
| `duplicate_eps_tol` | 去重时认为两个候选 `epsilon` 相同的容差。 |

候选 Hopf 点按 `abs(S)` 从小到大排序，并在终端输出类似：

```text
[0] epsilon=..., omega=..., n=..., k=..., mu=..., S=...
```

`simulation.hopf_indices` 使用的就是这里的排序编号。

### `simulation`

| 参数 | 含义 |
| --- | --- |
| `enabled` | 是否在 Hopf 搜索后继续运行仿真。 |
| `hopf_indices` | 要仿真的候选点编号。`null` 表示仿真全部候选点。 |
| `amplitude` | 初值扰动幅值。初值形式为 `u_bar + amplitude * cos(n*pi*x/L)`。 |
| `default_initial_mode` | 保留字段；当前自动工作流使用 Hopf 候选点的 `n` 作为初值模态。 |

如果只想快速确认 Hopf 搜索结果，建议先设：

```json
"simulation": {
  "enabled": false
}
```

确认候选点编号后，再指定少量点仿真：

```json
"simulation": {
  "enabled": true,
  "hopf_indices": [0],
  "amplitude": 0.01
}
```

### `plots`

| 参数 | 含义 |
| --- | --- |
| `show` | 是否弹出 matplotlib 图窗。服务器或批量运行时建议为 `false`。 |
| `save` | 是否保存图片。 |
| `live` | 是否在仿真过程中实时更新图。要求 `show=true`。 |
| `hopf` | 是否绘制 Hopf 分支图。 |
| `final` | 是否绘制每次仿真的最终结果图。 |

命令行也可临时覆盖：

```bash
conda run -n note python main.py --config test_b4.json --show-plots true --live-plots true
```

### `output`

| 参数 | 含义 |
| --- | --- |
| `dir` | 输出目录。 |
| `save_data` | 是否保存 `.npz` 仿真数据。 |
| `save_resolved_config` | 是否保存实际生效的配置。 |

## 输出文件

假设 `output.dir` 为 `output`，常见输出包括：

| 文件 | 内容 |
| --- | --- |
| `output/resolved_config.json` | 合并配置文件与命令行覆盖后的最终配置。 |
| `output/hopf_root_branches.png` | `G(omega, epsilon)=0` 根分支图。 |
| `output/hopf_s_values.png` | 各分支的 `S(epsilon)` 曲线及局部拟合。 |
| `output/hopf_candidates.png` | 候选 Hopf 点散点图。 |
| `output/run_001_result.npz` | 第 1 次仿真的数组数据和参数元数据。 |
| `output/run_001_final.png` | 第 1 次仿真的最终状态、时空图、残差和探针时间序列。 |

`.npz` 数据中包含：

- `x`, `t`, `u`：空间网格、时间网格和解；
- `residuals`, `sweep_counts`：每步求解残差和 sweep 次数；
- `min_u`, `max_u`, `mean_u`, `cfl`：诊断量；
- `h_weights`, `s_vec`：离散时滞核；
- `hopf_epsilon`, `hopf_omega`, `hopf_n`, `hopf_k`, `hopf_mu`, `hopf_s_value`：对应 Hopf 候选点信息。

## 示例配置

`test_b4.json` 是当前可直接运行的示例。它会扫描：

```json
"n_list": [1, 3, 5, 7],
"k_list": [0],
"num_eps": 220,
"num_omega": 12000
```

并在 `simulation.hopf_indices` 为 `null` 时仿真所有候选点。若候选点较多，建议改成指定编号，例如：

```json
"hopf_indices": [0]
```

## 常见问题

### `tau must be an integer multiple of dt`

时滞历史队列需要与时间步对齐。修改 `dt`，使 `tau / dt` 为整数。例如 `tau=2.0` 时，`dt=0.01`、`0.02`、`0.05` 都可以。

### 没有找到 Hopf 候选点

可以尝试：

- 扩大 `omega_max`；
- 增大 `num_eps` 或 `num_omega`；
- 调整 `n_list`、`k_list`；
- 检查 `d1`、`d2`、`r1`、`u_bar`、`tau` 是否对应当前理论模型。

### 仿真太慢

优先尝试：

- 先运行 `--simulate false` 只看 Hopf 候选点；
- 减小 `Tend`；
- 减小 `Nx`；
- 只设置少量 `hopf_indices`；
- 降低 `num_eps` 和 `num_omega` 做粗扫，确认区域后再精扫。

### 图窗不显示

默认配置通常适合批量保存图片，不弹窗。如果需要图窗：

```bash
conda run -n note python main.py --config test_b4.json --show-plots true
```

实时图还需要：

```bash
conda run -n note python main.py --config test_b4.json --show-plots true --live-plots true
```
