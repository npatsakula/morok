---
sidebar_label: 强度削减
---

# 强度削减与晚期重写模式

强度削减将昂贵的操作替换为更廉价的等价操作。这些模式在流水线的后期（阶段 18-19）运行，因为早期 pass 需要看到原始操作结构。例如，`Add(Mul(a, b), c)` 必须保持可见，以便代数化简进行处理，之后才能融合为 `MULACC(a, b, c)`。

Tinygrad 源码：`tinygrad/uop/decompositions.py:438-480`（`get_late_rewrite_patterns`）。
Svod 源码：`schedule/src/rangeify/patterns.rs`（晚期分解组）+ `schedule/src/symbolic/fast_div.rs`。

*本页所有周期估算为现代 x86-64 的近似值。实际延迟因微架构和流水线状态而异。*

所有模式被组合到一个不动点重写 pass（`PM_FINAL`）中，与 `symbolic_simple()`（代数清理）一起运行。

---

## 1. 二的幂优化

影响最大的强度削减。整数除法和常量取模在张量索引中极其常见——步长计算、分块和从平坦索引恢复坐标都会产生它们。

| 模式 | 之前 | 之后 | 周期节省 |
|---------|--------|-------|---------------|
| `x % 2^n` | `idiv` + `imul` + `isub`（约 25 周期） | `and`（1 周期） | 约 24 倍 |
| `x * 2^n` | `imul`（约 3-4 周期） | `shl`（1 周期） | 约 3 倍 |
| `x // 2^n`（无符号） | `idiv`（约 20-40 周期） | `shr`（1 周期） | 约 20-40 倍 |

取模优化之所以有效是因为 `2^n - 1` 是低 n 位的位掩码。示例：`x % 8` = `x & 0b111`。

Tinygrad：`decompositions.py:448-454`。Svod：`rangeify/patterns.rs` 中的 `pm_mod_to_and`、`pm_mul_to_shl`、`pm_div_to_shr`。

:::caution[有符号除法]
对于有符号整数，`x // 2^n` **不是**简单的 `x >> n`。算术右移向负无穷取整，但整数除法向零取整。

修正：`(x + (x < 0 ? 2^n - 1 : 0)) >> n`

为负值添加的偏置 `2^n - 1` 修正了取整方向。这匹配以下恒等式：

```
floor(x / 2^n) = (x + 2^n - 1) >> n    when x < 0
                  x >> n                  when x >= 0
```

Svod 通过范围分析（`VminVmaxProperty`）检查 `vmin >= 0`，当被除数可证明非负时跳过偏置。Tinygrad 使用 dtype 成员（`dtypes.uints`）达到相同目的。

Tinygrad：`decompositions.py:452-454`。Svod：`rangeify/patterns.rs` 中的 `pm_div_to_shr`。
:::

有符号二的幂除法的生成 C 代码：

```c
// Before: x / 8
int result = x / 8;

// After: strength reduction (signed path)
int result = (x + ((x >> 31) & 7)) >> 3;
//           bias for negatives ^^^   ^shift
```

当 `x` 可证明非负时（索引计算中常见），有符号路径被完全消除：

```c
// After: strength reduction (unsigned path, vmin >= 0)
int result = x >> 3;
```

---

## 2. 快速整数除法（Hacker's Delight）

对于非二次幂常量，将 `x / d` 替换为乘移法：`(x * M) >> S`。

### 数学原理

对于正常量 `d` 和值范围 `[0, max_val]`，找到魔术数 `M` 和移位量 `S`，使得：

```
(x * M) >> S == x / d    for all 0 <= x <= max_val
```

**原理**：除以 `d` 等价于乘以 `1/d`。我们将 `1/d` 近似为 `M / 2^S`，其中 `M` 和 `S` 的选择使近似在值范围内是精确的。关键洞察是整数截断使精确表示成为可能——我们只需要 `floor(x * M / 2^S) == floor(x / d)`，不需要实值相等。

### 算法

来自 Hacker's Delight 第 10 章（Tinygrad 的 `magicgu`，`decompositions.py:272-280`）：

1. 计算 `nc = floor((max_val + 1) / d) * d - 1`（临界阈值）
2. 计算 `nbits = bit_length(max_val)`
3. 对 `s` 从 0 到 `2 * nbits`：
   - 如果 `2^s > nc * (d - 1 - (2^s - 1) mod d)`：找到有效移位
   - 计算 `M = ceil((2^s + d - 1 - (2^s - 1) mod d) / d)`
4. 返回 `(M, s)`——最小的有效 `(multiplier, shift)` 对

循环找到产生有效魔术数的最小 `s`。更小的 `s` 意味着更小的 `M`，这对于在窄整数类型中容纳中间乘积 `x * M` 至关重要。

Svod 实现：`schedule/src/symbolic/fast_div.rs` 中的 `magic_unsigned()`。

### 三阶段策略

对应 Tinygrad `decompositions.py:282-300`（`fast_idiv`）：

| 阶段 | 条件 | 变换 | 示例 |
|-------|-----------|-----------|---------|
| 1. 同 dtype | `M * vmax` 在 dtype 范围内 | `(x * M) >> S` | i32 范围内的 `x / 3` |
| 2. 因式二次幂 | `d = 2^k * d'`，`d' > 1` | `(x >> k) / d'` 然后对 `d'` 用魔术数 | `x / 6` 变为 `(x >> 1) / 3` |
| 3. 扩展到 i64 | `x * M` 在 Int32 中溢出 | 转 i64，乘，移位，转回 | 大 `M` 的后备方案 |

因式分解阶段（2）很重要：除以 12（`= 4 * 3`）变为右移 2 后再用魔术数除以 3，这通常在原始 dtype 中就能容纳，而直接对 12 用魔术数会溢出。

对于有符号值，添加修正：`((x * M) >> S) + (x < 0 ? 1 : 0)`。这处理截断趋零语义——没有它，负被除数会向错误方向取整。

### 具体示例

```
x / 7 where x in [0, 255]:
  magic_unsigned(255, 7) → M = 293, S = 11

  Verify: (100 * 293) >> 11 = 29300 >> 11 = 14 = floor(100 / 7)
  Verify: (  7 * 293) >> 11 =  2051 >> 11 =  1 = floor(  7 / 7)
  Verify: (255 * 293) >> 11 = 74715 >> 11 = 36 = floor(255 / 7)

  Generated: (x * 293) >> 11  instead of  x / 7
  Cost: 1 imul + 1 shr (~4-5 cycles) vs 1 idiv (~20-40 cycles)
```

### 生成的 LLVM IR

```llvm
; Before: x / 7
%result = sdiv i32 %x, 7

; After: fast integer division (unsigned path)
%mul = mul i32 %x, 293
%result = lshr i32 %mul, 11
```

---

## 3. 浮点除法转乘法

`x / c` 变为 `x * (1/c)`，其中 `c` 为浮点常量。

浮点乘法为 1-2 周期（全流水线），而浮点除法为 10-20 周期（大多数硬件上不流水线）。这是常见模式 5-10 倍的直接加速。

**守卫**：
- 如果 `c == 0.0` 则跳过——必须保留除以零以保持 IEEE 754 语义（`x / 0.0` 产生 `+/-inf` 或 `NaN`）
- 如果 `1/c` 不是有限值则跳过（溢出到 `inf` 意味着 `c` 太小）
- 仅用于浮点类型

Tinygrad：`decompositions.py:477-479`（基于 FDIV 的后端将 `RECIP` 作为 `1/x` 发出）。Svod：`rangeify/patterns.rs` 中的 `pm_fdiv_to_mul`。

```c
// Before
float result = x / 3.14159f;

// After
float result = x * 0.31831f;  // 1/pi
```

---

## 4. FMA 融合（融合乘加）

`a * b + c` 变为 `MULACC(a, b, c)`。

这映射到硬件 FMA 指令（x86 AVX 的 `vfmadd`、ARM NEON 的 `fmadd`、CUDA 的 `fma.rn`）。用一条指令替代两条，只有一次取整步骤而不是两次——使 FMA 比分开的乘法 + 加法更快且更精确。

**为何晚期应用**：早期 pass 需要看到 `Add(Mul(a, b), c)` 结构进行代数化简。如果过早融合，`(x*2 + x*3)` 这样的模式无法化简为 `x*5`，因为 `Mul` 节点会被埋入 MULACC 内。

**移位-加法融合**：`(x << n) + c` 同样会融合为 `MULACC(x, 2^n, c)`，捕获在同一不动点 pass 中 MUL 转 SHL 先运行的情况。当渲染器同时支持 `MulAcc` 和 `Shl` 时，Svod 会把 `pm_shl_add_to_mulacc` 与 `pm_fma_decomposition` 一起加入。

**守卫**：仅当三个操作数（`a`、`b`、`c`）共享相同的浮点 dtype 时匹配。整数 FMA 不做融合，因为硬件 FMA 指令仅支持浮点。

Tinygrad：`decompositions.py:472-475`。Svod：`rangeify/patterns.rs` 中的 `pm_fma_decomposition`。

---

## 5. 取反提取

`x * -1` 变为 `NEG(x)`。

NEG 是单条指令（浮点通过 `xorps` 翻转符号位，整数通过 `neg` 取反）。乘以 -1 不必要地占用乘法器流水线 3-4 个周期。

仅当后端支持 `NEG` 作为原生操作时触发。Tinygrad：`decompositions.py:458-459`。Svod：`pm_neg_from_mul`。

---

## 6. 比较取反

整数上取反和复合比较的晚期重写。这些模式简化了早期 pass 中布尔逻辑优化产生的指令序列。

| 模式 | 之前 | 之后 | 节省 |
|---------|--------|-------|---------|
| `!(x < c)` | NOT + CMP | `(c-1) < x` | 消除 NOT |
| `!(c < x)` | NOT + CMP | `x < (c+1)` | 消除 NOT |
| `(c1 < x) & (x < c2)`，`c2 == c1+2` | 2 CMP + AND | `x == (c1+1)` | 消除 2 个操作 |
| `x * -1 < c` | MUL + CMP | `-c < x` | 消除 MUL |
| `x * -1 < y * c` | 2 MUL + CMP | `y * (-c) < x` | 消除 1 个 MUL |

范围压缩（第 3 行）特别有价值。当开区间 `(c1, c2)` 恰好包含一个整数值时，两次比较和一个逻辑 AND 折叠为单次相等检查。这在分块索引计算中自然出现，范围变量恰好选择一个分块。

:::caution[常量中的整数溢出]
取反模式防范溢出：`!(x < c)` 变为 `(c-1) < x` 仅当 `c-1` 不下溢时，`!(c < x)` 变为 `x < (c+1)` 仅当 `c+1` 不上溢时。两者都使用 `checked_sub` / `checked_add`，溢出时返回 `None`（不做变换）。
:::

Tinygrad：`decompositions.py:461-470`。Svod：`rangeify/patterns.rs` 中的 `pm_comparison_negations`。

---

## 7. 德摩根定律（晚期）

```
!a & !b  -->  !(a | b)
!a | !b  -->  !(a & b)
```

这出现在流水线的*两个*位置：

1. **早期**（阶段 4-5）：`schedule/src/symbolic/patterns.rs` 中的 `boolean_dsl_patterns()`，是完整 `symbolic()` 匹配器的一部分。捕获原始表达式结构中的德摩根机会。

2. **晚期**（阶段 18-19）：`symbolic_simple()` 包含布尔模式，与 `PM_FINAL` 中的强度削减模式一起运行。这捕获比较取反模式创建的新德摩根机会——例如，`!(x < 3)` 和 `!(x < 7)` 被重写为 `2 < x` 和 `6 < x` 后，组合它们的 AND/OR 可能有新的 NOT 消除机会。

Svod：`schedule/src/symbolic/patterns.rs` 中的 `boolean_dsl_patterns()`。

---

## 8. ERF 分解

`erf(x)` 被替换为多项式近似（Abramowitz & Stegun 7.1.26）：

```
erf(x) = sign(x) * (1 - t * P(t) * exp(-x^2))
where t = 1 / (1 + 0.3275911 * |x|)
      P(t) = Horner(t, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592])
```

**原因**：`@llvm.erf` 是一个库调用内置函数（需要 libm 链接），不是原生硬件指令。LLVM JIT 后端不链接 libm，所以 `erf` 必须在代码生成之前分解。Tinygrad 在张量层面分解 `erf`（`elementwise.py`），所以它永远不会到达渲染器；Svod 将 `Erf` 保留为 UOp 直到此晚期 pass。

最大误差：约 1.5e-7（对 float32 ML 工作负载足够）。

Svod：`rangeify/patterns.rs` 中的 `pm_erf_decomposition`。

---

## 模式组合：每个模式何时运行

所有强度削减模式被组合为单个 `PM_FINAL` 匹配器，作为不动点图重写运行：

```
PM_FINAL = pm_commit_weak() + pm_cast_weak() + pm_decomp
         + renderer.extra_matcher()   -- optional, per-backend
         + pm_split_ends()
```

其中 `pm_decomp` 串联了早期分解（它本身从 `symbolic_simple()` 开始）、超越函数分解，以及 `get_late_rewrite_patterns()`：

```
Stage 18-19（PM_FINAL 不动点重写）：
  symbolic_simple()              -- 代数清理（恒等式、常量折叠）
  + pm_erf_decomposition         -- erf(x) -> 多项式近似
  + pm_mod_to_and                -- x % 2^n -> x & (2^n-1)
  + pm_demorgan                  -- !a & !b -> !(a | b)
  + pm_mul_to_shl                -- x * 2^n -> x << n
  + pm_div_to_shr                -- x // 2^n -> x >> n
  + fast_division_patterns       -- x // d -> (x * M) >> S
  + pm_neg_from_mul              -- x * -1 -> NEG(x)
  + pm_comparison_negations      -- !(x<c) -> (c-1)<x 等
  + pm_fma_decomposition         -- a*b+c -> MULACC(a,b,c)
  + pm_shl_add_to_mulacc         -- (x<<n)+c -> MULACC(x, 2^n, c)
  + pm_fdiv_to_mul               -- x / c -> x * (1/c)
```

由于重写器运行到不动点，模式可以相互馈送。例如：

1. `pm_mul_to_shl` 将 `x * 4` 转换为 `x << 2`
2. 下一次迭代中，`pm_fma_decomposition` 将 `(x << 2) + c` 融合为 `MULACC(x, 4, c)`
3. `symbolic_simple()` 清理变换创建的任何恒等式

`pm_mod_to_and` 之后的每一项都是有条件的：只有当渲染器声明支持该模式所产生的操作时，`get_late_rewrite_patterns()` 才会加入它（移位重写需要 `Shl`，FMA 融合需要 `MulAcc`，倒数重写需要 `Fdiv`）。`pm_split_ends` 在同一个不动点内运行，为线性化器把多范围 END 拆成嵌套的单范围 END；最后由 `pm_remove_invalid()` 重写清除残留的 Invalid 标记。

交叉引用：[代码生成流水线概览](../codegen/overview.md)，完整阶段列表。
