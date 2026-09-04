---
sidebar_label: 索引算术
---

# 索引算术

张量编译器将大部分优化预算花在索引算术上。一个形状为 `[H, W]` 的 `tensor[i, j]` 访问变成 `i * W + j`。经过分块、向量化和循环变换后，这些表达式会积累嵌套的除法和取模。化简它们至关重要——一个多余的 `idiv` 需要 20-40 个周期，而等价的移位只需 1 个周期（近似值，现代 x86-64）。

本页文档记录了化简索引表达式的模式。这些不是传统意义上的优化——它们是使张量索引高效工作的代数。

**关键概念——值范围分析**：每个 UOp 跟踪它在运行时可取的最小值（`vmin`）和最大值（`vmax`），在节点构造时从输入的边界急切地计算。许多索引模式使用这些边界在编译期证明化简的正确性（例如"`x` 始终在 `[0, N)` 内"使 `x % N` → `x` 成立）。

这些模式在[代码生成流水线](../codegen/overview.md)的多个阶段运行：
- **阶段 4**（rangeify 期间的初始符号化简）
- **阶段 8**（后优化符号化简）
- **阶段 15**（通过 `pm_lower_index_dtype` 进行索引 dtype 降级）
- **阶段 16**（索引后符号化简）

Svod 源码：`schedule/src/symbolic/patterns.rs`、`schedule/src/symbolic/index_lowering.rs`

Tinygrad 源码：`tinygrad/uop/divandmod.py`、`tinygrad/uop/symbolic.py`

---

## 1. Div-Mod 恒等式

整数除法的基本定理：

$$
x = \lfloor x / n \rfloor \cdot n + (x \bmod n)
$$

模式集中有五个变体利用此恒等式：

| # | 模式 | 条件 | 名称 |
|---|---------|-----------|------|
| 1 | `x%n + (x//n)*n` -> `x` | -- | 核心恒等式 |
| 2 | `((x//a) % c) + (x//b)*c` -> `x//a` | `a*c == b` | 组合除数 |
| 3 | `(x%c1)*c2 + (x//c1)*c3` -> `x*c2` | `c1*c2 == c3` | 缩放 |
| 4 | `y + (x%n) + (x//n)*n` -> `y + x` | -- | 三项 |
| 5 | `(a//c1 + c2) // c3` -> `(a + c1*c2) // (c1*c3)` | `c1>0, c3>0` | 嵌套除法 |

**#1 的证明。** 由除法算法，对于整数 `x` 和 `n > 0`，存在唯一整数 `q` 和 `r` 使得 `x = q*n + r`，其中 `0 <= r < n`。根据定义，`q = x // n`，`r = x % n`。代入：`(x % n) + (x // n) * n = r + q*n = x`。证毕。

**为什么 #2-#5 是推论。**

变体 #2 组合两层除法。由于 `b = a*c`，我们有 `x // b = (x // a) // c`。在内层应用核心恒等式：`((x//a) % c) + ((x//a) // c) * c = x // a`。但 `(x//a) // c = x // (a*c) = x // b`，得到该模式。

变体 #3 将核心恒等式两边乘以 `c2`。从 `x = (x % c1) + (x // c1) * c1`，乘以 `c2`：`x * c2 = (x % c1) * c2 + (x // c1) * c1 * c2`。由于 `c1 * c2 = c3`，得 `(x % c1) * c2 + (x // c1) * c3 = x * c2`。

变体 #4 将独立项 `y` 加到 #1 两边。

变体 #5 展平嵌套地板除法。给定 `(a // c1 + c2) // c3`，将 `c2` 乘以内层除数得到等价的单层除法：`(a + c1*c2) // (c1*c3)`。当 `a >= 0` 且 `c2 >= 0`（或都非正）时成立，确保地板除法语义被保持。

所有五个模式在重复变量名上使用 `Arc::ptr_eq` 检查（例如 `x` 出现两次意味着两者必须是同一个 hash consing 节点）。

### 实现

```rust
// From schedule/src/symbolic/patterns.rs — div_mod_recombine_dsl_patterns()

// #1: x%n + (x//n)*n -> x
Add[Mod(x, n), Mul[Idiv(x, n), n]] => x,

// #2: ((x//a) % c) + (x // b) * c -> x // a  when a*c == b
Add[Mod(Idiv(x, a @const(a_val)), c @const(c_val)), Mul[Idiv(x, _b @const(b_val)), c]]
    => { /* guard: a_val * c_val == b_val */ },

// #5: (a//c1 + c2) // c3 -> (a + c1*c2) // (c1*c3)
Idiv(Add[Idiv(a, c1 @const(c1_val)), _c2 @const(c2_val)], _c3 @const(c3_val))
    => { /* guard: c1>0, c3>0, same-sign */ },
```

---

## 2. 基于范围的 Mod/Div

值范围分析（`vmin`/`vmax`）使纯语法模式匹配看不到的化简成为可能。每个 UOp 携带构造时计算的缓存边界。

| 模式 | 守卫 | 示例 |
|---------|-------|---------|
| `x % n` -> `x` | `0 <= vmin(x)` 且 `vmax(x) < n` | `RANGE(3) % 3` -> `RANGE(3)` |
| `(a*m + b) % n` -> `b % n` | `m == n` | `(row*512 + col) % 512` -> `col % 512` |
| `(a*m + b) / n` -> `a + b/n` | `m == n` 且 `0 <= b < n` | `(row*512 + col) / 512` -> `row` |
| `x / n` -> `k` | 所有值在桶 `[k*n, (k+1)*n)` 内 | `RANGE(3) / 3` -> `0` |
| `(x + c) // d` -> `x // d` | `max_remainder + c < d` | `(R*4 + 1) // 8` -> `R*4 // 8` |
| `(x + c) // d` -> `(x + c%d) // d + c//d` | `c >= d` | `(x + 70) // 8` -> `(x + 6) // 8 + 8` |

第一个模式是主力。范围分割后，`RANGE(n)` 产生 `[0, n)` 内的值，所以 `RANGE(n) % n` 平凡化简为 `RANGE(n)`。这条规则消除了分块产生的大部分取模。

第五个模式（小常量）对范围 `[vmin, vmax]` 内的最大余数使用紧边界。如果范围跨度少于 `d` 个值且加上 `c` 不会跨越桶边界，则常量是无用的。

第六个模式（大偏移拆分）规范化大于除数的偏移量。这为下一次重写迭代暴露了小常量模式。

:::caution
`(a*m + b) / n` -> `a + b/n` 模式要求 `0 <= b < n`。没有范围检查，负余数会因截断趋零语义产生错误商。实现显式检查 `vmin(b) >= 0 && vmax(b) < n`。
:::

---

## 3. `fold_divmod_general` 算法

Index dtype 上 `Idiv` 和 `Mod` 的兜底算法。按优先级顺序实现 Tinygrad `divandmod.py:8-93` 的全部 8 条规则，包括递归的 `nest_div_by_smallest_factor`。每条规则按顺序尝试；第一个匹配的获胜。

入口：当 `Idiv(x, y)` 或 `Mod(x, y)` 的 `dtype == Index` 时，模式委托给 `fold_divmod_general(op, x, y)`。

### 规则 1——cancel_divmod

如果整个范围 `[x_min, x_max]` 在 `(x, y)` 的所有角落组合上映射到同一商，结果就是那个常量。

**守卫**：`y_min * y_max > 0`（分母不跨零），且四个角落商 `x_min/y_min`、`x_min/y_max`、`x_max/y_min`、`x_max/y_max` 全部相等。

**行为**：对 `Idiv`，返回常量商。对 `Mod`，返回 `x - q*y`。

**示例**：`RANGE(3) // 3` -> `0`。值 0、1、2 全部除以 3 得 0。

### 规则 2——remove_nested_mod

`(a%4 + b) % 2` -> `(a + b) % 2`，当 `2 | 4` 时。外层取模整除内层，所以内层取模是冗余的。

**守卫**：`op == Mod`，`x_min >= 0`，且对每个为 `Mod(inner_x, inner_y)` 的项，分母 `y` 整除 `inner_y`。

**行为**：剥离模数为外层模数倍数的内层 `Mod` 操作，然后重新应用 `Mod`。

**示例**：`(RANGE(8) % 4 + RANGE(2)) % 2` -> `(RANGE(8) + RANGE(2)) % 2`

### 规则 3——fold_binary_numerator

当单个非常量项恰好有 2 个值（`vmax - vmin == 1`）时，结果是线性插值：`(y2 - y1) * (v - v_min) + y1`。

**守卫**：分解后恰好一个非常量项，且该项的范围恰好跨 2 个值。

**行为**：在两个端点求值 div/mod，构造它们之间的线性映射。这完全避免了除法。

**示例**：对 `(v * 3 + 2) % 5`，其中 `v` 在 `{0, 1}` 内：
- `v=0`：`(0 + 2) % 5 = 2`
- `v=1`：`(3 + 2) % 5 = 0`
- 结果：`(0 - 2) * (v - 0) + 2 = -2*v + 2`

### 规则 4——fold_divmod_congruence

对每个项 `f_i * v_i`，按绝对值计算最近的残余 `r_i = min(f_i % c, f_i % c - c)`。如果残余和保持在 `c` 的一个地板除法桶内，mod/div 可以化简。这是模算术优化。

**守卫**：`x_min >= 0`，常量分母 `c > 0`，且 `rem_min // c == rem_max // c`（所有残余和值落在同一桶内）。

**行为**：将每个因子替换为其模 `c` 的残余。对 `Mod`，返回残余和（按桶偏移调整）。对 `Idiv`，返回商系数和。

**示例**：`(r*8 + v) % 7` -> `(r + v) % 7`，因为 `8 = 1 (mod 7)`，所以 `8` 的残余为 `1`。

### 规则 5——gcd_with_remainder

计算所有加法项和分母的符号 GCD。如果 GCD > 1，提取出来：`(g*a + g*b) // (g*c)` -> `(a + b) // c`（`Mod` 时残余按比例缩放回去）。

**守卫**：`x_min >= 0`，常量分母，GCD > 1，且化简后的分子 `vmin >= 0`。

**行为**：将分子项和分母都除以 GCD，递归地使更简单的模式得以触发。

**示例**：`(6*a + 4*b) // 8`，`GCD(6, 4, 8) = 2` -> `(3*a + 2*b) // 4`

### 规则 6——divide_by_gcd

规则 5 的变量分母版本。计算 `GCD(所有项..., y)` 包括分子和分母，然后两边除以。不同于规则 5，分母不需要是常量。

**守卫**：GCD 非平凡（不为 1），且 `x` 和 `y` 都能被 GCD 整除。

**示例**：`(4*a) // (2*b)` -> `(2*a) // b`

### 规则 7——factor_remainder

最后手段。将项分为可整除（商）和余数。

**守卫**：`x_min >= 0` 且 `y_min >= 0`，且至少一项能整除 `y`。

**行为**：对 `Idiv`：`quo_sum + rem // y`。对 `Mod`：`rem % y`（常量 `y` 时进行系数化简）。

**示例**：`(8*a + 3*b) // 8` -> `a + (3*b) // 8`

### 规则 8——nest_div_by_smallest_factor

常量除数的递归分解。找到除数和任何项的系数之间共享的最小因子，两边除以它，然后递归。

**守卫**：`x_min >= 0`，常量 `y > 1`，且至少一个非常量项的因子 `f > 1` 满足 `y % f == 0`。

**行为**：选择 `div = min(|f|)` 在合格因子中，将 `x // y` 重写为 `(x // div) // (y / div)`。每步减小 `y`，收敛到规则 1-7。

**示例**：`(6*a + 4*b) // 12` -> `((6*a + 4*b) // 2) // 6` -> `(3*a + 2*b) // 6` -> `(3*a + 2*b) // 6`（然后规则 7 完成）。

Tinygrad：`divandmod.py:62-67`。Svod：`fold_divmod_general` 中的 `nest_div_by_smallest_factor`。

:::caution
规则 5-8 要求分子非负（`x_min >= 0`）。负操作数的地板除法有不同的取整语义（Python/Tinygrad 中向负无穷取整，硬件中向零取整）。实现对负范围返回 `None`，交由后续 pass 处理。
:::

---

## 4. 高级除法模式

`fold_divmod_general` 之外处理额外情况的独立模式：

| 模式 | 守卫 | 来源 |
|---------|-------|--------|
| `(a // b) // c` -> `a // (b*c)` | `b != 0, c != 0` | `advanced_division_dsl_patterns` |
| `expr // divisor` -> 精确商 | `expr` 可精确整除 | `advanced_division_dsl_patterns` |
| `(a + b) % c` 系数化简 | `a` 或 `b` 有可被 `c` 整除的因子 | `advanced_division_dsl_patterns` |
| `(a + b) // c` -> `a//c + b//c` | 两边都能整除 | `advanced_division_dsl_patterns` |
| `(a - b) // c` -> `a//c - b//c` | 两边都能整除 | `advanced_division_dsl_patterns` |
| `c * (a + b)` -> `c*a + c*b` | `c` 是常量 | `advanced_division_dsl_patterns` |

嵌套除法折叠 `(a // b) // c` -> `a // (b*c)` 在分块后尤其重要——将范围拆分为外/内组件会创建两层除法，应折叠为一层。

精确除法模式使用 `divides()`，检查每个加法项的常量因子是否能被除数整除。成功时，`Idiv` 被完全消除——不生成除法指令。

系数化简模式将 `(r*8 + v) % 7` 转换为 `(r*1 + v) % 7 = (r + v) % 7`，将每个因子对除数取模化简。当没有因子是模数的整倍数但残余更小时触发。

---

## 5. 索引 dtype 降级（三阶段级联）

Tinygrad：`ops.py:1291-1313`。Svod：`schedule/src/symbolic/index_lowering.rs`。

抽象 `Index` 类型不携带宽度——它表示"此索引需要的任何整数宽度"。降级 pass 根据值边界将 `Index` 转换为具体的 `i32` 或 `i64`。

### 第一阶段——创建包装器（叶节点）

`Index` dtype 的叶节点被替换为包装在转换回 `Index` 的 cast 中的具体等价物：

| 输入 | 输出 |
|-------|--------|
| `CONST(Index)` | `CONST(concrete).cast(Index)` |
| `DEFINE_VAR(Index)` | `DEFINE_VAR(concrete).cast(Index)` |
| `VCONST(Vector<Index, N>)` | `VCONST(Vector<concrete, N>).cast(Vector<Index, N>)` |

### 第二阶段——向上处理包装值

二元操作、控制流和结构节点通过 `.cast(Index)` 包装器传播具体类型：

| 输入 | 输出 |
|-------|--------|
| `Binary(x.cast(Index), y.cast(Index))` | `Binary(x.cast(dt), y.cast(dt)).cast(result_dtype)` |
| `WHERE(cond, x.cast(Index), y.cast(Index))` | `WHERE(cond, x.cast(dt), y.cast(dt)).cast(Index)` |
| `RANGE(end.cast(Index))` | `RANGE(end, end.dtype).cast(Index)` |
| `SPECIAL(end.cast(Index))` | `SPECIAL(end, i32).cast(Index)` |
| `VECTORIZE(e0.cast(Index), ...)` | `VECTORIZE(e0.cast(dt), ...).cast(Vector<Index, N>)` |
| `BIND(var.cast(Index), val.cast(Index))` | `var.cast(dt).bind(val.cast(dt)).cast(Index)` |

`dt` 计算为 `least_upper_dtype(select_dtype(result), x.dtype, y.dtype)`——任何操作数或结果所需的最宽类型。

### 第三阶段——在终端节点剥离包装器

终端节点消费索引并丢弃 `Index` 包装器：

| 输入 | 输出 |
|-------|--------|
| `INDEX(buf, idx.cast(Index))` | `INDEX(buf, idx)` |
| `INDEX(buf, WHERE(cond, idx, Invalid))` | `INDEX(buf, idx, gate=cond)` |
| `SINK(sources with .cast(Index))` | `SINK(unwrapped sources)` |
| `END(computation.cast(Index))` | `END(unwrapped computation)` |

`WHERE(cond, idx, Invalid)` -> `gate=cond` 的变换很重要：它将有效性条件从索引表达式中提取到 `INDEX` 节点的门控字段，代码生成后端使用它来发出谓词化加载。

### `select_dtype()`

如果 UOp 的值边界在 `[-2^31, 2^31 - 1]` 内则返回 `i32`，否则返回 `i64`。大多数张量索引适合 `i32`——即使 20 亿元素的张量的平坦索引也适合。`i64` 路径用于非常大的张量或累积偏移。

---

## 6. 交换律规范化

```rust
// For Index dtype ONLY:
op(a, b) -> op(b, a)   when b.id < a.id
```

确保交换操作根据 UOp 唯一 ID 具有确定性的操作数顺序。应用于：`Add`、`Mul`、`Max`、`Eq`、`Ne`、`And`、`Or`、`Xor`。

**为何仅限 Index**：没有规范化，`R1*8000 + R2*16` 和 `R2*16 + R1*8000` 在 hash consing 后是不同节点，破坏 `expand_vector_index` 中的分组。展开器需要识别向量各 lane 间相同的索引模式，非规范顺序会使之失败。

**为何不应用于非 Index 类型**：对浮点/整数算术应用规范化会重排 VECTORIZE 元素，破坏后续 pass 中的向量数学合并。Tinygrad 做了相同的选择（`symbolic.py:178-182`）。

:::caution
规范化与重写引擎的不动点迭代有交互。如果两个模式在操作数顺序上不一致（一个规范化，另一个产生非规范输出），引擎可能振荡。所有索引生成模式必须遵守规范顺序，否则 1000 次迭代安全限制会触发。
:::

---

## 完整示例

考虑形状为 `[4, 8]` 的 `tensor[i, j]`，以对 32 个元素的平坦迭代访问。

### 初始状态

范围 `R0` 迭代 `0..32`（平坦索引）。访问模式分解为：

```text
row = R0 // 8       (which of the 4 rows)
col = R0 % 8        (which of the 8 columns)
addr = row * 8 + col = (R0 // 8) * 8 + (R0 % 8)
```

根据 div-mod 恒等式（#1），`(R0 // 8) * 8 + (R0 % 8) = R0`。地址就是平坦索引——不需要除法。

### 分块后（UPCAST 4 倍）

范围分割将 `R0` 分解为 `R1 * 4 + R2`，其中 `R1` 在 `[0, 8)` 内，`R2` 在 `[0, 4)` 内：

```text
row = (R1*4 + R2) // 8
col = (R1*4 + R2) % 8
```

**化简 `row`**：表达式 `(R1*4 + R2) // 8` 进入 `fold_divmod_general`。

规则 4（同余）触发：因子 `4` 的残余为 `4 % 8 = 4`，`R2` 的残余为 `1 % 8 = 1`。残余和为 `4*R1 + R2`，范围 `[0, 31]`。由于 `0 // 8 != 31 // 8`，规则 4 无法将其折叠为常量。规则 7（因子余数）接替触发：`4` 不能整除 `8`，但表达式可以分解。由于没有项能整除 8，我们回退到基于范围的模式 `(a*m + b) / n`，其中 `m = 4, n = 8`——不匹配（`m != n`）。

表达式保持为 `(R1*4 + R2) // 8`。在生成代码中，如果 `R2` 被向量化（UPCAST），后端将其作为 4 宽向量的单次除法。

但如果我们进一步将 `R1` 拆分为 `R3 * 2 + R4`（其中 `R3` 在 `[0, 4)` 内，`R4` 在 `[0, 2)` 内）：

```text
row = (R3*2*4 + R4*4 + R2) // 8
    = (R3*8 + R4*4 + R2) // 8
```

现在基于范围的模式 `(a*m + b) / n` 以 `m = n = 8` 触发：
- `a = R3`，`b = R4*4 + R2`
- `vmin(b) = 0`，`vmax(b) = 1*4 + 3 = 7 < 8`
- 结果：`R3 + (R4*4 + R2) // 8`

而 `(R4*4 + R2) // 8`：`vmax = 1*4 + 3 = 7`，`vmin = 0`，所以 `0 // 8 = 7 // 8 = 0`。cancel_divmod 规则触发：
- 结果：`R3 + 0 = R3`

**化简 `col`**：`(R3*8 + R4*4 + R2) % 8`

基于范围的模式 `(a*m + b) % n` 以 `m = n = 8` 触发：
- `(R3*8 + R4*4 + R2) % 8` -> `(R4*4 + R2) % 8`

然后 `vmin(R4*4 + R2) = 0`，`vmax(R4*4 + R2) = 7 < 8`，所以 `x % n` -> `x`：
- 结果：`R4*4 + R2`

### 最终树

```text
Before (after tiling, before simplification):
  STORE(
    INDEX(buf, (R3*8 + R4*4 + R2) // 8 * 8 + (R3*8 + R4*4 + R2) % 8),
    value)

After index arithmetic:
  STORE(
    INDEX(buf, R3*8 + R4*4 + R2),
    value)
```

整个地址计算折叠回线性表达式——零除法、零取模。模式证明了分块后的索引等价于平坦索引，完全通过代数重写。
