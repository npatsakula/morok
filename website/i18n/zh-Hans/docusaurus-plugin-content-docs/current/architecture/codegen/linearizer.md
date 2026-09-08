---
sidebar_label: 阶段 4 — Linearizer
---

# 阶段 4：Linearizer

**目标**：将 DAG 转换为线性指令序列。

---

## Stage 16：索引降低后符号化简

> **阶段速览**
>
> **目标**：索引降低之后的完整符号化简
> **关键模式**：全部符号规则（140+）
> **影响**：序列化前的最终清理

**做了什么**：索引降低之后的完整符号化简。

**为什么重要**：现在索引已经是具体整数（i32/i64），算术可以充分化简。这是线性化之前清理表达式的最后机会。

**模式**：`symbolic`

Svod 没有 GEP 操作——寻址是 `INDEX(STACK(...))`——因此 Tinygrad 的 `gep_pushing`
在这里没有对应物。最接近的类比是 `alu_vectorize_reorder_patterns`：
```text
Before:  ADD(STACK(x, x, x, x), STACK(y, y, y, y))
              ↓ [Reorder ALU over STACK]
After:   STACK(ADD(x, y), ADD(x, y), ADD(x, y), ADD(x, y))
```
*为什么？* 它让收缩后的操作可以做常量折叠和标量优化。该规则位于第 3 层的 `sym()` 中，所以它已经在 Stage 14 触发过，而不是在这里。

---

## Stage 17：Pre-Matcher（可选）

> **阶段速览**
>
> **目标**：分解之前的后端特定模式
> **关键模式**：Renderer 特定
> **影响**：硬件特定优化

**做了什么**：在分解之前应用 renderer 特定的模式。

**为什么重要**：每个后端可以添加自己的模式。例如，DSP 后端用这一步将通用模式替换为 DSP 特定的 SIMD 内联函数。这样无需修改通用流水线就能实现硬件特定优化。

**模式**：`renderer.pre_matcher`

大多数后端（CPU、GPU）不需要这一步。只有专用硬件使用它。

**注意**：Svod 没有 `pre_matcher`。后端钩子位于 `svod_device::device::Renderer` trait（`device/src/device.rs`）上：`decompositor()`、`extra_matcher()`、`pre_isel_matcher()` 和 `isel_matcher()`。后两者在 PROGRAM 边界处运行，即 Stage 20 和 21 之间，而不是在分解之前。（`svod_codegen::traits::Renderer` 是另一个更窄的 trait，只有 `render()`、`backend_name()` 和 `decompositor()`。）

---

## Stage 18：分解

> **阶段速览**
>
> **目标**：重写目标不支持的操作
> **关键模式**：2 的幂、超越函数近似
> **影响**：将高层操作映射到硬件指令

**做了什么**：对目标不支持的操作进行后期重写。

**为什么重要**：硬件并不支持所有操作。例如，大多数 CPU 没有直接的 `sin` 指令。我们用已有的操作（加法、乘法等）来近似它。

**模式**：`early_decomposition_patterns() + get_late_rewrite_patterns() + get_transcendental_patterns()`（当后端提供时还加上 `renderer.decompositor()`）。`early_decomposition_patterns()` 自身以 `symbolic_simple()` 开头。

注意：`pm_split_ends()` 不属于此 pass——它被并入了 Stage 19 的匹配器，并在 Stage 20 开头再次运行。

| 模式 | 示例 | 使用场景 |
|----------|---------|----------|
| `MOD → AND` | `x % 8 → x & 7` | 2 的幂除数 |
| `MUL → SHL` | `x * 16 → x << 4` | 2 的幂乘数 |
| `DIV → SHR` | `x / 8 → x >> 3` | 2 的幂除数（C 风格 CDIV） |
| `FDIV → MUL` | `x / 2.0 → x * 0.5` | 浮点常量除数 |
| `NEG` | `x * -1 → NEG(x)` | 当支持 NEG 时 |
| `MULACC` | `a * b + c → MULACC(a, b, c)` | 当支持 FMA 时 |
| 快速整数除法 | `x // 7 → (x * M) >> S` | 非 2 的幂除数 |
| 德摩根定律 | `(!x) & (!y) → !(x \| y)` | 布尔化简（仅限 NOT 的 AND 形式） |
| 比较取反 | `!(x < c) → (c-1) < x` | 整数比较 |

超越函数近似（EXP2、LOG2、SIN 等）来自 `get_transcendental_patterns()`（`ir/src/decompositions/mod.rs`，实现位于 `ir/src/decompositions/transcendentals.rs`）。当渲染器缺少某条指令时，会按操作逐个启用；当 `TRANSCENDENTAL=2` 时，则对所有操作启用。可选的 `Renderer::decompositor()` 钩子可以在此之上追加后端特定规则；树内还没有后端使用它。

**Svod**：`optimizer/mod.rs`

---

## Stage 19：最终重写

> **阶段速览**
>
> **目标**：为线性化做准备
> **关键模式**：弱类型 cast 落实、renderer 重写、END 分割
> **影响**：为线性化准备好干净的表示

**做了什么**：为线性化做准备。

**为什么重要**：有些模式在分解之后更容易应用。这个阶段在转换为线性序列之前做最后的清理。

**模式**：`pm_commit_weak() + pm_cast_weak() + pm_decomp`（即 Stage 18 的分解），再加上 `renderer.extra_matcher()` 和 `pm_split_ends()`——全部汇总成一个匹配器。随后 `pm_remove_invalid()` 和 `add_implicit_barriers()` 作为独立的 pass 运行。

注意：`extra_matcher` 和 `pm_split_ends` 是这个组合匹配器的一部分，而不是独立的 pass。Svod 没有 CONST 向量化或 GEP 解析步骤；Tinygrad 的 `pm_render` 在这里没有对应物。

**分割多 range END**：
```text
// Before: END closing multiple ranges
END(op, [range_a, range_b])

// After: nested single ENDs
END(END(op, range_a), range_b)
```

这些 range 按 `(axis_id, axis_type.priority())` 降序排序，因此最内层的 END 最先构建。Void/Bool 的"回边"源会被分离出来，重新挂到最外层的 END 上，并保留原有的 tag。

**extra_matcher**：每个后端可以添加自己的最终模式。这样无需修改通用流水线就能实现硬件特定优化。

**Svod**：`optimizer/mod.rs`、`linearize/mod.rs`

---

## Stage 20：添加控制流

> **阶段速览**
>
> **目标**：构建控制流图并添加 range 依赖
> **关键概念**：三种关系类型（嵌套、依赖、独立）
> **影响**：正确的指令排序

**做了什么**：构建控制流图并添加 range 依赖。

**为什么重要**：操作必须按有效顺序执行。如果一个 load 使用了 RANGE 的值，那么 RANGE 必须先执行。这个阶段跟踪并强制执行这些依赖。

**模式**：`pm_add_control_flow`（自底向上），在它之前还会再运行一次 `pm_split_ends`

```text
// Analyze which END operations depend on which
END(computation, [RANGE_A]) and END(other_computation, [RANGE_B]) are siblings
→ Creates edge: RANGE_B.src += END(computation)

// Add explicit dependency
RANGE_B waits for RANGE_A to complete
```

**三种关系类型**：

| 关系 | 条件 | 含义 |
|--------------|-----------|---------|
| 嵌套 | END_A 是 END_B 的依赖**并且** RANGE_B 是 END_A 的依赖 | A 的循环位于 B 的循环内部，因此 A 在 B 之前关闭 |
| 依赖 | END_A 是 END_B 的依赖，但没有这种嵌套关系 | B 的循环必须在 A 的循环之后发射 |
| 独立 | 两个 END 互不依赖 | 顺序自由；可以并行运行 |

自底向上遍历确保依赖从叶到根正确传播。

**Svod**：`schedule/src/linearize/mod.rs`、`schedule/src/linearize/cfg_context.rs`

---

## Stage 21：线性化

> **阶段速览**
>
> **目标**：将 DAG 转换为线性指令序列
> **关键算法**：优先级感知的拓扑排序
> **影响**：有效的执行顺序

**做了什么**：通过优先级感知的拓扑排序将 DAG 转换为线性指令序列。

**为什么重要**：图结构不指定执行顺序。我们需要在尊重依赖的前提下将其展平。优先级确保合理的排序（定义在使用之前、load 在计算之前、store 在最后）。

**函数**：`linearize(sink)`

| 操作 | 优先级 | 原因 |
|-----------|----------|-----|
| PARAM | -20 | 内核参数（以及符号变量）必须最先定义；同优先级时按参数槽位打破平局 |
| BUFFER | -18 | 分配优先 |
| BUFFER（`AddrSpace::Local`） | -17 | 本地分配紧跟在全局分配之后 |
| END | -5 | 关闭 range |
| LOAD | -1 | Load 在使用之前 |
| 其他所有（CONST、ALU 等） | 0 | 下沉到其消费者旁边 |
| STORE | +1 | Store 在计算之后 |
| RANGE | +5 | Range 在内容之前打开 |

优先级越低 = 序列中越靠前。这确保了：
- 定义最先
- Load 在计算之前
- Store 最后
- Range 在其内容之前打开，之后关闭

**run_count 排序**：操作主要按执行频率（run_count）排序，然后按优先级，再按 PARAM 槽位和 tuplize 秩排序。执行频率较低的操作（内层循环之外）先调度，而内层循环中的操作（run_count 更高）后调度。例如：执行 100 次的 CONST 出现在执行 100 万次的 CONST 之前。

**run_count 计算**：
```text
run_count = prod(int(r.vmax) + 1 for r in u.in_scope_ranges())
```
这根据包围它的作用域内 range 计算一个操作执行多少次；`vmax` 不是具体整数的 range 贡献 1。

**Svod**：`schedule/src/linearize/linearize.rs` 中的 `linearize()`

---

## Stage 22：清理 IF/ENDIF

> **阶段速览**
>
> **目标**：线性指令列表的最终清理
> **关键变换**：门控 STORE → IF/STORE/ENDIF
> **影响**：处理不支持谓词写入的硬件

**做了什么**：线性指令列表的最终清理。

**为什么重要**：某些硬件（现代 GPU）支持"谓词写入"——仅在条件为真时写入内存。较老的硬件不支持。对于那些硬件，我们将 store 包装在 IF 语句中。此阶段仅在缺少谓词写入支持的后端才需要；LLVM、CUDA 和 Metal 原生处理这个 gate，因此 `linearize_with_cfg()` 不会运行它。

**模式**：`line_rewrite_cleanups`（通过 `line_rewrite`，而非 `graph_rewrite`）

```text
// Gated STORE becomes a conditional store
STORE(INDEX(ptr, idx), value, gate=cond)
→ IF(cond) { STORE(INDEX(ptr, idx), value) } ENDIF
```

**注意**：此阶段使用 `line_rewrite` 而非 `graph_rewrite`，因为它操作的是已线性化的指令列表而非 DAG。

到此为止，指令列表已准备好进行代码生成。

**Svod**：`schedule/src/linearize/mod.rs` 中的 `line_rewrite_cleanups()`
