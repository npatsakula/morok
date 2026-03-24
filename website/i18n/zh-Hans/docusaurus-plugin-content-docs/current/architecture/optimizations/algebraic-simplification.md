---
sidebar_label: 代数化简
---

# 代数化简模式

Morok 的符号化简器使用 `schedule/src/symbolic/patterns.rs` 中定义的 140+ 个代数模式重写 UOp 计算图。这些模式在流水线的多个阶段触发：

| 位置 | 匹配器 | 上下文 |
|-------|---------|---------|
| 预优化 | `symbolic()` | rangeify + 范围分割之后，内核优化之前 |
| 后优化（阶段 8） | `symbolic()` | 优化动作之后，展开之前 |
| 索引后处理（阶段 16） | `symbolic()` | 索引 dtype 降级之后，最终清理 |
| 分解+渲染（阶段 18-19） | `symbolic_simple()` | 与晚期重写和渲染模式组合 |

`symbolic()` = `symbolic_simple()` + GEP 推送模式。除最终分解+渲染 pass 外，所有阶段运行完整的 `symbolic()` 集合。

**范围分析**：每个 UOp 跟踪它在运行时可取的最小值（`vmin`）和最大值（`vmax`），在节点构造时从输入的边界急切地计算。许多模式使用这些边界在编译期证明条件（例如"x 始终非负"或"x < n 对所有值成立"）。

**符号约定**：`OP[a, b]` 表示交换律模式（两种操作数顺序都会尝试）。`OP(a, b)` 表示有序。`@zero`/`@one`/`@const` 匹配常量值。当同一变量名出现两次时（例如 `Idiv(x, x)`），两个操作数必须是同一节点（`Arc::ptr_eq`——即通过 hash consing 进行结构去重）。

**Tinygrad 参考**：`tinygrad/uop/symbolic.py`、`tinygrad/uop/divandmod.py`

---

## 示例：优化级联

一个简单表达式展示模式如何组合：

```text
Before:
  ADD
  ├── MUL
  │   ├── ADD
  │   │   ├── x
  │   │   └── CONST(0)    <- identity
  │   └── CONST(1)         <- identity
  └── ADD
      ├── CONST(3)
      └── CONST(4)          <- constant fold

Step 1 (identity):    ADD(x, 0) -> x
Step 2 (identity):    MUL(x, 1) -> x
Step 3 (const fold):  ADD(3, 4) -> CONST(7)
Step 4 (result):      ADD(x, 7)

After:
  ADD
  ├── x
  └── CONST(7)
```

重写引擎自底向上应用模式：先化简子节点，再重新匹配父节点。这使得多步级联优化在一次遍历中完成。

---

## 模式排序

`symbolic_simple()` 匹配器按特定顺序组合模式组。在组内，模式按顺序尝试直到有一个匹配。组通过 `+` 运算符串联：

```text
propagate_invalid          -- MUST be first (before x*0=0)
fold_invalid_load_store
constant_folding_dsl_patterns
vconst_folding_patterns
identity_and_zero_patterns
commutative_canonicalization
self_folding_dsl_patterns
zero_folding_dsl_patterns
division_dsl_patterns
cast_dsl_patterns
cast_where_dsl_patterns
term_combining_dsl_patterns
alu_folding_dsl_patterns
advanced_division_dsl_patterns
div_mod_recombine_dsl_patterns
comparison_dsl_patterns
boolean_dsl_patterns
minmax_dsl_patterns
where_bound_patterns
power_dsl_patterns
negation_dsl_patterns
range_based_mod_div_patterns
dce_dsl_patterns
dead_loop_patterns
after_simplification_patterns
pm_move_where_on_load       -- WHERE->INDEX embedding for masked loads
```

---

## 1. 常量折叠

在编译期常量上使用 dtype 感知的算术求值。结果遵循类型边界（例如 Int32 在 32 位处回绕）。

**Tinygrad**：`symbolic.py:40-118`

### 标量常量

| 类别 | 操作 | 模式 |
|----------|-----|---------|
| 一元（7） | Neg, Sqrt, Exp2, Log2, Sin, Reciprocal, Trunc | `op(CONST(c))` -> `CONST(eval(op, c))` |
| 二元（13） | Add, Mul, Sub, Mod, Max, Pow, Idiv, Fdiv, And, Or, Xor, Shl, Shr | `op(CONST(a), CONST(b))` -> `CONST(eval(op, a, b))` |
| 三元（2） | Where, MulAcc | `op(CONST(a), CONST(b), CONST(c))` -> `CONST(eval(op, a, b, c))` |

### 向量常量

| 模式 | 结果 |
|---------|--------|
| `op(VCONST(a), VCONST(b))` | `VCONST(eval(op, a, b))` 逐元素 |
| `op(CONST(a), VCONST(b))` | `VCONST(eval(op, broadcast(a), b))` |
| `op(VCONST(a), CONST(b))` | `VCONST(eval(op, a, broadcast(b)))` |
| `unary_op(VCONST(v))` | `VCONST(eval(op, v))` 逐元素 |

VConst 折叠覆盖 11 个二元操作（不含 Pow 和 Fdiv）以及全部 7 个一元操作。

---

## 2. 恒等与零值传播

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `ADD[x, 0]` | `x` | 交换律 |
| `MUL[x, 1]` | `x` | 交换律 |
| `OR[x, 0]` | `x` | 交换律 |
| `XOR[x, 0]` | `x` | 交换律 |
| `SUB(x, 0)` | `x` | 有序 |
| `IDIV(x, 1)` | `x` | 有序 |
| `FDIV(x, 1)` | `x` | 有序 |
| `MOD(x, 1)` | `0` | 任何数模 1 为零 |
| `Floor/Ceil/Trunc/Round(x)` | `x` | 仅当 `x` 为整数时（取整为空操作） |
| `MUL[x, 0]` | `0` | 仅非浮点数 |
| `AND[_, 0]` | `0` | 交换律 |

:::caution IEEE 754：乘以零
`MUL[x, 0]` 对浮点数**不做**化简，因为 IEEE 754 要求：
- `NaN * 0 = NaN`
- `Inf * 0 = NaN`

守卫条件 `!x.dtype().is_float()` 阻止此优化用于浮点类型。
:::

---

## 3. 自折叠

同一操作数出现在两侧的模式。使用 `Arc::ptr_eq` 检查（hash consing 保证结构相等的子表达式共享同一指针）。

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `IDIV(x, x)` | `1` | |
| `IDIV(x, -1)` | `NEG(x)` | 常量检查右操作数 |
| `MOD(MOD(x, y), y)` | `MOD(x, y)` | 幂等取模 |
| `AND(x, x)` | `x` | |
| `OR(x, x)` | `x` | |

---

## 4. 零值折叠

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `MOD(x, x)` | `0` | |
| `LT(x, x)` | `false` | 非浮点数（NaN < NaN 为 false，但需要守卫以确保正确性） |
| `NE(x, x)` | `false` | 仅整数——IEEE 754 中 `NaN != NaN` 为 `true` |

---

## 5. 除法化简

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `FDIV(0.0, 0.0)` | `NaN` | IEEE 754 不定型 |
| `FDIV(MUL[_, 0], 0)` | `NaN` | 零表达式除以零 |
| `FDIV(x, x)` | `1.0` | 浮点自除 |
| `FDIV(MUL(x, y), y)` | `x` | 消去（浮点） |
| `IDIV(MUL(x, y), y)` | `x` | 消去（整数） |

:::caution 模式优先级
`FDIV(0, 0) -> NaN` 必须在匹配器中排在 `FDIV(x, x) -> 1` 之前以获得优先权。`division_dsl_patterns()` 中的排序保证了这一点。
:::

---

## 6. 类型转换优化

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `CAST(CONST(c), dtype)` | `CONST(c.cast(dtype))` | 编译期类型转换折叠 |
| `CAST(x, dtype)` | `x` | 当 `x.dtype() == dtype` 时（空操作） |
| `CAST(CAST(x, a), b)` | `x` | 当 `x.dtype() == b` 且 `a` 保留 `b` 的所有值 |
| `CAST(CAST(x, a), b)` | `CAST(x, b)` | 当 `a` 不窄化 `x`（宽化链） |
| `CAST(WHERE(s, a, b), dtype)` | `WHERE(s, CAST(a, dtype), CAST(b, dtype))` | 将类型转换推入分支 |

`can_safe_cast(to, from)` 函数判断中间类型是否能容纳所有值。它检查位宽、符号性以及浮点/整数类别。

:::caution 截断破坏往返
`CAST(CAST(x, i8), i64)` 当 `x` 为 `i64` 时**不会**折叠为 `x`。中间的 `i8` 截断值——`can_safe_cast(i64, i8)` 返回 `false`，因为 `i8` 无法容纳所有 `i64` 值。

安全示例：`CAST(CAST(x, i32), bool)` -> `CAST(x, bool)`，当 `x` 为 `bool` 时，因为 `i32` 能表示 `true` 和 `false`。
:::

---

## 7. 同类项合并

| 模式 | 结果 |
|---------|--------|
| `ADD(x, x)` | `MUL(2, x)` |
| `ADD(MUL(c1, x), MUL(c2, x))` | `MUL(c1+c2, x)` |
| `ADD(MUL(x, c1), MUL(x, c2))` | `MUL(x, c1+c2)` |

两种有序变体都会匹配（常量在 MUL 的左侧或右侧）。

---

## 8. ALU 链折叠

折叠结合律运算链中的常量，并将常量推向外层以获得规范形式。

### 常量折叠

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `ADD[ADD[x, c1], c2]` | `ADD(x, c1+c2)` | 外层 Add 交换律 |
| `MUL[MUL[x, c1], c2]` | `MUL(x, c1*c2)` | 外层 Mul 交换律 |
| `ADD[SUB(x, c1), c2]` | `ADD(x, c2-c1)` 或 `SUB(x, c1-c2)` | 符号归一化 |
| `SUB(ADD(x, c1), c2)` | `ADD(x, c1-c2)` 或 `SUB(x, c2-c1)` | 符号归一化 |
| `SUB(SUB(x, c1), c2)` | `SUB(x, c1+c2)` | |

### 常量外推

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `ADD[ADD[x, c], y]` | `ADD(ADD(x, y), c)` | 将常量推向外层；`y` 不能是常量 |

常量外推对索引提取至关重要。它确保常量冒泡到最外层，使下游模式（如 div-mod 化简）能看到干净的 `变量 + 偏移` 形式。

### Sub 规范化

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `SUB(a, SUB(b, x))` | `ADD(x, SUB(a, b))` | 暴露内部变量 |

Morok 保留 `SUB` 作为一等 IR 操作（不同于 Tinygrad 将 `a-b` 规范化为 `ADD(a, NEG(b))`）。此模式确保嵌套 `SUB` 不会阻断进一步化简。

---

## 9. 布尔逻辑

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `NOT(NOT(x))` | `x` | 双重否定消除 |
| `XOR(x, x)` | `0` | 自消去 |
| `OR[x, NOT(x)]` | `true` | 重言式（仅布尔） |
| `AND[x, NOT(x)]` | `false` | 矛盾（仅布尔） |
| `OR[true, x]` | `true` | 吸收元 |
| `AND[false, x]` | `false` | 吸收元 |
| `AND[true, x]` | `x` | 单位元 |
| `OR[false, x]` | `x` | 单位元 |
| `AND[NOT(x), NOT(y)]` | `NOT(OR(x, y))` | 德摩根定律 |
| `OR[NOT(x), NOT(y)]` | `NOT(AND(x, y))` | 德摩根定律 |

所有使用 `[]` 的模式都是交换律的（两种操作数顺序都会尝试）。

---

## 10. 比较化简

### 自比较（非浮点，ptr_eq）

| 操作 | 结果 |
|----|--------|
| `LT(x, x)`, `GT(x, x)`, `NE(x, x)` | `false` |
| `LE(x, x)`, `GE(x, x)`, `EQ(x, x)` | `true` |

:::caution 浮点自比较
自比较模式受 `!x.dtype().is_float()` 守卫。对于浮点数，`NaN != NaN` 为 `true`，`NaN == NaN` 为 `false`，因此这些恒等式不成立。
:::

### 常量与范围分析

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `op(CONST(a), CONST(b))` | `CONST(eval(op, a, b))` | 直接常量折叠 |
| `op(x, y)` 当边界可证明时 | `true` 或 `false` | `ComparisonAnalyzer` 使用 vmin/vmax |

`ComparisonAnalyzer` 检查：如果 `x.vmax < y.vmin`，则 `LT(x, y)` 可证明为 `true`。

### 代数变换

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `LT(ADD[c0, x], c1)` | `LT(x, c1-c0)` | 偏移消除 |
| `LT(NEG(x), NEG(y))` | `LT(y, x)` | 取反翻转 |
| `LT(IDIV(x, d), c)` | `LT(x, c*d)` | 提升除法（d > 0） |

`LT(x//d, c)` 的除法提升处理正数和非正数 `c`：
- `c > 0`：等价于 `x < c*d`
- `c <= 0`：等价于 `x < c*d - (d-1)`

---

## 11. Min/Max 消除

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `MAX(x, x)` | `x` | 自身取最大值为恒等 |
| `MAX(x, y)` | `x` | 当 `x.vmin >= y.vmax`（边界证明占优） |
| `MAX(x, y)` | `y` | 当 `y.vmin >= x.vmax` |

使用 `VminVmaxProperty` 进行范围分析。没有单独的 `MIN` 模式——Morok 在这些模式触发之前将 `MIN(a,b)` 降级为 `NEG(MAX(NEG(a), NEG(b)))`。

---

## 12. WHERE 优化

### 条件消除

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `WHERE(cond, t, f)` | `t` | 当 `cond.vmin == cond.vmax == true` |
| `WHERE(cond, t, f)` | `f` | 当 `cond.vmin == cond.vmax == false` |
| `WHERE(LT(x, c), t, f)` | `t` | 当 `x.vmax < c.vmin`（始终为真） |
| `WHERE(LT(x, c), t, f)` | `f` | 当 `x.vmin >= c.vmax`（始终为假） |

### 分支化简

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `WHERE(_, t, t)` | `t` | 相同分支 |
| `WHERE(x, true, false)` | `x` | 布尔恒等 |
| `WHERE(x, false, true)` | `NOT(x)` | 布尔取反 |
| `WHERE(NOT(cond), t, f)` | `WHERE(cond, f, t)` | 条件翻转 |
| `WHERE(a, WHERE(b, c, d), d)` | `WHERE(AND(a, b), c, d)` | 分支合并（`d` 上使用 ptr_eq） |

:::caution 条件翻转的 Invalid 守卫
`WHERE(NOT(cond), t, f) -> WHERE(cond, f, t)` 当 `f` 包含 `Invalid` 时**不会**应用。填充操作创建 `WHERE(valid, idx, Invalid)` 结构，交换后会将 `Invalid` 移到真分支，使下游模式无法匹配。标量 `Invalid` 和向量化 `VECTORIZE(Invalid, ...)` 都会被检查。

Tinygrad 有相同的守卫：`symbolic.py:201-202`。
:::

---

## 13. Invalid 传播

Invalid 是 Morok 对填充操作创建的越界张量区域的哨兵值。这些模式必须在恒等模式（如 `x*0=0`）**之前**运行，否则有效性标记会被破坏。

### 模式优先级示例

```text
Without ordering:  MUL(0, WHERE(cond, x, Invalid)) -> 0    (x*0=0 fires, loses Invalid)
With ordering:     MUL(0, WHERE(cond, x, Invalid))
                 -> WHERE(cond, MUL(0, x), Invalid)         (Invalid propagation fires first)
                 -> WHERE(cond, 0, Invalid)                  (then x*0=0 is safe)
```

### WHERE-Invalid 合并

| 模式 | 结果 |
|---------|--------|
| `WHERE(c1, WHERE(c2, x, Inv), Inv)` | `WHERE(AND(c1, c2), x, Inv)` |
| `WHERE(c1, WHERE(c2, x, Inv), y)` | `WHERE(AND(c1, c2), x, y)` |

多维填充在通过线性化索引算术传播后会创建嵌套 WHERE-Invalid。合并为单层确保 `pm_lower_index_dtype` 能一步处理。

### 将操作推入 WHERE-Invalid

| 模式 | 结果 | 操作 |
|---------|--------|-----|
| `CAST(WHERE(c, x, Inv))` | `WHERE(c, CAST(x), Inv)` | |
| `op(WHERE(c, x, Inv), y)` | `WHERE(c, op(x, y), Inv)` | 13 个二元操作（非比较） |
| `op(y, WHERE(c, x, Inv))` | `WHERE(c, op(y, x), Inv)` | 13 个二元操作（非比较） |
| `cmp(WHERE(c, x, Inv), y)` | `cmp(x, y)` | Lt, Le, Eq, Ne, Gt, Ge |
| `cmp(y, WHERE(c, x, Inv))` | `cmp(y, x)` | Lt, Le, Eq, Ne, Gt, Ge |

对于比较操作，WHERE-Invalid 被剥离——Invalid 区域已在下游被门控。

### 裸 Invalid 传播

| 模式 | 结果 | 守卫 |
|---------|--------|-------|
| `op(Invalid, y)` | `Invalid` | `y.dtype() == DType::Index`，仅左侧位置 |

Tinygrad 对齐：`symbolic.py:37`。右侧位置的裸 Invalid **不会**传播，以避免污染非索引计算。

### Invalid 索引的死加载/存储

| 模式 | 结果 |
|---------|--------|
| `LOAD(INDEX(buf, Invalid))` | `CONST(0)` |
| `LOAD(CAST(INDEX(buf, Invalid)))` | `CONST(0)` |
| `STORE(INDEX(buf, Invalid), val)` | `NOOP` |
| `STORE(CAST(INDEX(buf, Invalid)), val)` | `NOOP` |

---

## 14. 死代码消除

### 死范围

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `RANGE(end)` 且 `vmax < 0` | `CONST(0)` | 空范围（永不执行） |
| `RANGE(CONST)` 且 `vmin == vmax` | `CONST(vmin)` | 平凡范围（单一值） |
| `END(computation, ranges)` | `END(computation, live_ranges)` | 从 END 过滤死范围 |
| `END(computation, [])` | `computation` | 所有范围已死——展开 |

### 死规约

| 规约操作 | 单位元 |
|-----------|-----------------|
| Add | `0` |
| Mul | `1` |
| Max | `-inf`（dtype 最小值） |
| Min | `+inf`（dtype 最大值） |

当 REDUCE 的所有范围都为死（空）时，REDUCE 被替换为其单位元。

### 依赖化简

| 模式 | 结果 |
|---------|--------|
| `AFTER(x, [])` | `x` |

无依赖意味着无排序约束。

---

## 15. 幂与取反

| 模式 | 结果 |
|---------|--------|
| `POW(x, 0)` | `1` |
| `POW(x, 1)` | `x` |
| `NEG(NEG(x))` | `x` |

---

## 16. GEP 推送

GEP（Get Element Pointer）从向量中提取元素。这些模式将 GEP 推过其他操作以到达向量源，在反向量化之后实现标量化简。

仅包含在 `symbolic()`（阶段 4）中，不包含在 `symbolic_simple()`（阶段 8、16）中。

### 组合与提取

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `GEP(GEP(x, inner), outer)` | `GEP(x, inner[outer])` | 组合嵌套 |
| `GEP(VECTORIZE(x,x,x,x), [i])` | `x` | 穿过广播（全部 ptr_eq） |
| `GEP(VECTORIZE(elems), [i])` | `elems[i]` | 穿过 VECTORIZE |
| `GEP(scalar, [i])` | `scalar` | 标量恒等（vcount == 1） |
| `GEP(VCONST(vals), [i])` | `CONST(vals[i])` | 穿过 VConst |
| `GEP(x, [0,1,...,n-1])` | `x` | 恒等移除 |

### 推过操作

| 模式 | 结果 | 守卫 |
|---------|--------|-------|
| `GEP(op(a, b), idx)` | `op(GEP(a, idx), GEP(b, idx))` | 二元，仅 Index dtype |
| `GEP(unary(x), idx)` | `unary(GEP(x, idx))` | 一元，仅 Index dtype |
| `GEP(WHERE(c, t, f), idx)` | `WHERE(GEP(c, idx), GEP(t, idx), GEP(f, idx))` | 仅 Index dtype |
| `GEP(MULACC(a, b, c), idx)` | `MULACC(GEP(a, idx), GEP(b, idx), GEP(c, idx))` | 仅 Index dtype |

:::caution Index dtype 守卫防止图爆炸
GEP 推过 ALU 操作被限制为 `Index` dtype（Tinygrad：`symbolic.py:167`）。没有此守卫，GEP 推送与 `no_vectorized_alu` 的组合会在高维内核上导致指数级图膨胀。
:::

### 推过结构操作

| 模式 | 结果 |
|---------|--------|
| `GEP(CAT([a<4>, b<4>]), [5])` | `GEP(b, [1])` |
| `GEP(PTRCAT([a, b, c]), [1, 2])` | `PTRCAT([b, c])` |
| `GEP(CAST(x, dtype), idx)` | `CAST(GEP(x, idx), scalar_dtype)` |
| `GEP(BITCAST(x, dtype), idx)` | `BITCAST(GEP(x, idx), scalar_dtype)` |
| `GEP(WMMA(a, b, c), idx)` | `WMMA(GEP(a, ...), GEP(b, ...), GEP(c, ...))` |
| `GEP(UNROLL(x, ...), idx)` | `GEP(x, idx)` |
| `GEP(void_node, _)` | `void_node` |

WMMA 模式通过 upcast 轴映射 tile 索引以提取对应的输入子组。

### 重收集

| 模式 | 结果 |
|---------|--------|
| `VECTORIZE(GEP(x,[0]), GEP(x,[1]), ..., GEP(x,[N-1]))` | `GEP(x, [0,1,...,N-1])` |

这将 `no_vectorized_alu` 创建的 VECTORIZE 结构折叠回单个 GEP，然后恒等模式将其移除。

---

## 17. LOAD 上的 WHERE（仅阶段 8）

**函数**：`pm_move_where_on_load()`

通过将条件嵌入 INDEX 操作来变换带掩码的加载：

```text
Before:  WHERE(cond, INDEX(buf, idx), 0)
After:   INDEX(buf, WHERE(combined_cond, idx, Invalid))
```

这启用了硬件谓词化的带掩码加载，消除了 WHERE 开销。

### 工作原理

1. **拆分**条件为 AND 子句
2. **分区**子句为可移动与剩余：
   - 可移动：所有 RANGE 依赖在 INDEX 作用域内，无外部 INDEX 依赖
   - 剩余：其他一切
3. **嵌入**可移动子句为 `indices[0]` 中的 `WHERE(cond, idx, Invalid)`
4. 如果存在剩余子句，**包装**外层 WHERE

支持部分子句移动——仅移动范围在索引作用域内的子句。`indices[0]` 中已有的有效性子句会被去重。

反转模式 `WHERE(cond, 0, INDEX(buf, idx))` 也通过取反条件来处理。

---

## 18. 交换律规范化

对于 Index dtype 上的交换二元操作，操作数按 UOp id 排序（较小 id 在左）：

| 操作 | 守卫 |
|-----|-------|
| Add, Mul, Max, Eq, Ne, And, Or, Xor | `dtype == DType::Index && b.id < a.id` |

没有此规范化，数学等价的表达式如 `R1*8000 + R2*16` 和 `R2*16 + R1*8000` 不会被 hash consing 去重，从而破坏 `expand_vector_index` 中的分组。

仅应用于 Index dtype 以避免破坏向量数学合并。Tinygrad：`symbolic.py:178-182`。

---

## 19. Div-Mod 化简

### 基于范围的快速路径

| 模式 | 结果 | 条件 |
|---------|--------|-----------|
| `MOD(x, n)` | `x` | `0 <= vmin(x)` 且 `vmax(x) < n` |
| `IDIV(x, n)` | `k` | 范围内所有值除以同一个 `k` |
| `MOD(ADD[MUL[a, m], b], n)` | `MOD(b, n)` | `m == n`（提取倍数） |
| `IDIV(ADD[MUL[a, m], b], n)` | `a + IDIV(b, n)` | `m == n` |
| `IDIV(ADD[MUL[a, m], b], n)` | `a` | `m == n` 且 `0 <= b < n` |

### 统一 Div-Mod 引擎（`fold_divmod_general`）

对于 Index dtype 上的 IDIV 和 MOD，统一引擎按优先级顺序尝试化简规则。基于 Tinygrad 的 `fold_divmod_general`（`divandmod.py:8-93`）。

| 优先级 | 规则 | 描述 |
|----------|------|-------------|
| 1 | cancel_divmod | 范围位于单个除数区间内 |
| 2 | remove_nested_mod | `(a%4 + b)%2 -> (a+b)%2`，当 `2 | 4` |
| 3 | fold_binary_numerator | 范围恰好为 2 的单项 |
| 4 | fold_divmod_congruence | 因子同余模算术 |
| 5 | gcd_with_remainder | 从分子中提取公共 GCD |
| 6 | divide_by_gcd | 变量分母 GCD 分解 |
| 7 | factor_remainder | `(d*x+y)//d -> x + y//d`（最后手段） |

### Div-Mod 重组合

将分离的 div 和 mod 操作重组合回原始表达式的模式：

| 模式 | 结果 | 守卫 |
|---------|--------|-------|
| `ADD[MOD(x, n), MUL[IDIV(x, n), n]]` | `x` | x, n 上使用 ptr_eq |
| `ADD[MOD(IDIV(x, a), c), MUL[IDIV(x, b), c]]` | `IDIV(x, a)` | `a * c == b` |
| `ADD[MUL[MOD(x, c1), c2], MUL[IDIV(x, c1), c3]]` | `MUL(x, c2)` | `c1 * c2 == c3` |
| `ADD[ADD[y, MOD(x, n)], MUL[IDIV(x, n), n]]` | `ADD(y, x)` | x, n 上使用 ptr_eq |
| `IDIV(ADD[IDIV(a, c1), c2], c3)` | `IDIV(ADD(a, c1*c2), c1*c3)` | 嵌套除法 |

### 高级除法

| 模式 | 结果 | 说明 |
|---------|--------|-------|
| `IDIV(IDIV(a, b), c)` | `IDIV(a, b*c)` | 组合嵌套除法 |
| `IDIV(expr, d)` | `expr.divides(d)` | 通用精确除法 |
| `IDIV(ADD(a, b), c)` | `IDIV(a, c) + IDIV(b, c)` | 两边都能整除时 |
| `IDIV(SUB(a, b), c)` | `IDIV(a, c) - IDIV(b, c)` | 两边都能整除时 |
| `MUL(c, ADD(a, b))` | `ADD(MUL(c, a), MUL(c, b))` | 分配乘法 |

---

## 交叉引用

- [执行流水线](../pipeline.md)——这些模式运行的阶段
- [模式引擎](./pattern-system)——模式匹配引擎的工作原理
- [Rangeify](../codegen/rangeify.md)——阶段 4 上下文（模式在变换操作降级后运行）
- [Expander](../codegen/expander.md)——阶段 8 上下文（模式在优化动作后运行）
- [Linearizer](../codegen/linearizer.md)——阶段 16 上下文（最终清理）
