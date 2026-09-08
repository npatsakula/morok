---
sidebar_label: 实践示例
---

# 实践：从 Tensor 到模型

本章通过渐进式示例教你使用 Svod。你将从基本的张量操作开始，逐步构建出一个完整的神经网络分类器。

**你将学到：**
- 创建和操作张量
- 形状操作（reshape、transpose、broadcast）
- 矩阵乘法
- 构建可复用的层
- 组装完整的模型

**前置条件：**
- 基本的 Rust 知识
- 在 `Cargo.toml` 中添加 `svod_tensor`

**核心模式：** Svod 使用*惰性求值*。操作只构建计算图，不会立即执行。调用 `realize()` 时才会一次性编译和运行所有操作。

---

## 示例 1：Hello Tensor

创建张量、执行操作并获取结果。

```rust
use svod_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors from slices
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);

    // Lazy operations (no execution yet); a scalar is a valid right-hand side
    let sum = (&a + &b)?;
    let scaled = (&sum * 0.1)?;

    // Execute and get results
    scaled.realize()?;
    let data = scaled.as_ndarray::<f32>()?;
    println!("Result: {:?}", data);
    // Output: [1.1, 2.2, 3.3, 4.4]

    Ok(())
}
```

**发生了什么：**

1. `Tensor::from_slice()` 从数组数据创建一维张量。`f32` 后缀告诉 Rust 元素类型。

2. `&a + &b` 不会执行任何计算。它返回 `Result<Tensor>`——形状或 dtype 不匹配是可恢复的错误，所以需要 `?`——其中包装的张量*表示*这次加法。`&` 借用张量以便后续复用。`2.0 * &a` 同样可行：标量在左右两侧都被接受，并会按张量的 dtype 物化。

3. `realize()` 是关键所在。它接受 `&self`，因此已 realize 的张量可以一直处于共享借用之下。Svod 会：
   - 分析计算图
   - 尽可能融合操作
   - 生成优化后的代码
   - 在目标设备上执行

4. `as_ndarray()` 将已经算好的结果提取为 `ndarray::ArrayD` 以供查看。

**试试看：** 去掉 `realize()` 调用。此时 `as_ndarray()` 会以“没有缓冲区”的错误失败——什么都没有被计算，也就没有结果可读。`to_ndarray()`、`to_vec()` 和 `item()` 会按需 realize 而不是失败；`as_ndarray()` / `as_vec()` 从不 realize，因此在“触发 realize 本身就是 bug”的场景中依然可用。

---

## 示例 2：形状变换

神经网络不断地重塑数据。来掌握基础操作。

```rust
use svod_tensor::Tensor;
use ndarray::array;

fn shape_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 1D tensor with 6 elements
    let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("Original shape: {:?}", data.dims()?);  // [6]

    // Reshape to a 2x3 matrix (or create directly with from_ndarray)
    let matrix = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    println!("Matrix shape: {:?}", matrix.dims()?);  // [2, 3]
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Transpose to 3x2
    let transposed = matrix.try_transpose(0, 1)?;
    println!("Transposed shape: {:?}", transposed.dims()?);  // [3, 2]
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]

    // Broadcasting: add a row vector to every row
    // [3, 2] + [1, 2] → [3, 2]
    let bias = Tensor::from_ndarray(&array![[100.0f32, 200.0]]);
    let biased = (&transposed + &bias)?;

    biased.realize()?;
    println!("{:?}", biased.as_ndarray::<f32>()?);
    // [[101, 204],
    //  [102, 205],
    //  [103, 206]]

    Ok(())
}
```

**核心操作：**

| 操作 | 功能说明 |
|-----------|--------------|
| `try_reshape(&[2, 3])` | 改变形状（总元素数不变） |
| `try_reshape(&[-1, 3])` | 根据总大小自动推断维度 |
| `try_transpose(0, 1)` | 交换第 0 和第 1 维 |
| `try_squeeze(dim)` | 移除大小为 1 的维度 |
| `try_unsqueeze(dim)` | 添加大小为 1 的维度 |

**读取形状：** `dims()` 返回 `Vec<usize>`，若有任何一个轴是符号维则报错；`dim(axis)` 以 `SInt`（符号或常量）返回该轴，`dim_const(axis)` 以 `usize` 返回，当它不是常量时以 `NonConstDim` 失败；`shape()` 返回由 `SInt` 组成的完整 `Shape`。`dtype()` 不会失败；`Tensor` 实现了 `Debug`——打印形状、dtype、设备以及是否已 realize，但从不打印数据，因为那会强制读取设备。负数轴在任何地方都从末尾开始计数。

**广播规则**（与 NumPy/PyTorch 相同）：
- 形状从右侧对齐
- 每个维度必须匹配或为 1
- 大小为 1 的维度会被"拉伸"以匹配

```text
[3, 2] + [1, 2] → [3, 2]  ✓ (1 broadcasts to 3)
[3, 2] + [2]    → [3, 2]  ✓ (implicit [1, 2])
[3, 2] + [3]    → error   ✗ (2 ≠ 3)
```

---

## 示例 3：矩阵乘法

矩阵乘法是神经网络的核心运算，每一层都会用到它。

```rust
use svod_tensor::Tensor;
use ndarray::array;

fn matmul_example() -> Result<(), Box<dyn std::error::Error>> {
    // Input: 4 samples, 3 features each → shape [4, 3]
    let input = Tensor::from_ndarray(&array![
        [1.0f32, 2.0, 3.0],    // sample 0
        [4.0, 5.0, 6.0],       // sample 1
        [7.0, 8.0, 9.0],       // sample 2
        [10.0, 11.0, 12.0],    // sample 3
    ]);

    // Weights: 3 inputs → 2 outputs → shape [3, 2]
    let weights = Tensor::from_ndarray(&array![
        [0.1f32, 0.2],  // feature 0 → outputs
        [0.3, 0.4],     // feature 1 → outputs
        [0.5, 0.6],     // feature 2 → outputs
    ]);

    // Matrix multiply: [4, 3] @ [3, 2] → [4, 2]
    let output = input.dot(&weights)?;

    output.realize()?;
    println!("Output shape: {:?}", output.dims()?);  // [4, 2]
    println!("{:?}", output.as_ndarray::<f32>()?);
    // Each row: weighted sum of that sample's features

    Ok(())
}
```

**`dot()` 的形状规则：**

| 左操作数 | 右操作数 | 结果 |
|------|-------|--------|
| `[M, K]` | `[K, N]` | `[M, N]` |
| `[K]` | `[K, N]` | `[N]`（向量-矩阵） |
| `[M, K]` | `[K]` | `[M]`（矩阵-向量） |
| `[B, M, K]` | `[B, K, N]` | `[B, M, N]`（批量） |

内部维度必须匹配（即 `K`）。可以这样理解："对左矩阵的每一行，与右矩阵的每一列做点积。"

---

## 示例 4：构建线性层

线性层计算 `y = x @ W.T + b`。Svod 提供了开箱即用的 `nn::Linear`。

```rust
use svod_tensor::{Tensor, nn::{Linear, Layer}};

fn linear_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a layer: 4 inputs → 2 outputs, with a bias
    let layer = Linear::with_dims(4, 2, true, svod_dtype::DType::Float32);

    // Single sample with 4 features
    let input = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

    // Forward pass
    let output = layer.forward(&input)?;

    output.realize()?;
    println!("Output: {:?}", output.as_ndarray::<f32>()?);

    Ok(())
}
```

**为什么要转置权重？**

PyTorch 惯例将权重存储为 `[out_features, in_features]`。对于一个 4 → 2 的映射层：
- 权重形状：`[2, 4]`
- 输入形状：`[4]` 或 `[batch, 4]`
- 我们需要：`input @ weight.T` = `[batch, 4] @ [4, 2]` = `[batch, 2]`

这种惯例使权重矩阵易于阅读：第 `i` 行包含所有输入到第 `i` 个输出的权重。

---

## 示例 5：MNIST 分类器

使用 `sequential()` 链接层，构建一个完整的神经网络。

```rust
use svod_tensor::{Tensor, nn::{Linear, Relu, Layer}};

fn mnist_example() -> Result<(), Box<dyn std::error::Error>> {
    // Architecture: 784 (28×28 pixels) → 128 (hidden) → 10 (digits)
    let fc1 = Linear::with_dims(784, 128, true, svod_dtype::DType::Float32);
    let fc2 = Linear::with_dims(128, 10, true, svod_dtype::DType::Float32);

    // Simulate a 28×28 grayscale image (flattened to 784)
    let fake_image: Vec<f32> = (0..784)
        .map(|i| (i as f32) / 784.0)
        .collect();
    let input = Tensor::from_slice(fake_image)
        .try_reshape(&[1, 784])?;  // batch size 1

    // Forward pass: linear → ReLU → linear
    let logits = input.sequential(&[&fc1, &Relu, &fc2])?;
    let probs = logits.softmax(-1)?;

    // Get predicted class; realize both results in one compilation
    let prediction = logits.argmax(Some(-1))?;
    Tensor::realize_batch([&probs, &prediction])?;

    println!("Probabilities: {:?}", probs.as_ndarray::<f32>()?);
    println!("Predicted digit: {:?}", prediction.as_ndarray::<i32>()?);

    Ok(())
}
```

**核心概念：**

1. **`sequential()`** 将层串联起来：每层的输出自动作为下一层的输入。无需手动连线。

2. **ReLU 激活函数：** `Relu` 是一个零大小的层，应用 `max(0, x)`。它引入非线性——没有它的话，堆叠线性层只相当于一个大的线性层。

3. **Logits 与概率：** 最后一层的原始输出（logits）可以是任意实数。`softmax()` 将它们转换为总和为 1 的概率。

4. **argmax：** 返回最大值的索引——即预测的类别。

5. **批维度：** 单张图像使用形状 `[1, 784]`。如果有 32 张图像，使用 `[32, 784]`。模型会自动处理批次。

6. **`realize_batch`：** 两个共享子图的结果（这里是 logits）会一起编译并一起运行，共享的部分因此只计算一次。它接受共享引用——`[&a, &b]`——因为 realize 的状态记录在张量注册表中，而不是句柄里。

---

## 示例 6：深入内部

想看看 Svod 生成了什么？以下是如何查看 IR 和编译出的 kernel。

```rust
use svod_tensor::Tensor;

fn inspect_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
    let c = (&a + &b)?;

    // Print the computation graph (before compilation)
    println!("=== IR Graph ===");
    println!("{}", c.uop().tree());

    // Compile and inspect the execution plan
    let plan = c.prepare()?;  // prepare() takes &self
    println!("\nKernels: {}", plan.kernels().count());

    // Execute
    plan.execute()?;

    Ok(())
}
```

**你会看到：**

1. **IR 图：** UOp 树展示了 `BUFFER`、`LOAD`、`ADD`、`STORE` 等操作。这是 Svod 在优化之前的中间表示。

2. **执行计划：** `prepare()` 返回编译好的 kernel。注意 Svod 是如何将两次 load 和 add 融合到一个 kernel 中的——无需中间缓冲区。

**调试技巧：** 如果某些操作看起来慢或不对，打印 IR 树。注意检查：
- 意外的操作（冗余的 reshape、多余的拷贝）
- 缺少融合（本可以用一个 kernel 完成的地方却用了多个）
- 形状不匹配（通常是错误的根本原因）

---

## 示例 7：层、模块与 state dict

一个 layer 结构体持有自己的参数，以及 forward 所需的超参数。
`#[derive(Module)]` 把这些字段变成一个扁平的 `StateDict`
（`HashMap<String, Tensor>`），键名与 PyTorch 的命名完全一致，因此
checkpoint 无需手写映射即可加载：

```rust
use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::nn::{LayerNorm, Module, StateDict};

#[derive(Clone, Module)]
struct Block {
    intermediate: usize,            // primitives are skipped automatically
    #[module(skip)]                 // a non-primitive that carries no weights
    dtype: DType,
    norm: LayerNorm,                // child module: "norm.weight", "norm.bias"
    #[module(key = "Wi.weight")]    // checkpoint name, dots allowed
    wi: Tensor,
    #[module(key = "Wo.weight")]
    wo: Tensor,
    #[module(optional)]             // written when Some, absent-tolerant on load
    out_bias: Option<Tensor>,
}

fn load(checkpoint: &StateDict) -> Result<Block, Box<dyn std::error::Error>> {
    let mut block = Block {
        intermediate: 3072,
        norm: LayerNorm::with_dims(768, true, 1e-5, DType::Float32),
        wi: Tensor::zeros(&[3072, 768], DType::Float32),
        wo: Tensor::zeros(&[768, 3072], DType::Float32),
        out_bias: None,
    };
    // Reads "layers.0.norm.weight", "layers.0.Wi.weight", ...
    block.load_state_dict(checkpoint, "layers.0")?;
    // ...and writes them back out under any prefix
    let _round_trip: StateDict = block.state_dict("layers.0");
    Ok(block)
}
```

| 属性 | 作用 |
|-----------|--------|
| `#[module(key = "Wi.weight")]` | 替换由字段名生成的键片段（可包含点号和数字） |
| `#[module(key = "")]` | 展平：该字段的键直接沿用父级前缀 |
| `#[module(skip)]` | 忽略一个非原始类型字段（配置、dtype、模式） |
| `#[module(optional)]` | `Option<Tensor>` 上必须标注：为 `Some` 时保存，加载时容忍键缺失 |
| `#[module(optional = "self.has_bias")]` | 谓词成立时该键是必需的，否则跳过 |

子模块通过 blanket impl 组合：`Vec<Block>` 把元素的键设为 `0.`、
`1.`、……，数组、`Option`、元组和 `Box` 也以同样方式委托。枚举同样可以
derive。前向计算不属于 `Module`：签名允许时它位于 `Layer`
trait（`fn forward(&self, x: &Tensor) -> Result<Tensor>`），否则位于固有方法中。

内置层同时实现这两个 trait，`new` 用于已加载的张量，`with_dims`
则做一次全新的 Kaiming 均匀初始化（卷积）或恒等仿射初始化（归一化）：

| 层 | `with_dims` | State-dict 键 |
|-------|-------------|-----------------|
| `Linear` | `(in, out, bias, dtype)` | `weight`、`bias`（存在时） |
| `Conv1d` | `(in_c, out_c, kernel, bias, dtype)` | `weight`、`bias` |
| `Conv2d` / `ConvTranspose2d` | `(in_c, out_c, (kh, kw), bias, dtype)` | `weight`、`bias` |
| `BatchNorm2d` | `(channels, eps, dtype)` | `weight`、`bias`、`running_mean`、`running_var` |
| `LayerNorm` | `(size, bias, eps, dtype)` | `weight`、`bias`（存在时） |
| `RmsNorm` | `(size, eps, dtype)` | `weight` |
| `Embedding` | `(vocab_size, embed_dim, dtype)` | `weight` |

超参数通过结构体上 builder 风格的 `with_*` 方法设置——
`Conv1d::new(w, bias).with_stride(2).with_padding((1, 1)).with_groups(4)`、
`LayerNorm::with_dims(..).with_axis(-2)`。

---

## 示例 8：循环层

`rnn()`、`gru()` 和 `lstm()` 是 `Tensor` 上的构建器。它们既接受 PyTorch
的权重名（`weight_ih`、`weight_hh`、`bias_ih`、`bias_hh`、`h0`、
`c0`），也接受 ONNX 的（`w`、`r`/`r_weights`、`bias`、`initial_h`、
`initial_c`），并会替你重排门的分块：

```rust
use svod_tensor::Tensor;
use ndarray::Array3;

// seq=2, batch=1, input=3, hidden=4
let x = Tensor::from_ndarray(&Array3::from_elem((2, 1, 3), 0.1f32));
let w = Tensor::from_ndarray(&Array3::from_elem((1, 12, 3), 0.1f32));
let r = Tensor::from_ndarray(&Array3::from_elem((1, 12, 4), 0.1f32));

let out = x.gru().w(&w).r_weights(&r).hidden_size(4).call()?;
// ONNX-shaped: y [seq, num_directions, batch, hidden], y_h [num_directions, batch, hidden]
// PyTorch-shaped: output [seq, batch, D*hidden], h_n [num_directions, batch, hidden]
assert_eq!(out.y.dims()?, vec![2, 1, 1, 4]);
assert_eq!(out.output.dims()?, vec![2, 1, 4]);
```

`layout` 用于选择 `RnnLayout::SeqFirst`（`[seq, batch, input]`，默认）或
`BatchFirst`；`direction` 接受 `RnnDirection::{Forward, Backward,
Bidirectional}`，双向计算会在特征轴上拼接两个方向的结果。GRU 的
`linear_before_reset` 在使用 PyTorch 权重时默认采用 PyTorch 的放置方式，
使用 ONNX 权重时则默认采用 ONNX 的。`LstmOutput`
额外提供表示细胞状态的 `y_c` / `c_n`。

时间轴必须是具体值，但批轴可以是符号 `Variable`。若要手写循环——比如
逐 token 步进的解码器——可以直接使用 cell：`GruCell`/`LstmCell`/`RnnCell`
提供 `step(&x, &h) -> Result<..>`，而 `RnnStack::new(cells)` 可以一次步进
整个堆栈。

---

## 示例 9：频谱图

`stft()` 就是一次针对加窗 DFT 卷积核的 `conv1d`，因此整个变换都留在图里
（批轴也可以保持符号化）。结果是 `[B, F, T, 2]`——对未加批的 `[L]`
信号则是 `[F, T, 2]`——末轴放置 `(real, imag)`，与
`torch.stft(..., return_complex=false)` 一致：

```rust
use svod_tensor::Tensor;
use svod_tensor::nn::Window;

let x = Tensor::from_slice(vec![0.25f32; 64]);
let spec = x.stft().n_fft(16).hop(4).window(Window::Hann).call()?;
assert_eq!(spec.dims()?, vec![9, 17, 2]);   // [F, T, (re, im)]

let mag = spec.magnitude(0.0)?;             // sqrt(re² + im² + eps)
let signal = spec.istft().n_fft(16).hop(4).window(Window::Hann).length(64).call()?;
```

默认值遵循 torch：`hop = n_fft / 4`、`win_length = n_fft`、周期性 Hann
窗、`center`、`onesided`、不做归一化——`istft` 也必须传入同样的参数。
`Window` 可以是 `Hann`、`Hamming`、`Rectangular` 或 `Custom(tensor)`，
`Tensor::window(&Window::Hann, n, periodic, dtype)` 会物化出一个窗。
除 `magnitude` 外，末轴为 2 的表示还提供 `power`、`complex_abs`、
`complex_mul` 以及 `Tensor::complex_from_polar(&mag, &phase)`。

---

## 错误处理

每个可能失败的张量方法都返回 `svod_tensor::error::Result<T>`，其错误是
指针大小的 `Error(Box<ErrorKind>)`；通过 `err.kind()`（或用 `into_kind()`
按值取出）匹配具体原因。下游 crate 用 snafu 的 `context(false)` 转换它，
因此模型自己的错误枚举用一个普通的 `?` 就能吸收它——无需在每个调用点
写 `.context(TensorSnafu)`。

并非所有操作都可能失败。`cast`、`neg`、`abs`、`floor`、`ceil`、`round`、
`trunc`、`square`、`sign` 以及 `Tensor::full` / `zeros` / `ones`
构造函数都不会失败，直接返回普通的 `Tensor`；`-&a` 同样是普通值，
而二元运算符返回 `Result<Tensor>`。

---

## 总结

你已经学会了使用 Svod 的核心模式：

| 任务 | 代码 |
|------|------|
| 创建张量 | `Tensor::from_slice([1.0f32, 2.0])` |
| 算术运算 | `(&a + &b)?`, `(&a * 2.0)?`, `(2.0 * &a)?`, `-&a` |
| 重塑形状 | `t.try_reshape(&[2, 3])?` |
| 转置 | `t.try_transpose(0, 1)?` |
| 矩阵乘法 | `a.dot(&b)?` |
| 查看信息 | `t.dims()?`, `t.dim_const(-1)?`, `t.dtype()` |
| 线性层 | `Linear::with_dims(in, out, bias, dtype)` |
| 层链接 | `x.sequential(&[&fc1, &Relu, &fc2])?` |
| 激活函数 | `t.relu()?`, `t.softmax(-1)?` |
| 加载权重 | `model.load_state_dict(&sd, "")?` |
| 频谱图 | `x.stft().n_fft(512).hop(160).call()?` |
| 循环层 | `x.lstm().weight_ih(&w).weight_hh(&r).hidden_size(h).call()?` |
| 执行 | `t.realize()?` |
| 批量 realize | `Tensor::realize_batch([&a, &b])?` |
| 提取数据 | `t.to_vec::<f32>()?`, `t.to_ndarray::<f32>()?`, `t.item::<f32>()?` |

**惰性求值模式：**

1. 用各种操作构建计算图
2. 最后调用一次 `realize()`
3. Svod 统一优化并执行所有操作

**下一步：**

- [Op 手册](./architecture/op-bestiary) — IR 操作参考
- [执行流水线](./architecture/pipeline) — 编译过程详解
- [模式引擎](./architecture/optimizations/pattern-system) — 基于模式的重写
