---
sidebar_label: JIT 编译器
---

# JIT 编译器

大多数 ML 编译器要么将整个 LLVM 工具链链接到二进制文件中——增加数百兆字节的依赖——要么将临时文件写入磁盘再通过 `dlopen` 加载。Svod 两者都不需要。

当 kernel 需要执行时，Svod 通过 stdin 将生成的源代码传递给编译器，在 stdout 接收可重定位的 ELF 对象，在进程内解析，将机器码复制到匿名内存映射中，应用重定位，将页面权限切换为可执行，然后直接通过函数指针调用。整个过程在内存中完成——把代码交给编译器不需要临时文件，不会 `dlopen` 任何 JIT 产出的共享库，除了 PATH 中的 `clang` 之外也不需要任何 LLVM 安装。

C 后端的编译器就是 stdin/stdout 上的 `clang`。**LLVM IR 后端是默认的**（`SVOD_CPU_BACKEND=clang` 挑选另一个），它优先用 `libloading` 把 libLLVM 绑定在进程内，当该库加载不了、或 `SVOD_LLVM_INPROCESS=0` 说不要时，则回退到同一个 `clang -x ir` 子进程。无论走哪条路，下面的加载器看到的都是同样的 ELF 字节。

本章描述 CPU JIT 加载器的工作原理。GPU 后端通过同一个 `clang` 传输，但经由各自的驱动调度：见 [AMD 后端](./amd/overview.md)（KFD 直连）与 [CUDA 后端](./cuda/overview.md)（驱动 API；PTX 由驱动 JIT，装了 CUDA toolkit 时则是一个 `ptxas` cubin）。把一张模型图编译一次、再重放许多次的那层更上层的图包装 API，记录在 [JIT 图](../architecture/jit-graphs.md) 中。

## 流水线

```mermaid
flowchart TD
  A["C source / LLVM IR"] --> B["in-process libLLVM, or clang -c (stdin to stdout)"]
  B --> C["ELF .o bytes (in memory)"]
  C --> D["Parse sections (object crate)"]
  D --> E["Anonymous mmap + copy sections"]
  E --> F["Apply relocations (arch-specific)"]
  F --> G["mprotect(PROT_READ, PROT_EXEC)"]
  G --> H["Flush I-cache (non-x86_64)"]
  H --> I["Call function pointer via libffi"]
```

**Clang** 后端（C 源码，通过 `-x c`）和 **LLVM** 后端（LLVM IR 文本，通过 `-x ir`）共享同一个加载器。唯一区别是 clang 的输入语言标志。

:::tip[回退模式]
用于调试或自定义 ELF 加载器不工作的平台，Cargo feature `dlopen-fallback` 切换到传统流水线：`clang -shared` 将 `.so` 写入临时目录，通过 `dlopen` 加载。这较慢（磁盘 I/O + 动态链接器开销），但更具可移植性。
:::

## 支持的架构

| 架构 | Target triple | 编译标志 | I-cache | 备注 |
|---|---|---|---|---|
| **x86_64** | `x86_64-none-unknown-elf` | `-march=native` | 自动一致 | AMD64, Intel 64 |
| **aarch64** | `aarch64-none-unknown-elf` | `-mcpu=native` | `__clear_cache` | Apple Silicon, Ampere, Graviton |
| **riscv64** | `riscv64-none-unknown-elf` | `-march=rv64g` | `__clear_cache` | RV64I + M + A + F + D 扩展 |
| **loongarch64** | `loongarch64-none-unknown-elf` | `-march=native` | `__clear_cache` | 龙芯 3A5000+ |
| **ppc64le** | `powerpc64le-none-unknown-elf` | `-mcpu=native` | `__clear_cache` | ELFv2 ABI, 仅小端 |

在 ARM 上，`-march=native` 只设定基础 ISA 家族，因此 C 后端改为要求 `-mcpu=native`；上表的编译标志一列是该后端的，而 LLVM IR 后端在所有地方都传 `-march=native`。架构检测通过运行时 `std::env::consts::ARCH` 自动完成——无需编译时 feature flag。

### 重定位支持

加载器为每种架构实现了一个最小化的 ELF 重定位器。它处理 `clang -c -O2` 为小型自包含计算 kernel 实际生成的重定位类型——而非完整的链接器。

**x86_64** — PC 相对（`R_X86_64_PC32`、`PLT32`、`GOTPCRELX`、`REX_GOTPCRELX`），绝对 32/64 位（`R_X86_64_32`、`32S`、`64`）。

**aarch64** — 26 位分支（`CALL26`、`JUMP26`），当目标超出 ±128 MiB 范围时自动生成跳板，页相对 ADRP（`ADR_PREL_PG_HI21`），带访问大小移位的 12 位页偏移（`ADD_ABS_LO12_NC`、`LDST8/16/32/64/128_ABS_LO12_NC`）。

**riscv64** — 调用对（`CALL`、`CALL_PLT`），带状态跟踪的 PC 相对分离寻址（`PCREL_HI20` + `PCREL_LO12_I/S`），绝对（`HI20`、`LO12_I/S`），分支（`BRANCH`、`JAL`），数据（`32`、`64`）。链接器松弛提示（`RELAX`）被跳过。

**loongarch64** — 26 位分支（`B26`），页对齐分离寻址（`PCALA_HI20`、`PCALA_LO12`），数据（`32`、`64`）。链接器松弛提示（`RELAX`）被跳过。

**ppc64le** — 24 位分支（`REL24`），带 `.TOC.` 符号查找的 TOC 相对寻址（`TOC16_HA`、`TOC16_LO`、`TOC16_LO_DS`、`TOC16`、`TOC16_HI`），PC 相对（`REL32`），绝对（`ADDR32`、`ADDR64`）。

## 编译标志

加载器使用裸机 target 编译，生成干净、自包含、无运行时依赖的 ELF 对象：

| 标志 | C 后端 | LLVM IR 后端 | 用途 |
|---|---|---|---|
| `-c` | 是 | 是 | 仅编译（不链接） |
| `-O2` | 是 | 是 | 优化级别 |
| `-march=native` | 按架构（见上） | 是 | 使用宿主 CPU 特性 |
| `-fPIC` | 是 | 是 | 位置无关代码 |
| `-ffreestanding` | 是 | 否 | 不假设托管环境 |
| `-fno-math-errno` | 是 | 是 | 数学内建函数不设置 errno |
| `-fno-stack-protector` | 是 | 是 | 无栈保护开销 |
| `-nostdlib` | 是 | 否 | 无标准库 |
| `-fno-ident` | 是 | 否 | 抑制 `.comment` section |
| `--target=<arch>-none-unknown-elf` | 是 | 是 | 裸机 ELF target |
| `-ffixed-x18` | aarch64 macOS | aarch64 macOS | 保留平台寄存器 |
| `-funroll-loops` | 否 | 是 | 激进循环展开 |
| `-fvectorize` | 否 | 是 | 循环向量化 |
| `-fslp-vectorize` | 否 | 是 | SLP（直线代码）向量化 |

C 后端使用 `__builtin_*` 函数（如 `__builtin_sqrtf`、`__builtin_fmaf`）代替 `#include <math.h>`，因此 `-ffreestanding -nostdlib` 在不失去数学支持的情况下正常工作——这些是编译器内建函数，直接降低为硬件指令。

## 外部符号解析

如果 clang 生成了对外部函数的调用（很少——大部分数学由内建函数处理），加载器在加载时通过 `dlsym(RTLD_DEFAULT, name)` 解析。这涵盖了 `memcpy`，或者 clang 可能不做内联而直接生成的平台特定 libm 符号等情况。

### 分支跳板（aarch64、x86_64）

在 aarch64 上，`CALL26`/`JUMP26` 重定位将 PC 相对偏移编码在 26 位中，范围为 ±128 MiB；在 x86_64 上，`PC32`/`PLT32` 给出 ±2 GiB。一个长命的进程会自顶向下填满它的 mmap 区域，因此一块匿名的 JIT 映射最终会落到 libm 之流够不着的地方。

当一次直接分支够不到目标时，加载器会把它路由经由 mmap 末尾保留区域中的一个**跳板**（veneer，分支蹦床）：

```text
LDR X16, [PC, #8]   // 加载 64 位目标地址
BR  X16              // 间接跳转
.quad <address>      // 完整 64 位地址
```

x86_64 上的形式是 `MOVABS $target, %r11` + `JMP *%r11`，而且只有当补丁点前一个字节确实是一个 `call`/`jmp rel32` 操作码时才会走这条路——一处超出范围的 RIP 相对数据引用会响亮地失败，而不是被悄悄接过去。跳板空间在 mmap 分配之前就为每一个唯一的外部直接分支符号预留好，跳板本身也会去重，因此共用同一个符号的调用点共享同一个蹦床。

### 平台寄存器（aarch64）

在 macOS ARM 上，寄存器 `x18` 被保留为平台寄存器，内核会在一次上下文切换中把它踩掉。由于我们使用 `--target=aarch64-none-unknown-elf`（裸机）编译，编译器本来会将 `x18` 视为自由 GPR。`-ffixed-x18` 标志阻止了这一行为，避免 JIT 代码在 macOS 进程中运行时崩溃。Linux ARM 把 `x18` 当作普通 GPR，而 Windows ARM 不是 Svod 支持的 target。

## 指令缓存一致性

在 x86_64 上，指令缓存和数据缓存自动一致——将机器码写入内存并跳转执行无需额外步骤。在所有其他架构上，加载器在 `mprotect` 之后调用 `__clear_cache(start, end)` 以确保指令缓存看到新代码。
