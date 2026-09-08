---
sidebar_label: AM 驱动
---

# AM 驱动（用户态）

**AM** 驱动是第二个 [`AmdIface`](./overview.md) 后端，它直接驱动
GPU 的 PCI BAR，彻底绕过内核的 `amdgpu`/KFD 驱动。它
是 tinygrad 用户态 AM 驱动的移植。其动机很具体：在单 XCC 的
gfx11+ 硬件上，无锁的 [多队列调度](./queues-and-dispatch.md) 路径会把 CP
微引擎停在内核 MES 固件无法抢占的等待中，从而把它卡进一次
不可恢复的复位。内核调度器是那个共同的弱点——正是同一类故障
迫使通道池在每一种硬件上都保持保守。如果我们拥有 GPU——页表、
固件、调度——内核就永不处于调度路径中，也就无法被卡死。

:::caution[开发中——尚不可选]
本页同时记录当下存在什么，以及其余部分的路线图。
**`SVOD_AMD_BACKEND=am` 目前会返回错误**（`device.rs` 只接受
`kfd`）：尚无 AM 类型实现 [`AmdIface`](./overview.md) 接缝，因此
今天能够触及的启动只能通过 `am_*` 示例驱动，而非通过
`AmdDevice`。已存在的部分验证**到引擎交接为止**；尚无 GPU 引擎
被确认在目标上消费工作（见 [VF 边界](#the-vf-boundary)）。下面各节
明确标注每个部件的状态。
:::

代码位于 `device/src/amd/am/` 之下。它在**每一台 Unix 宿主上**编译
（`cfg(unix)`，与后端其余部分一样——见 [运行时检测的提供者
模型](./overview.md)），因此始终被类型检查、lint 和单元测试——
后端在*运行时*选择，绝不藏在一个可能腐烂的 cargo feature 之后。

---

## 目标硬件：一块 CDNA3 SR-IOV VF（gfx9.4.3）

该驱动的目标是一块 **CDNA3** GPU——**gfx9.4.3**，SPX 模式下的
8 个 XCC——并且特指其 **SR-IOV 虚拟功能（Virtual Function，VF）**形态
（该 GPU 是一个被透传进 KVM 客户机的 VF）。`AmDev::open` 会直接拒绝
其他一切：非 VF 的功能，或 major.minor 不是 `(9, 4)` 的 GC 版本，
都会快速失败（`device/src/amd/am/dev.rs`）。gfx1151（RDNA3.5）不再是
*目标*，但 gfx11 的 arch 分支仍保持已实现且有单元测试——而且正是
它的页表几何与 palloc 范围辅助函数被 gfx9 路径复用。

> 启动用硬件：一块 AMD Instinct MI300X（gfx942 / GC 9.4.3）的 SR-IOV VF。
> `AmDev::open` 不接受其他任何东西。

**VF**（而非裸金属）这一身份是决定性的约束，它决定了
整个驱动的形态：

- **GC MMIO 是宿主门控的。** 每一次对 GC 寄存器的*直接*读取都返回
  `0xffffffff`。所有 GC / GCVM 寄存器访问都必须**经由 RLC 间接进行**
  （RLCG 路径）——把值暂存进 RLC scratch，触发 `RLC_SPARE_INT`，
  轮询完成。
- **VRAM/discovery 在被授予前是门控的。** 帧缓冲区（以及由此而来的
  IP-discovery 表）在宿主 **GIM**（SR-IOV 宿主驱动）通过一次
  **mailbox 握手**授予访问权之前都不可读，因此该握手会运行在
  discovery *之前*。
- **宿主 PF 拥有那些特权子系统：** PSP、SMU、时钟、固件 /
  world-switch、L2 缓存配置、系统 aperture，以及——关键地——
  **doorbell aperture 路由**。AM 只编程客户机被允许触碰的那些每 VF 状态
  （页表 context0、每引擎失效范围、TLB 刷新、ring/queue MQD），
  这恰如内核的 `*_v*` IP 代码在 `amdgpu_sriov_vf` 下跳过这些块一样。

这与 tinygrad 的 AM 相反，后者是**仅裸金属**的（它 unbind
`amdgpu` 并拥有整张设备）。VF 形态是一个不同的驱动：mailbox
+ RLCG 间接寄存器访问 + 仅每 VF 的 hub 编程。

---

## 当下存在什么

凡是纯逻辑之处，一切都**在没有 GPU 的情况下编译并单元测试**；
面向硬件的部分还额外通过 `device/examples/am_*.rs` 程序在活动的 VF 上
验证。页表以一个可注入的 `PhysMem` trait 作为后备存储（测试中是一个普通缓冲区，
真实驱动中是 BAR 映射的 VRAM）。

| 分组 | 模块 | 它实现什么 | 状态 |
|---|---|---|---|
| **Discovery** | `pci.rs`, `discovery.rs` | sysfs BAR mmap（BAR0 VRAM / BAR2 doorbell / BAR5 MMIO）、配置空间读写、带边界检查的 IP-discovery 解析器（每 XCC 段基址，`gc_info` v1/v2） | **HW 验证**；discovery 解析器有单元测试 |
| **寄存器访问** | `regaccess.rs`, `rlcg.rs`, `mailbox.rs`, `regs.rs`, `regs_gen.rs` | mxgpu VF↔GIM mailbox 握手、RLCG 间接 GC/GCVM 读写（每 XCC）、MMIO/RLCG 路由器、vendored 寄存器表 | **HW 验证**；寄存器表的选择/编码逻辑有单元测试 |
| **内存（GMMU）** | `mm/{tlsf,pagetable,manager,mod}.rs` | TLSF VA/PA/页表分配器、4 级/48 位遍历、gfx9 **与** gfx11 的 PTE/PDE 编码、大页选择、表回收、`valloc`/`vfree` | **完成** + 测试（PTE 写路径已硬件演练） |
| **GMC 启动** | `ip/gmc.rs` | 编程两个 hub 的 context0（start/end/base + CNTL）、MX_L1_TLB 使能、每引擎失效范围、ENG17 TLB 刷新、HDP 刷新、故障状态解码 | **HW 验证**到 context 编程级别 |
| **GFX 启动** | `ip/gfx.rs` | 使能 MEC（icache 失效、golden `GB_ADDR_CONFIG`、doorbell 范围、unhalt）、构建一个 v9 compute MQD、激活 HQD（`CP_HQD_ACTIVE=1`）、`WRITE_DATA` PM4 | **MEC HQD 激活**；队列尚不运行 |
| **SDMA 启动** | `ip/sdma.rs` | unhalt F32、编程 RB base/rptr/wptr + doorbell、提交 + `wait_idle` | **ring 已编程**；引擎尚不消费 |
| **编排器** | `dev.rs` | `AmDev::open` = mailbox → discovery → GMMU → GMC context0 → flush；`valloc`、`vram_read/write`、`release` | **HW 验证**至 GMC |

### GMMU 与 gfx9

页表几何是 **4 级 / 48 位**（`va_shifts = [12, 21, 30, 39]`），
一种**跨 gfx9/11/12 共享**的形状——因此几何本身不随 arch 分支。
只有叶 PTE 编码（尤其是 MTYPE 内存类型字段）才是 arch 特定的，而
**gfx9（CDNA）与 gfx11（RDNA3）现已实现并单元测试**——
gfx9 把 MTYPE 放在第 57–58 位，在 PDB1 表项上置 `bfs`、在 PDB0 表项上置
translate-further 位，并用 `PDE_PTE` 标记 PDB1/PDB2 的叶（一个 2 MiB 的
PDB0 叶恰恰是 translate-further 的*缺席*）。**gfx12 是唯一
剩下的 `unimplemented!`**（常量已捕获，尚未经过硬件验证；
有一个测试断言它 panic）。`MemoryManager` 运行三个 TLSF 子分配器
（VA 空间、物理 VRAM、页表池），并以 `Inspect` / `Create` /
`Free` 模式遍历表，在 unmap 时回收空表。

### 寄存器表是生成一次，然后 vendored

tinygrad 是一个有时缺席的子模块，因此构建绝不能依赖它。
取而代之，`device/tools/gen_am_regs.py` 在添加或更新一个 arch 时
被**手动**运行：它解析 tinygrad 的 `autogen/am/regs.py` 并发出已提交的
`am/regs_gen.rs`。`regs.rs` 只是 `include!` 它。在启动时，正确的表由
发现到的 `ip_ver` 选定（`select` 挑选共享同一 major 的最大版本 `≤ ip_ver`
——tinygrad 的 `import_module` 规则）。已提交的表如今同时覆盖
gfx9.4.3/CDNA3 集（`gc 9.4.3`、`mmhub 1.8.0`、`osssys 4.4.2`、
`sdma 4.4.2`、`nbio 7.9.0`、`hdp 4.4.2`、`mp 11.0.0`/`13.0.0`）与 gfx11.5.0
集。添加一个 arch 就是拓宽生成器的模块列表并重新运行它——没有
构建或运行时逻辑改动。

---

## VF 边界 {#the-vf-boundary}

这就是当前启动止步的那道墙。客户机能**编程**引擎，
但无法**驱动**它们，因为把 ring 写指针送达命令处理器的那个
doorbell aperture 归 **PF 所有**。从 VF 启用它（写那些 `_PF`
BIF doorbell-access 寄存器）会卡住 VF↔GIM mailbox，并需要一次完整的
VM 重启——因此 `enable_doorbell_aperture` 在 `ip/gfx.rs` 中存在，
但被明确标注为**在 VF 上不可调用**。

具体后果，二者都由示例复现：

- **MEC compute 队列激活但不执行**（`am_compute`）：HQD
  报告 `CP_HQD_ACTIVE = 1`，但 `WRITE_DATA` 数据包始终没有把它的哨兵值
  写入 VRAM——CP 从未看到 doorbell。
- **SDMA ring 已编程但不消费**（`am_sdma`）：读指针保持
  卡住；MM-hub 页表遍历故障仍被门控。

所以今天的 AM **HW 验证到引擎交接为止**——discovery、所有权、
GMMU 和 GMC 都已在活动的 VF 上得到证实——而 KFD 仍是那个可工作的 VF
后端。跨越这条边界正是其余里程碑的主题。

---

## 今天什么在硬件上运行

每个 `am_*` 示例都是一个独立的启动检验器，在活动的 VF 上运行：

| 示例 | 它证明了什么 | 状态 |
|---|---|---|
| `am_discovery` | BAR map + IP discovery（8× GC 9.4.3、SDMA、AID），只读——与一个已绑定的 `amdgpu` 共存 | **可工作** |
| `am_own` | mailbox 授予 + RLCG scratch 回声 + 全部 8 个 XCC 上非门控的 `GRBM_STATUS` | **可工作** |
| `am_gmc` | GC + MM context0 已编程；全部 8 个 XCC 上 ENG17 TLB 刷新 ACK；无保护故障被锁存 | **可工作** |
| `am_sdma` | SDMA ring 设置 + 提交 | ring 已编程，**引擎不消费** |
| `am_compute` | MEC 使能 + MQD 激活 + `WRITE_DATA` | **HQD 激活**，队列不执行 |

---

## 还推迟了什么

那些特权、PF 拥有的子系统**不在代码树中**——在 VF 上它们
由 GIM 拥有，客户机无事可做；在裸金属上它们则是最后、风险最高的
移植：

- **PSP 固件加载**——sOS bootloader 握手 / TMR / 每 IP 固件
  加载。在 VF 上由 GIM 拥有。
- **SMU / 时钟**——电源与时钟管理。在 VF 上由 GIM 拥有。
- **中断处理器（IH）**——不存在 `ip/ih.rs`；OSSSYS 寄存器表
  已 vendored 但未使用。启动改为轮询，而非接收中断。
- **`AmIface` 接缝实现者**——尚无 AM 类型实现
  [`AmdIface`](./overview.md)，因此 AM 无法被选为设备后端；
  `AmDev` 仅可通过示例触及。

---

## 路线图

工作被分阶段为里程碑，每个都可在活动的 VF 上独立测试
（并且，对于那些 PF 拥有的块，以裸金属 tinygrad AM 作为检验器）。早先的里程碑
均已实现；完整的 AM 端到端集成尚属未来工作。

一旦某个引擎消费工作且接缝接好，AM 便可
经由 `SVOD_AMD_BACKEND=am` 选择，并原封不动地运行整个现有的上半部。
催生 AM 的那种诱发崩溃的并发那时就无法崩溃了——内核被绕过了。

---

## 为什么这很重要

AM 驱动是对那个固件卡死问题的真正答案，而把通道池夹紧
（[`SVOD_AMD_HW_QUEUES=1`](./queues-and-dispatch.md)）只是绕开了它。那些昂贵、
不需 GPU 的部分——GMMU、寄存器表、mailbox/RLCG 间接访问
机制——已经在活动的 VF 上构建并验证，而页表、GMC 与
所有权握手全都可工作。剩下的差距是一道硬件边界（PF 拥有的
doorbell aperture），而非设计上的。而且因为它接在同一道
[接缝](./overview.md) 之后——五个必需方法加三个默认实现的钩子方法——
当它落地时，调度、编译或图机制没有一样需要改动。
