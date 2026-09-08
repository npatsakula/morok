---
sidebar_label: AM Driver
---

# The AM Driver (Userspace)

The **AM** driver is a second [`AmdIface`](./overview.md) backend that drives the
GPU's PCI BARs directly, bypassing the kernel `amdgpu`/KFD driver entirely. It
is a port of tinygrad's userspace AM driver. The motivation is concrete: on
single-XCC gfx11+ parts the lock-free
[multi-queue dispatch](./queues-and-dispatch.md) path can park CP micro-engines
in waits that the kernel's MES firmware cannot preempt, wedging it into an
unrecoverable reset. The kernel scheduler is the shared weakness — the same
class of failure is what forces the lane pool to stay conservative on every
part. If we own the GPU — page tables, firmware, scheduling — the kernel is
never in the dispatch path and can't be wedged.

:::caution[Work in progress — not yet selectable]
This page documents both what exists today and the roadmap for the rest.
**`SVOD_AMD_BACKEND=am` currently returns an error** (`device.rs` accepts only
`kfd`): no AM type implements the [`AmdIface`](./overview.md) seam yet, so the
bring-up reachable today is driven only through the `am_*` examples, not through
`AmdDevice`. What exists is validated **up to engine hand-off**; no GPU engine
is yet confirmed to consume work on the target (see [the VF
boundary](#the-vf-boundary)). The sections below mark each piece's status
explicitly.
:::

The code lives under `device/src/amd/am/`. It is compiled on **every Unix host**
(`cfg(unix)`, like the rest of the backend — see the [runtime-detected provider
model](./overview.md)), so it is always type-checked, linted, and unit-tested —
the backend is chosen at *runtime*, never behind a cargo feature that could rot.

---

## Target hardware: a CDNA3 SR-IOV VF (gfx9.4.3)

The driver targets a **CDNA3** GPU — **gfx9.4.3**, 8 XCCs in SPX mode — and
specifically its **SR-IOV Virtual Function** flavor (the GPU is a VF passed into
a KVM guest). `AmDev::open` hard-rejects anything else: a non-VF function, or a
GC version whose major.minor isn't `(9, 4)`, fails fast
(`device/src/amd/am/dev.rs`). gfx1151 (RDNA3.5) is no longer the *target*, but
the gfx11 arch branch stays implemented and unit-tested — and its page-table
geometry and palloc-range helpers are what the gfx9 path reuses.

> Bring-up hardware: an SR-IOV VF of an AMD Instinct MI300X (gfx942 /
> GC 9.4.3). `AmDev::open` accepts nothing else.

Being a **VF** (rather than bare metal) is the defining constraint, and it shapes
the whole driver:

- **GC MMIO is host-gated.** Every *direct* read of a GC register returns
  `0xffffffff`. All GC / GCVM register access must go **indirectly through the
  RLC** (the RLCG path) — stage the value into RLC scratch, kick `RLC_SPARE_INT`,
  poll for completion.
- **VRAM/discovery is gated until granted.** The framebuffer (and thus the
  IP-discovery table) is unreadable until the host **GIM** (the SR-IOV host
  driver) grants access via a **mailbox handshake**, which therefore runs
  *before* discovery.
- **The host PF owns the privileged subsystems:** PSP, SMU, clocks, firmware /
  world-switch, L2 cache config, the system aperture, and — critically — the
  **doorbell aperture routing**. AM programs only the per-VF state the guest is
  allowed to touch (page-table context0, per-engine invalidation ranges, TLB
  flushes, ring/queue MQDs), exactly as the kernel's `*_v*` IP code skips these
  blocks under `amdgpu_sriov_vf`.

This is the inverse of tinygrad's AM, which is **bare-metal only** (it unbinds
`amdgpu` and owns the whole device). The VF flavor is a different driver: mailbox
+ RLCG indirect register access + per-VF-only hub programming.

---

## What exists today

Everything is **compiled and unit-tested without a GPU** where it is pure logic;
the hardware-facing pieces are additionally validated on the live VF through the
`device/examples/am_*.rs` programs. The page tables are backed by an injectable
`PhysMem` trait (a plain buffer in tests, BAR-mapped VRAM in the real driver).

| Group | Module(s) | What it does | Status |
|---|---|---|---|
| **Discovery** | `pci.rs`, `discovery.rs` | sysfs BAR mmap (BAR0 VRAM / BAR2 doorbell / BAR5 MMIO), config-space r/w, bounds-checked IP-discovery parser (per-XCC segment bases, `gc_info` v1/v2) | **HW-validated**; the discovery parser is unit-tested |
| **Register access** | `regaccess.rs`, `rlcg.rs`, `mailbox.rs`, `regs.rs`, `regs_gen.rs` | the mxgpu VF↔GIM mailbox handshake, RLCG indirect GC/GCVM r/w (per-XCC), the MMIO/RLCG router, vendored register tables | **HW-validated**; the register-table select/encode logic is unit-tested |
| **Memory (GMMU)** | `mm/{tlsf,pagetable,manager,mod}.rs` | TLSF VA/PA/page-table allocators, 4-level/48-bit walk, gfx9 **and** gfx11 PTE/PDE encoding, huge-page selection, table reclaim, `valloc`/`vfree` | **Done** + tests (PTE write path HW-exercised) |
| **GMC bring-up** | `ip/gmc.rs` | program both hubs' context0 (start/end/base + CNTL), MX_L1_TLB enable, per-engine invalidation ranges, ENG17 TLB flush, HDP flush, fault-status decode | **HW-validated** to context-program level |
| **GFX bring-up** | `ip/gfx.rs` | enable the MEC (icache invalidate, golden `GB_ADDR_CONFIG`, doorbell range, unhalt), build a v9 compute MQD, activate the HQD (`CP_HQD_ACTIVE=1`), `WRITE_DATA` PM4 | **MEC HQD activates**; queue does not yet run |
| **SDMA bring-up** | `ip/sdma.rs` | unhalt the F32, program RB base/rptr/wptr + doorbell, submit + `wait_idle` | **ring programmed**; engine does not yet consume |
| **Orchestrator** | `dev.rs` | `AmDev::open` = mailbox → discovery → GMMU → GMC context0 → flush; `valloc`, `vram_read/write`, `release` | **HW-validated** through GMC |

### The GMMU and gfx9

The page-table geometry is **4-level / 48-bit** (`va_shifts = [12, 21, 30, 39]`),
a shape **shared across gfx9/11/12** — so the geometry itself does not branch on
arch. Only the leaf PTE encoding (notably the MTYPE memory-type field) is
arch-specific, and **both gfx9 (CDNA) and gfx11 (RDNA3) are now implemented
and unit-tested** — gfx9 puts MTYPE at bits 57–58, sets `bfs` on PDB1 table
entries and the translate-further bit on PDB0 table entries, and marks PDB1/PDB2
leaves with `PDE_PTE` (a 2 MiB PDB0 leaf is the *absence* of translate-further).
**gfx12 is the only remaining `unimplemented!`** (constants captured, not yet
hardware-validated; a test asserts it panics). The `MemoryManager` runs three
TLSF sub-allocators (VA space, physical VRAM, page-table pool) and walks the
table in `Inspect` / `Create` / `Free` modes, reclaiming empty tables on unmap.

### Register tables are generated-once, then vendored

tinygrad is a sometimes-absent submodule, so the build must never depend on it.
Instead `device/tools/gen_am_regs.py` is run **manually** when adding or updating
an arch: it parses tinygrad's `autogen/am/regs.py` and emits the committed
`am/regs_gen.rs`. `regs.rs` just `include!`s it. At boot the right table is
chosen by the discovered `ip_ver` (`select` picks the greatest version `≤ ip_ver`
sharing the same major — tinygrad's `import_module` rule). The committed tables
now cover both the gfx9.4.3/CDNA3 set (`gc 9.4.3`, `mmhub 1.8.0`, `osssys 4.4.2`,
`sdma 4.4.2`, `nbio 7.9.0`, `hdp 4.4.2`, `mp 11.0.0`/`13.0.0`) and the gfx11.5.0
set. Adding an arch is widening the generator's module list and re-running it — no
build or runtime logic change.

---

## The VF boundary

This is the wall the bring-up currently stops at. The guest can **program** the
engines but cannot **drive** them, because the doorbell aperture that delivers a
ring's write-pointer to the command processor is **PF-owned**. Enabling it from
the VF (writing the `_PF` BIF doorbell-access registers) wedges the VF↔GIM
mailbox and requires a full VM reboot — so `enable_doorbell_aperture` exists in
`ip/gfx.rs` but is explicitly marked **do-not-call on the VF**.

The concrete consequences, both reproduced by the examples:

- **The MEC compute queue activates but doesn't execute** (`am_compute`): the HQD
  reports `CP_HQD_ACTIVE = 1`, but a `WRITE_DATA` packet never lands its sentinel
  in VRAM — the CP never sees the doorbell.
- **The SDMA ring is programmed but doesn't consume** (`am_sdma`): the read
  pointer stays stuck; the MM-hub page-table walk faults are still gated.

So AM today is **HW-validated up to engine hand-off** — discovery, ownership,
GMMU, and GMC are proven on the live VF — and KFD remains the functioning VF
backend. Crossing this boundary is what the remaining milestones are about.

---

## What runs on hardware today

Each `am_*` example is a standalone bring-up oracle, run on the live VF:

| Example | What it proves | Status |
|---|---|---|
| `am_discovery` | BAR map + IP discovery (8× GC 9.4.3, SDMA, AIDs), read-only — coexists with a bound `amdgpu` | **works** |
| `am_own` | mailbox grant + RLCG scratch echo + non-gated `GRBM_STATUS` on all 8 XCC | **works** |
| `am_gmc` | GC + MM context0 programmed; ENG17 TLB-flush ACK on all 8 XCC; no protection fault latched | **works** |
| `am_sdma` | SDMA ring setup + submit | ring programmed, **engine does not consume** |
| `am_compute` | MEC enable + MQD activate + `WRITE_DATA` | **HQD activates**, queue does not execute |

---

## What is still deferred

The privileged, PF-owned subsystems are **absent from the tree** — on a VF they
are owned by GIM and there is nothing for the guest to do; on bare metal they are
the last, highest-risk port:

- **PSP firmware load** — the sOS bootloader handshake / TMR / per-IP firmware
  load. GIM-owned on the VF.
- **SMU / clocks** — power and clock management. GIM-owned on the VF.
- **The interrupt handler (IH)** — no `ip/ih.rs` exists; the OSSSYS register table
  is vendored but unused. Bring-up polls instead of taking interrupts.
- **The `AmIface` seam implementor** — no AM type implements
  [`AmdIface`](./overview.md) yet, so AM cannot be selected as a device backend;
  `AmDev` is reachable only through the examples.

---

## Roadmap

The work is staged as milestones, each independently testable on the live VF
(and, for the PF-owned blocks, against bare-metal tinygrad AM as the oracle).
Earlier milestones are implemented; full AM end-to-end integration is future
work.

Once an engine consumes work and the seam is wired, AM becomes selectable via
`SVOD_AMD_BACKEND=am` and runs the entire existing upper half unchanged. The
crash-inducing concurrency that motivated AM can't crash then — the kernel is
bypassed.

---

## Why this matters

The AM driver is the real answer to the firmware-wedge problem that clamping the
lane pool ([`SVOD_AMD_HW_QUEUES=1`](./queues-and-dispatch.md)) only sidesteps.
The expensive, GPU-free parts — the GMMU, the register tables, the mailbox/RLCG
indirect-access machinery — are built and validated on the live VF, and the page
tables, GMC, and ownership handshake all work. The remaining gap is a hardware
boundary (the PF-owned doorbell aperture), not a design one. And because it
slots in behind the same [seam](./overview.md) — five required methods plus
three defaulted hooks — none of the dispatch, compile, or graph machinery has to
change when it lands.
