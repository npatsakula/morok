---
sidebar_label: JIT Compiler
---

# JIT Compiler

Most ML compilers either link an entire LLVM toolchain into the binary — adding hundreds of megabytes of dependencies — or write temporary files to disk and `dlopen` the result. Morok does neither.

When a kernel needs to execute, Morok pipes the generated source through `clang` on stdin, receives a relocatable ELF object on stdout, parses it in-process, copies the machine code into an anonymous memory mapping, applies relocations, flips the page permissions to executable, and calls the function pointer directly. The whole process happens in memory — no temp files touch the disk, no shared libraries are loaded, and no LLVM installation is required beyond `clang` on the PATH.

This chapter describes how the CPU JIT loader works. GPU backends (CUDA, Metal, etc.) use their respective driver APIs for compilation and dispatch, and will be documented separately as they are added.

## Pipeline

```text
C source / LLVM IR
       │
       ▼
 clang -c (stdin → stdout)
       │
       ▼
  ELF .o bytes (in memory)
       │
       ▼
 Parse sections (object crate)
       │
       ▼
 Anonymous mmap + copy sections
       │
       ▼
 Apply relocations (arch-specific)
       │
       ▼
 mprotect(PROT_READ | PROT_EXEC)
       │
       ▼
 Flush I-cache (non-x86_64)
       │
       ▼
 Call function pointer via libffi
```

Both the **Clang** backend (C source via `-x c`) and the **LLVM** backend (LLVM IR text via `-x ir`) share this loader. The only difference is the clang input language flag.

:::tip Fallback mode
For debugging or platforms where the custom ELF loader doesn't work, the `dlopen-fallback` Cargo feature switches to a traditional pipeline: `clang -shared` writes a `.so` to a temp directory, which is loaded via `dlopen`. This is slower (disk I/O + dynamic linker overhead) but more portable.
:::

## Supported Architectures

| Architecture | Target triple | Compile flag | I-cache | Notes |
|---|---|---|---|---|
| **x86_64** | `x86_64-none-unknown-elf` | `-march=native` | Coherent | AMD64, Intel 64 |
| **aarch64** | `aarch64-none-unknown-elf` | `-march=native` | `__clear_cache` | Apple Silicon, Ampere, Graviton |
| **riscv64** | `riscv64-none-unknown-elf` | `-march=rv64gc` | `__clear_cache` | RV64I + M + A + F + D + C extensions |
| **loongarch64** | `loongarch64-none-unknown-elf` | `-march=native` | `__clear_cache` | Loongson 3A5000+ |
| **ppc64le** | `powerpc64le-none-unknown-elf` | `-mcpu=native` | `__clear_cache` | ELFv2 ABI, little-endian only |

Architecture detection is automatic via `std::env::consts::ARCH` at runtime — no compile-time feature flags needed.

### Relocation Support

The loader implements a minimal ELF relocator for each architecture. It handles the relocation types that `clang -c -O2` actually emits for small, self-contained compute kernels — not a full linker.

**x86_64** — PC-relative (`R_X86_64_PC32`, `PLT32`, `GOTPCRELX`, `REX_GOTPCRELX`), absolute 32/64-bit (`R_X86_64_32`, `32S`, `64`).

**aarch64** — 26-bit branches (`CALL26`, `JUMP26`), page-relative ADRP (`ADR_PREL_PG_HI21`), 12-bit page offsets with access-size shifts (`ADD_ABS_LO12_NC`, `LDST8/16/32/64/128_ABS_LO12_NC`).

**riscv64** — Call pairs (`CALL`, `CALL_PLT`), PC-relative split addressing with state tracking (`PCREL_HI20` + `PCREL_LO12_I/S`), absolute (`HI20`, `LO12_I/S`), branches (`BRANCH`, `JAL`), data (`32`, `64`). Linker relaxation hints (`RELAX`) are skipped.

**loongarch64** — 26-bit branches (`B26`), page-aligned split addressing (`PCALA_HI20`, `PCALA_LO12`), data (`32`, `64`). Linker relaxation hints (`RELAX`) are skipped.

**ppc64le** — 24-bit branches (`REL24`), TOC-relative addressing with `.TOC.` symbol lookup (`TOC16_HA`, `TOC16_LO`, `TOC16_LO_DS`, `TOC16`, `TOC16_HI`), PC-relative (`REL32`), absolute (`ADDR32`, `ADDR64`).

## Compilation Flags

The loader compiles with a bare-metal target to produce clean, self-contained ELF objects with no runtime dependencies:

| Flag | C backend | LLVM IR backend | Purpose |
|---|---|---|---|
| `-c` | yes | yes | Compile only (no linking) |
| `-O2` | yes | yes | Optimization level |
| `-march=native` | yes | yes | Use host CPU features |
| `-fPIC` | yes | yes | Position-independent code |
| `-ffreestanding` | yes | no | No hosted environment assumed |
| `-fno-math-errno` | yes | yes | Math builtins don't set errno |
| `-fno-stack-protector` | yes | yes | No stack canary overhead |
| `-nostdlib` | yes | no | No standard library |
| `-fno-ident` | yes | no | Suppress `.comment` section |
| `--target=<arch>-none-unknown-elf` | yes | no | Bare-metal target |
| `-funroll-loops` | no | yes | Aggressive loop unrolling |
| `-fvectorize` | no | yes | Loop vectorization |
| `-fslp-vectorize` | no | yes | SLP (straight-line) vectorization |

The C backend uses `__builtin_*` functions (e.g. `__builtin_sqrtf`, `__builtin_fmaf`) instead of `#include <math.h>`, so `-ffreestanding -nostdlib` works without losing math support — these are compiler intrinsics that lower to hardware instructions directly.

## External Symbol Resolution

If clang emits a call to an external function (rare — most math is handled by builtins), the loader resolves it via `dlsym(RTLD_DEFAULT, name)` at load time. This covers cases like `memcpy` or platform-specific libm symbols that clang might emit instead of inlining.

## Instruction Cache Coherence

On x86_64, the instruction and data caches are coherent — writing machine code to memory and jumping to it works without extra steps. On all other architectures, the loader calls `__clear_cache(start, end)` after `mprotect` to ensure the instruction cache sees the new code.
