# svod-codegen

Backend code generation from optimized UOp graphs.

## Example

```rust
use svod_codegen::{Renderer, render};

let code = render(&kernel_graph, backend)?;
```

## Backends

| Backend | Output | Feature | Default |
|---------|--------|---------|---------|
| **LLVM** (default) | LLVM IR text → in-process `libLLVM` (or `clang -x ir`) → JIT ELF loader | always | no |
| **Clang** | C source → `clang -c` → JIT ELF loader | always | yes |

Select at runtime via `SVOD_CPU_BACKEND` env var (`clang` or `llvm`).

**Planned:**

- PTX renderer (CUDA)
- Metal renderer
- WebGPU (WGSL) renderer

## Testing

```bash
cargo test -p svod-codegen
```
