# morok-codegen

Backend code generation from optimized UOp graphs.

## Example

```rust
use morok_codegen::{Renderer, render};

let code = render(&kernel_graph, backend)?;
```

## Backends

| Backend | Output | Feature | Default |
|---------|--------|---------|---------|
| **Clang** | C source → `clang -c` → JIT ELF loader | always | yes |
| **LLVM JIT** | LLVM IR text → `clang -x ir` → JIT ELF loader | always | no |
| **MLIR** | MLIR (arith/scf/llvm dialects) → MLIR ExecutionEngine | `mlir` | no |

Select at runtime via `MOROK_CPU_BACKEND` env var (`clang`, `llvm`, `mlir`).

**Planned:**

- PTX renderer (CUDA)
- Metal renderer
- WebGPU (WGSL) renderer

## Testing

```bash
cargo test -p morok-codegen
```
