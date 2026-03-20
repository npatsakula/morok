# morok-runtime

Kernel execution interface bridging codegen to hardware.

## Example

```rust
use morok_runtime::CompiledKernel;

let kernel = compile(code)?;
kernel.execute(&[buf_a.ptr(), buf_b.ptr(), buf_out.ptr()])?;
```

## Backends

| Backend | How it works | Feature |
|---------|-------------|---------|
| **Clang** (default) | Compiles C via `clang -c`, loads via JIT ELF loader | always |
| **LLVM JIT** | Compiles LLVM IR via `clang -x ir`, loads via JIT ELF loader | always |
| **MLIR** | Lowers MLIR dialects to LLVM, JIT via MLIR ExecutionEngine | `mlir` |

Select at runtime: `MOROK_CPU_BACKEND=clang|llvm|mlir`

**Planned:**

- CUDA kernel execution
- Metal kernel execution

## Testing

```bash
cargo test -p morok-runtime
```
