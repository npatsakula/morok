# svod-runtime

Kernel execution interface bridging codegen to hardware.

## Example

```rust
use svod_runtime::CompiledKernel;

let kernel = compile(code)?;
kernel.execute(&[buf_a.ptr(), buf_b.ptr(), buf_out.ptr()])?;
```

## Backends

| Backend | How it works | Feature |
|---------|-------------|---------|
| **LLVM** (default) | Compiles LLVM IR in-process through `libLLVM` (falls back to `clang -x ir`), loads via JIT ELF loader | always |
| **Clang** | Compiles C via `clang -c`, loads via JIT ELF loader | always |

Select at runtime: `SVOD_CPU_BACKEND=clang|llvm`

**Planned:**

- CUDA kernel execution
- Metal kernel execution

## Testing

```bash
cargo test -p svod-runtime
```
