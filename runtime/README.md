# morok-runtime

Kernel execution interface bridging codegen to hardware.

## Example

```rust
use morok_runtime::CompiledKernel;

let kernel = compile(code)?;
kernel.execute(&[buf_a.ptr(), buf_b.ptr(), buf_out.ptr()])?;
```

## Features

**Supported:**

- LLVM JIT compilation and execution
- CompiledKernel trait for backends

**Planned:**

- CUDA kernel execution
- Metal kernel execution
- Kernel caching
- Profiling hooks

## Testing

```bash
cargo test -p morok-runtime
```
