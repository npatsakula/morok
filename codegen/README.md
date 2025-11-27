# morok-codegen

Backend code generation from optimized UOp graphs.

## Example

```rust
use morok_codegen::{Renderer, render};

let code = render(&kernel_graph, backend)?;
```

## Features

**Supported:**
- LLVM IR generation for CPU
- Renderer trait for backend abstraction

**Planned:**
- PTX renderer (CUDA)
- Metal renderer
- WebGPU (WGSL) renderer
- C-style renderer

## Testing

```bash
cargo test -p morok-codegen
```
