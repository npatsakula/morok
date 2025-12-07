# morok-dtype

Type system for the morok compiler: scalar types, vectors, pointers, and images.

## Example

```rust
use morok_dtype::{DType, AddrSpace};

let f32_type = DType::float32();
let vec4 = f32_type.vec(4);
let ptr = f32_type.ptr(AddrSpace::Global);
```

## Features

**Supported:**

- Scalar types: Bool, Int8-64, UInt8-64, Float16/32/64, BFloat16, Index
- Vector types with configurable width
- Pointer types with address spaces (Global, Local, Register)
- Image types for texture-based computation

**Planned:**

- FP8 variants (e4m3, e5m2)
- Type promotion lattice

## Testing

```bash
cargo test -p morok-dtype
```
