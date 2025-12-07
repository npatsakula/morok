# morok-device

Device abstraction with lazy buffer allocation, zero-copy views, and LRU caching.

## Example

```rust
use morok_device::{Buffer, BufferOptions, registry};
use morok_dtype::DType;

// CPU buffer (lazy allocation)
let cpu = registry::cpu();
let buf = Buffer::new(cpu, DType::float32(), &[1024], BufferOptions::default());

// CUDA buffer with unified memory (CPU-accessible)
let cuda = registry::cuda(0);
let opts = BufferOptions { cpu_accessible: true, ..Default::default() };
let unified = Buffer::allocate(cuda, DType::float32(), &[1024], opts)?;

// Zero-copy view
let view = buf.view(0, 512);

// Device-to-device copy
dst.copy_from(&src)?;
```

## Features

**Supported:**

- Lazy buffer allocation via `OnceLock`
- Zero-copy buffer views with offset tracking
- LRU allocation cache (per-size pooling)
- CPU allocator
- CUDA allocator (feature `cuda`)

**CUDA Optimizations:**
| Feature | Implementation | Notes |
|---------|---------------|-------|
| Unified memory | `cudaMallocManaged` | `cpu_accessible: true` |
| Device memory | `cuMemAlloc` | Faster GPU access |
| D2D copy | `memcpy_dtod` | Direct device-to-device |
| H2D/D2H copy | `memcpy_htod/dtoh` | Host transfers |
| Zero-init | `memset_zeros` | Stream-based |

**Planned:**

- Metal allocator
- WebGPU allocator
- Multi-GPU peer access
- Custom stream management
- Pinned host memory

## Copy Matrix

All combinations supported:

| | CPU | CudaDevice | CudaUnified |
|--|-----|------------|-------------|
| **CPU** | slice | H2D | slice |
| **CudaDevice** | D2H | D2D | D2D |
| **CudaUnified** | slice | D2D | slice |

## Device Registry

```rust
registry::cpu()              // CPU allocator
registry::cuda(0)            // CUDA device 0
registry::get_device("CUDA:1")  // Parse string

DeviceSpec::parse("cuda:0")  // Case-insensitive parsing
spec.canonicalize()          // → "CUDA:0"
```

## Testing

```bash
cargo test -p morok-device
cargo test -p morok-device --features cuda  # with CUDA
```
