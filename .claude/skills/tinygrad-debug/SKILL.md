---
name: tinygrad-debug
description: Extract IR, AST, generated code, and patterns from the Tinygrad codebase at submodules/tinygrad. Use when comparing Morok with Tinygrad, debugging pipeline differences, or understanding Tinygrad's implementation.
---

# Tinygrad Investigation

## Running Tinygrad Code

Always use `uv run` from the tinygrad directory:

```bash
cd submodules/tinygrad && uv run python -c "..."
```

## Extracting AST/IR

### Get scheduled AST

```python
from tinygrad import Tensor
from tinygrad.uop.ops import pyrender

t = Tensor([1.0, 2.0, 3.0, 4.0])
result = t.sum()

# Get AST
schedule = result.schedule()
ast = schedule[-1].ast

# Pretty-print as reconstructible Python
print(pyrender(ast))

# Get linearized nodes
for node in ast.toposort():
    print(node.op, node.dtype, node.arg)
```

### UOp attributes

| Attribute | Purpose |
|-----------|---------|
| `node.op` | Operation type (Ops enum) |
| `node.dtype` | Data type |
| `node.arg` | Operation argument |
| `node.src` | Child nodes (tuple) |

## Extracting Generated Code

### Via DEBUG environment variable

| Level | Output |
|-------|--------|
| DEBUG=4 | Generated source code (C/Metal/CUDA) |
| DEBUG=5 | AST UOps before lowering |
| DEBUG=6 | Full lowered UOps |

```bash
# Metal code (default on Mac)
DEBUG=4 uv run python -c "from tinygrad import Tensor; print(Tensor([1,2,3,4]).sum().item())"

# C code (CPU backend)
CPU=1 DEBUG=4 uv run python -c "from tinygrad import Tensor; print(Tensor([1,2,3,4]).sum().item())"
```

### Programmatic extraction

```python
from tinygrad import Tensor, Device
from tinygrad.engine.realize import lower_schedule

Device.DEFAULT = 'CPU'
t = Tensor([1.0, 2.0, 3.0, 4.0]).sum()
sched, var_vals = t.schedule_with_vars()

for si, ei in lower_schedule(sched):
    if hasattr(ei, 'prg') and hasattr(ei.prg, 'p'):
        p = ei.prg.p
        print("Source:", p.src)      # C/Metal/CUDA code
        print("Name:", p.name)       # Kernel name
        print("UOps:", p.uops)       # Lowered UOps
```

### LLVM IR extraction

```python
from tinygrad.renderer.llvmir import LLVMRenderer

renderer = LLVMRenderer()
llvm_ir = renderer.render(p.uops)
print(llvm_ir)
```

## Key Tinygrad Files

| File | Purpose |
|------|---------|
| `tinygrad/codegen/late/expander.py` | UNROLL expansion, CONTRACT, do_expand |
| `tinygrad/codegen/late/linearizer.py` | Topological sort with priority |
| `tinygrad/codegen/late/devectorizer.py` | Devectorization transforms |
| `tinygrad/schedule/rangeify.py` | Range assignment, bufferization |
| `tinygrad/uop/ops.py` | UOp, Ops, UPat, PatternMatcher, AxisType |

## Inspecting PatternMatcher

```python
from tinygrad.codegen.late.expander import expander
from tinygrad.uop.ops import Ops

# List all patterns
for upat, fxn in expander.patterns:
    print(upat.op, fxn.__name__)

# Get patterns by Op type
expander.pdict[Ops.UNROLL]  # All patterns matching UNROLL

# UPat attributes
# upat.op       - Tuple of Ops matched
# upat.name     - Capture name
# upat.src      - Child patterns
# upat.location - (filename, lineno)
```

## Comparing Morok vs Tinygrad IR

1. **Tinygrad IR**:
```python
from tinygrad import Tensor
from tinygrad.uop.ops import pyrender
ast = Tensor([1,2,3,4]).sum().schedule()[-1].ast
print(pyrender(ast))
```

2. **Morok IR** (from AGENTS.md):
```bash
RUST_LOG=morok_tensor::realize=debug cargo test test_name 2>&1 | rg 'range assignment complete'
```

3. **Compare**: Look for differences in RANGE axis types, INDEX structure, REDUCE patterns.
