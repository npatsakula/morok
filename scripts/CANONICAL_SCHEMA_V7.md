# Canonical UOp Schema Version 6

Schema v6 is the non-verbose parity contract for Tinygrad at
`8c8b43de62515abe6c820b1de5aa26b30f48e43a` and Svod. A graph document has
exactly `schema_version`, `stage`, ordered `roots`, and `nodes`. Node IDs are dense
graph-local integers. Every source, symbolic shape, PROGRAM dimension, and
PROGRAM variable refers to a smaller existing ID. Node table order is not
semantic; `id` is. Each node has exactly `id`, `op`, `dtype`, `shape`, typed
`arg`, and source-order-preserving `src`.

The dtype union has scalar, vector, pointer, and image variants. Svod emits all
four. The pinned Tinygrad revision has scalar `DType` only: vector lanes are
STACK/shape semantics, addresses are `uint64`, and image identity is a PARAM
shape convention. Therefore no strict cross-language fixture can claim a
Tinygrad vector/pointer/image dtype. An unknown Tinygrad dtype is an error.

Typed args currently emitted are CONST, CAST/BITCAST dtype, MSELECT/GETTUPLE
index, SPECIAL name, GETADDR/COPY device, PARAM/BUFFER `ParamArg`, SLICE size,
STAGE options, PERMUTE/FLIP axes, REDUCE/ALLREDUCE, RANGE, VCONST, DEFINE_VAR,
CALL/FUNCTION, SINK `KernelInfo`, WMMA, verbose-only SOURCE/BINARY, INS,
CONTIGUOUS hints, CUSTOM/CUSTOMI, CUSTOM_FUNCTION, and PROGRAM. Other ops must
have no arg. Exact float bits, range paths, source order, gates, ABI slots,
launch dimensions, and targets are preserved.

PARAM and BUFFER slots are authored values. The sole normalization is the
cross-language `-1` sentinel for `usize::MAX`; stage names and numeric high bits
never imply a slot namespace. Deterministic schedule-local BUFFER slots retain
their high bit. Their proven `BUFFER` origin is represented explicitly in the
schedule artifact, and any remaining slot mismatch stays visible as EVID-01B.

PAD is the one movement operation with normalized typed metadata. Svod stores
logical `(begin-padding, end-padding)` while Tinygrad stores `(begin-padding,
output-extent)`. Canonical PAD stores nonnegative logical `begin` and `end`
lists and keeps only the data operand in `src`; unsupported symbolic or negative
extents fail explicitly. This retains padding extent and validity semantics
without comparing representation-only metadata UOps.

Svod's UOp-valued PROGRAM dimensions are normalized to integer or float values
when constant because Tinygrad stores those dimensions as Python numbers.
Unsigned values through `i64::MAX` normalize to `int`; larger values use `uint`.
Python integers outside the combined `i64`/`u64` range are rejected.
Non-constant dimensions and PROGRAM variables refer to graph-local node IDs.
Tinygrad's `AddrSpace.ALU` is normalized to Svod's `None` variable address
space, and CPU device spelling is normalized to uppercase. PROGRAM targets
accept only Tinygrad's `Target.device`; nonempty `renderer`, `arch`, `interface`,
or `indices` are rejected because Svod `DeviceSpec` cannot represent them.

WMMA preserves the pinned common tuple: dimensions, input and accumulator
dtypes, device, thread count, and A/B/C upcast axes. Svod-only WMMA `name`
and `reduce_axes` are not parity fields. CALL/FUNCTION preserves
Tinygrad `aux` only when it is a sequence of strings; callable `grad_fxn` and
Svod-only `grad_tag` are rejected. SINK preserves the common `name`,
`opts_to_apply`, `applied_opts`, and `dont_use_locals`; nondefault
Tinygrad-only `axis_types`, `estimates`, and `beam` are rejected.
ANSI presentation is stripped from names, but semantic name text is retained.
Only Tinygrad's implicit `KernelInfo.name == "test"` at `kernel_ast` and
`scheduled` normalizes to Svod's absent generated name; explicit names at other
stages are never erased.

SOURCE and BINARY executable identities were introduced after canonical v6 and
are therefore rejected in non-verbose oracle documents rather than silently
erased. Verbose diagnostics may show their payloads but are never parity inputs.
Tuple devices and tuple targets are rejected because Svod has no multi-device
`DeviceSpec`. UNIQUE/LUNIQUE are rejected rather than serializing allocation
IDs. SOURCE/BINARY are rejected in non-verbose mode. Verbose documents may add runtime
IDs, tags, backend dtype text, and implementation-specific content hashes; the
comparator rejects verbose documents as parity inputs. Unsupported metadata is
an explicit typed error, never `repr` or silent erasure.

## Production Evidence

`tensor/examples/canonical_stages.rs` and
`scripts/tinygrad-canonical.py --production-stage` start from the same
one-dimensional tensor expression and
run cumulatively through rangeification, kernel callification/schedule
extraction, optimization, expansion, coalescing, gating, PROGRAM creation, and
linearization. `scheduled` is a distinct schedule document with exactly
`schema_version`, `stage`, ordered `items`, and ordered top-level `output_slots`.
Svod serializes `create_pre_schedule` descriptors and concrete invocation order,
merged with current user variable values; Tinygrad serializes the CALL stream and
`var_vals` returned together by pinned `create_linear_with_vars`.
Each item records its order, callable descriptor index, complete canonical
`kernel_ast`, ordered buffers, AST output slots, predecessor item indices, and
sorted concrete bindings. A binding records `kind`, `slot`, `name`, canonical
`dtype`, signed 64-bit `value`, and whether it is a `schedule_loop` value; this
preserves symbolic PARAM identity rather than reducing bindings to names.
Buffers record argument/global position, authored
buffer slot, and proven PARAM or BUFFER origin. Top-level outputs refer to an
item and buffer position. This observes kernel identity, execution order, ABI
mapping, generated-buffer identity, AFTER/dependency ordering, outputs, and
schedule-loop bindings without relabeling the callified graph. Svod also applies
PADTO in the optimizer transaction, making `optimized` and `postrange` one
concrete boundary. The shared production expression has two chained kernels;
both Python self-test and Rust runner require item 1 to depend on item 0 and the
top-level output to select item 1 buffer 0.

Strict requested capture uses `SVOD_CAPTURE_CANONICAL_STAGE`,
`SVOD_CAPTURE_CANONICAL_LABEL`, and `SVOD_CAPTURE_CANONICAL_PATH`. A matching
request panics on graph serialization, JSON serialization, or file-write
failure. `SVOD_DUMP_CANONICAL_STAGE` remains a best-effort diagnostic stream.

The direct corpus and production fixture have strict parity, including every
captured production stage from `tensor` through `linearized`. The checked
mismatch manifest is empty. Ordinary script mode fails on any new mismatch.
`CANONICAL_RECORD_KNOWN_GAPS=1` succeeds only when both independent captures are
deterministic and exactly match the checked manifest.

Every mismatch diagnostic starts with SHA-256 hashes of both complete validated
documents. Hash input is deterministic JSON with recursively sorted object keys,
no insignificant whitespace, ASCII escaping, and rejected non-finite numbers.
Evidence mode therefore accepts only the checked-in full-document pair,
including fields in unaligned inserted/replaced nodes. The comparator self-test
mutates both operation and dtype inside an unmatched node and requires a new
signature.

The direct corpus exercises weak-int plus strong promotion and weak-float plus
strong commitment (including negative-zero bits) through public dtype
derivation. Those currently expose Svod's explicit commitment CAST and remain
EVID-01B evidence rather than being forced to agree. Strict direct fixtures cover
Invalid through WHERE, scalar and shaped STACK, structured BUFFER, scalar LOAD
and STORE, gated LOAD in the modern `(INDEX, zero alternate, gate)` layout,
separate mixed-validity INDEX coverage, COPY and ALLREDUCE metadata,
multi-output CALL, local/shared WMMA staging, range paths, typed logical PAD,
and rich PROGRAM metadata created through `ProgramInfo.from_sink` with dense
mixed storage/scalar slots, symbolic launch value, sizes, ABI lists, name, and
common target.

Padded reduction now uses the pinned tensor-form `REDUCE` representation, so no
exact expected-failure fixture remains. Symbolic FUNCTION result shape and real
production multi-output callification also compare strictly.
Pointer/image/vector dtype,
tuple-device/target, callable gradient identity, nondefault Tinygrad-only
KernelInfo/Target fields, UNIQUE/LUNIQUE, and non-verbose BINARY remain explicit
expected rejections as described above.
