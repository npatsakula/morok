#!/usr/bin/env bash
# Extract IR trees from each pipeline stage for a given test.
#
# Uses JSON structured logging (tracing-subscriber .json()) + rg + jaq.
# No intermediate buffers — cargo output streams directly through the pipe.
#
# Usage:
#   ./scripts/extract-ir.sh <test_name> [-p <package>] [-o <output>] [-t <target>]
#
# Examples:
#   ./scripts/extract-ir.sh test_sum_axis1_value -p morok-tensor
#   ./scripts/extract-ir.sh light_densenet121 -p morok-onnx -o /tmp/ir.txt
#   ./scripts/extract-ir.sh light_densenet121 -p morok-onnx -t optimizer  # only optimizer stages

set -euo pipefail

for cmd in rg jaq; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd is required" >&2; exit 1; }
done

usage() {
    cat >&2 <<'EOF'
Usage: extract-ir.sh <test_name> [-p <package>] [-o <output>] [-t <target>]

Arguments:
  test_name       Cargo test name filter (passed to `cargo test`)

Options:
  -p <package>    Cargo package filter (e.g. morok-tensor)
  -o <output>     Output file (default: ir_<test_name>.txt)
  -t <target>     Filter by target module (regex, e.g. "optimizer", "codegen")
  -h              Show this help
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage
[[ "$1" == "-h" || "$1" == "--help" ]] && usage

TEST_NAME="$1"; shift
PACKAGE=""
OUTPUT=""
TARGET_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p) PACKAGE="$2"; shift 2 ;;
        -o) OUTPUT="$2"; shift 2 ;;
        -t) TARGET_FILTER="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

OUTPUT="${OUTPUT:-ir_${TEST_NAME}.txt}"

RUST_LOG="morok_schedule::rangeify::transforms=debug,\
morok_schedule::rangeify::indexing=debug,\
morok_schedule::rangeify::kernel=debug,\
morok_schedule::optimizer=debug,\
morok_schedule::expand=debug,\
morok_schedule::devectorize=debug,\
morok_schedule::linearize=debug,\
morok_codegen=debug"

CARGO_ARGS=(test --release "$TEST_NAME" -- --nocapture --test-threads=1)
[[ -n "$PACKAGE" ]] && CARGO_ARGS=(test --release "$TEST_NAME" -p "$PACKAGE" --lib -- --nocapture --test-threads=1)

echo "Running: RUST_LOG=... cargo ${CARGO_ARGS[*]}" >&2

# Stream directly: cargo → rg (fast JSON line filter) → jaq (structured extract).
# (|| true) prevents pipe failure when the test itself fails.
#
# JSON fields (from tracing-subscriber .json()):
#   .fields.message          — stage name
#   .fields["uop.tree"]      — rangeify IR tree
#   .fields["ast.pre"]       — pre-optimization IR tree
#   .fields["ast.optimized"] — post-optimization IR tree
#   .fields["ast.initial"]   — kernel input tree (emitted once per kernel)
#   .fields["generated_c"]   — generated C kernel code
#   .fields.elapsed_ms       — stage timing (ms)
#   .target                  — module path

(RUST_LOG="$RUST_LOG" RUSTFLAGS="-C target-cpu=native" cargo "${CARGO_ARGS[@]}" 2>&1 || true) \
| rg --line-buffered '^\{' \
| jaq -rs --arg tfilter "$TARGET_FILTER" '

def hdr(msg; ms):
  if ms then "--- \(msg) [\(ms)ms] ---" else "--- \(msg) ---" end;

def section(title):
  "============================================================\n  \(title)\n============================================================\n";

def target_ok:
  if $tfilter == "" then true else (.target | test($tfilter)) end;

[ .[] | select(.level == "DEBUG") | select(target_ok) ] |
reduce .[] as $e (

  { out: "", phase: "", kernel: 0, last_initial: null };

  ($e.fields["uop.tree"]      // null) as $uop   |
  ($e.fields["ast.pre"]       // null) as $pre    |
  ($e.fields["ast.optimized"] // null) as $opt    |
  ($e.fields["ast.initial"]   // null) as $init   |
  ($e.fields["generated_c"]   // null) as $ccode  |
  ($e.fields.elapsed_ms       // null) as $ms     |
  ($e.fields.message          // "?")  as $msg    |

  if $uop then
    (if .phase != "rangeify" then
       .phase = "rangeify" |
       .out += section("RANGEIFY PHASE (single pass, pre-kernel-split)")
     else . end) |
    .out += "\(hdr($msg; $ms))\n\($uop)\n\n"

  elif $init then
    # "kernel initial" event — start new kernel section
    .kernel += 1 |
    .last_initial = $init |
    .phase = "initial" |
    .out += section("KERNEL \(.kernel)") |
    .out += "--- INITIAL: kernel input ---\n\($init)\n\n"

  elif $pre then
    (if .phase != "pre-opt" then
       .phase = "pre-opt"
     else . end) |
    .out += "\(hdr($msg; $ms))\n\($pre)\n\n"

  elif $opt then
    (if .phase != "post-opt" then
       (if .phase == "pre-opt" then
          .out += "\n  --- post-optimization ---\n\n"
        else . end) |
       .phase = "post-opt"
     else . end) |
    .out += "\(hdr($msg; $ms))\n\($opt)\n\n"

  elif $ccode then
    .out += "--- C KERNEL CODE ---\n\($ccode)\n\n"

  elif $ms then
    .out += "\(hdr($msg; $ms))\n\n"

  else . end
) | .out

' > "$OUTPUT"

STAGES=$(rg -c '^--- ' "$OUTPUT" || true)
KERNELS=$(rg -c 'C KERNEL CODE' "$OUTPUT" || echo 0)

if [[ "${STAGES:-0}" -eq 0 ]]; then
    echo "No stages found. Is the test using .json() tracing?" >&2
    rm -f "$OUTPUT"
    exit 1
fi

echo "Wrote $OUTPUT ($KERNELS kernel(s), $STAGES stages)" >&2
