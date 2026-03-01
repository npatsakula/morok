#!/usr/bin/env bash
# Extract IR trees from each pipeline stage for a given test.
#
# Prerequisite: the target test must have the #[tracing_test::traced_test]
# attribute so that a tracing subscriber is active during the test.
#
# Pipeline structure:
#   rangeify_with_map (1 call)    → mega-pass + buffer limit (uop.tree field)
#   kernel splitting              → no tracing
#   apply_pre_optimization (N×)   → per-kernel stages (ast.pre field)
#   apply_post_optimization (N×)  → per-kernel stages (ast.optimized field)
#   linearizer                    → Stages 20-22 (no tree tracing)
#   C codegen (N×)                → generated C code per kernel (generated_c field)
#
# Usage:
#   ./scripts/extract-ir.sh <test_name> [-p <package>] [-o <output>]
#
# Examples:
#   ./scripts/extract-ir.sh test_sum_axis1_value -p morok-tensor
#   ./scripts/extract-ir.sh test_argmax_1d -p morok-tensor -o /tmp/ir.txt

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: extract-ir.sh <test_name> [-p <package>] [-o <output>]

Arguments:
  test_name       Cargo test name filter (passed to `cargo test`)

Options:
  -p <package>    Cargo package filter (e.g. morok-tensor)
  -o <output>     Output file (default: ir_<test_name>.txt)
  -h              Show this help

Note: the target test MUST have #[tracing_test::traced_test] attribute.
EOF
    exit 1
}

[[ $# -lt 1 ]] && usage
[[ "$1" == "-h" || "$1" == "--help" ]] && usage

TEST_NAME="$1"; shift
PACKAGE=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p) PACKAGE="$2"; shift 2 ;;
        -o) OUTPUT="$2"; shift 2 ;;
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
RAW=$(RUST_LOG="$RUST_LOG" cargo "${CARGO_ARGS[@]}" 2>&1) || true
# Note: cargo test exits non-zero when the test fails, but we still want
# to extract the pipeline IR from the captured output.

# Strip ANSI escape codes
RAW=$(printf '%s' "$RAW" | sed 's/\x1b\[[0-9;]*m//g')

# Parse stage markers.
#
# Each DEBUG line looks like:
#   <ts> DEBUG <test>:<span>{field=...}: <target>: <message> <field>="<tree>"
#
# The tree is a single-line quoted string with literal \n for newlines.
# Three field names: uop.tree (rangeify), ast.optimized (optimizer), generated_c (C codegen).
#
# Kernel boundary detection:
#   - Rangeify runs ONCE for the whole SINK (pre-split). origin.tree span field
#     is the input; extract it from the first Stage 0 line.
#   - Optimizer runs ONCE PER KERNEL (post-split). Each kernel's ast.initial
#     span field marks its input. Detect new kernel when "Stage 8:" appears.

awk '
function unescape_tree(raw,    tree) {
    tree = raw
    gsub(/\\n/, "\n", tree)
    gsub(/\\t/, "\t", tree)
    return tree
}

function extract_field(line, field,    pat, m) {
    pat = field "=\"(([^\"\\\\]|\\\\.)*)\""
    if (match(line, pat, m))
        return m[1]
    return ""
}

function extract_message(line, field,    pat, m) {
    # message is between "morok_schedule::<module_path>: " and " <field>="
    # module path has multiple :: segments (e.g. rangeify::transforms)
    pat = "morok_schedule(::[a-z_]+)+: (.+) " field "="
    if (match(line, pat, m))
        return m[2]
    return ""
}

function extract_elapsed(line,    pat, m) {
    pat = "elapsed_ms=([0-9]+)"
    if (match(line, pat, m))
        return m[1] + 0
    return -1
}

function format_stage_header(msg, elapsed_ms) {
    if (elapsed_ms >= 0)
        return sprintf("--- %s [%dms] ---", msg, elapsed_ms)
    else
        return sprintf("--- %s ---", msg)
}

# ─── Rangeify phase (global, pre-kernel-split) ───
# origin.tree span field appears on every line; extract it only once.

/DEBUG/ && /uop\.tree=/ {
    # On first rangeify stage, emit origin tree and section header
    if (!rangeify_started) {
        rangeify_started = 1
        printf "%s\n", "============================================================"
        printf "  RANGEIFY PHASE (single pass, pre-kernel-split)\n"
        printf "%s\n\n", "============================================================"

        origin = extract_field($0, "origin\\.tree")
        if (origin != "") {
            printf "--- ORIGIN: input tree ---\n%s\n\n", unescape_tree(origin)
        }
    }

    msg = extract_message($0, "uop\\.tree")
    tree = extract_field($0, "uop\\.tree")
    if (msg != "" && tree != "") {
        printf "%s\n%s\n\n", format_stage_header(msg, extract_elapsed($0)), unescape_tree(tree)
    }
}

# ─── Pre-optimization phase (per kernel) ───
# ast.pre field from apply_pre_optimization (load_collapse, split_ranges, etc.)
# ast.initial span field carries the kernel input tree.

/DEBUG/ && /ast\.pre=/ {
    msg = extract_message($0, "ast\\.pre")
    tree = extract_field($0, "ast\\.pre")

    # Detect kernel boundary from ast.initial span field
    if (!pre_opt_started || extract_field($0, "ast\\.initial") != last_pre_initial) {
        initial = extract_field($0, "ast\\.initial")
        if (initial != "") {
            last_pre_initial = initial
            kernel_count++
            printf "%s\n", "============================================================"
            printf "  KERNEL %d — PRE-OPTIMIZATION\n", kernel_count
            printf "%s\n\n", "============================================================"
            printf "--- INITIAL: pre-opt input ---\n%s\n\n", unescape_tree(initial)
            pre_opt_started = 1
        }
    }

    if (msg != "" && tree != "") {
        printf "%s\n%s\n\n", format_stage_header(msg, extract_elapsed($0)), unescape_tree(tree)
    }
}

# ─── Pre-optimization phase: elapsed-only lines (no ast.pre field) ───
# Some stages only emit elapsed_ms without a tree (e.g. linearize_multi_index).

/DEBUG/ && /elapsed_ms=/ && !/uop\.tree=/ && !/ast\.pre=/ && !/ast\.optimized=/ && !/generated_c=/ {
    # Extract message: everything between "morok_schedule...: " and " elapsed_ms="
    pat = "morok_schedule(::[a-z_]+)+: (.+) elapsed_ms="
    if (match($0, pat, m)) {
        elapsed = extract_elapsed($0)
        printf "%s\n\n", format_stage_header(m[2], elapsed)
    }
}

# ─── Post-optimization phase (per kernel) ───
# "Stage 8:" marks the start of each kernel post-optimization pipeline.
# ast.initial span field carries the optimizer input tree.

/DEBUG/ && /ast\.optimized=/ {
    msg = extract_message($0, "ast\\.optimized")
    tree = extract_field($0, "ast\\.optimized")

    # Detect kernel boundary: "Stage 8:" is always the first post-opt stage
    if (msg ~ /^Stage 8:/) {
        post_kernel_count++
        # Only print header if pre-opt did not already print one for this kernel
        if (post_kernel_count > kernel_count) {
            kernel_count = post_kernel_count
            printf "%s\n", "============================================================"
            printf "  KERNEL %d — POST-OPTIMIZATION\n", kernel_count
            printf "%s\n\n", "============================================================"
        } else {
            printf "\n  --- post-optimization ---\n\n"
        }

        initial = extract_field($0, "ast\\.initial")
        if (initial != "") {
            printf "--- INITIAL: post-opt input ---\n%s\n\n", unescape_tree(initial)
        }
    }

    if (msg != "" && tree != "") {
        printf "%s\n%s\n\n", format_stage_header(msg, extract_elapsed($0)), unescape_tree(tree)
    }
}

# ─── C codegen (per kernel, after optimizer) ───
# Emitted by morok_codegen::c after linearization + rendering.

/DEBUG/ && /generated_c=/ {
    code = extract_field($0, "generated_c")
    if (code != "") {
        printf "--- C KERNEL CODE ---\n%s\n\n", unescape_tree(code)
    }
}
' <<< "$RAW" > "$OUTPUT"

# Trim trailing whitespace
sed -i 's/[[:space:]]*$//' "$OUTPUT"

STAGES=$(grep -c '^--- ' "$OUTPUT" || true)
KERNELS=$(grep -oP 'KERNEL \K\d+' "$OUTPUT" | sort -un | wc -l || true)

if [[ "$STAGES" -eq 0 ]]; then
    echo "No stages found. Is #[tracing_test::traced_test] on the test?" >&2
    rm -f "$OUTPUT"
    exit 1
fi

echo "Wrote $OUTPUT ($KERNELS kernel(s), $STAGES stages)" >&2
