#!/usr/bin/env python3
"""Generate ONNX operator parity table from test results.

Usage:
    uv run --with='onnx' python onnx/scripts/parity.py

Runs the ONNX node tests (nightly required), parses results in-memory,
and writes onnx/PARITY.md with per-operator test counts.
"""

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import onnx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NODE_DIR = (
    REPO_ROOT / "submodules" / "onnx" / "onnx" / "backend" / "test" / "data" / "node"
)
REGISTRY_PATH = REPO_ROOT / "onnx" / "src" / "registry" / "mod.rs"
OUTPUT_PATH = REPO_ROOT / "onnx" / "PARITY.md"
BADGE_PATH = REPO_ROOT / "onnx" / "assets" / "coverage.svg"

# Operators implemented in registry (extracted from match arms in dispatch_multi)
# Plus If (handled in importer, not registry)
IMPLEMENTED_OPS: set[str] = set()

# Categories for grouping operators in the table
CATEGORIES: dict[str, list[str]] = {
    "Arithmetic": [
        "Abs",
        "Add",
        "Div",
        "Mean",
        "Mod",
        "Mul",
        "Neg",
        "Pow",
        "Sub",
        "Sum",
    ],
    "Bitwise": [
        "BitShift",
        "BitwiseAnd",
        "BitwiseNot",
        "BitwiseOr",
        "BitwiseXor",
    ],
    "Math": [
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "Ceil",
        "Cos",
        "Cosh",
        "Erf",
        "Exp",
        "Floor",
        "IsInf",
        "IsNaN",
        "Log",
        "Reciprocal",
        "Round",
        "Sign",
        "Sin",
        "Sinh",
        "Sqrt",
        "Tan",
        "Det",
    ],
    "Activation": [
        "Celu",
        "Elu",
        "Gelu",
        "HardSigmoid",
        "HardSwish",
        "LeakyRelu",
        "Mish",
        "PRelu",
        "Relu",
        "Selu",
        "Sigmoid",
        "Softmax",
        "LogSoftmax",
        "Softplus",
        "Softsign",
        "Swish",
        "Tanh",
        "ThresholdedRelu",
    ],
    "Comparison & Logic": [
        "And",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "Less",
        "LessOrEqual",
        "Not",
        "Or",
        "Where",
        "Xor",
    ],
    "Conditional & Selection": [
        "Clip",
        "Hardmax",
        "Max",
        "Min",
        "Shrink",
    ],
    "Type & Cast": [
        "Cast",
        "CastLike",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "QLinearConv",
        "QLinearMatMul",
        "QuantizeLinear",
    ],
    "Shape & Transform": [
        "CenterCropPad",
        "Concat",
        "ConstantOfShape",
        "DepthToSpace",
        "Expand",
        "EyeLike",
        "Flatten",
        "Pad",
        "Range",
        "Reshape",
        "Shape",
        "Size",
        "Slice",
        "SpaceToDepth",
        "Split",
        "Squeeze",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Col2Im",
        "ReverseSequence",
        "SplitToSequence",
        "Unique",
    ],
    "Indexing & Gather": [
        "Compress",
        "CumSum",
        "Gather",
        "GatherElements",
        "GatherND",
        "NonZero",
        "OneHot",
        "Scatter",
        "ScatterElements",
        "ScatterND",
        "TopK",
        "Trilu",
        "TensorScatter",
    ],
    "Reduction": [
        "ArgMax",
        "ArgMin",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
    ],
    "Neural Network": [
        "AffineGrid",
        "Attention",
        "AveragePool",
        "BatchNormalization",
        "Conv",
        "ConvTranspose",
        "Dropout",
        "Einsum",
        "Gemm",
        "GlobalAveragePool",
        "GlobalMaxPool",
        "GridSample",
        "GroupNormalization",
        "InstanceNormalization",
        "LRN",
        "LayerNormalization",
        "LpNormalization",
        "MatMul",
        "MatMulInteger",
        "MaxPool",
        "MeanVarianceNormalization",
        "NegativeLogLikelihoodLoss",
        "RMSNormalization",
        "RNN",
        "Resize",
        "RotaryEmbedding",
        "SoftmaxCrossEntropyLoss",
        "Upsample",
        "ConvInteger",
        "DeformConv",
        "GRU",
        "GlobalLpPool",
        "LSTM",
        "LpPool",
        "MaxRoiPool",
        "MaxUnpool",
        "NonMaxSuppression",
        "RoiAlign",
    ],
    "Control Flow": ["If", "Loop", "Scan"],
    "Constants & Identity": [
        "Constant",
        "Identity",
        "OptionalGetElement",
        "OptionalHasElement",
        "Optional",
    ],
    "Sequence": [
        "ConcatFromSequence",
        "SequenceAt",
        "SequenceConstruct",
        "SequenceEmpty",
        "SequenceErase",
        "SequenceInsert",
        "SequenceLength",
        "SequenceMap",
    ],
    "Random": [
        "Bernoulli",
        "Multinomial",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniform",
        "RandomUniformLike",
    ],
    "Signal & Text": [
        "BlackmanWindow",
        "DFT",
        "HammingWindow",
        "HannWindow",
        "ImageDecoder",
        "MelWeightMatrix",
        "STFT",
        "RegexFullMatch",
        "StringConcat",
        "StringNormalizer",
        "StringSplit",
        "TfIdfVectorizer",
    ],
}

# Microsoft extension operators
MS_EXTENSIONS: dict[str, list[str]] = {
    "com.microsoft Extensions": [
        "Attention",
        "EmbedLayerNormalization",
        "RotaryEmbedding",
        "SkipLayerNormalization",
    ],
}


def extract_implemented_ops() -> set[str]:
    """Extract implemented operator names from the registry match statement."""
    ops = set()
    src = REGISTRY_PATH.read_text()
    # Match arms like: "OpName" => ... or "OpName" | "OpName2" => ...
    import re

    for m in re.finditer(r'"([A-Z][A-Za-z0-9]+)"', src):
        ops.add(m.group(1))
    # If is handled in importer, not registry
    ops.add("If")
    # Upsample is handled as alias for Resize
    ops.add("Upsample")
    return ops


def scan_tests() -> tuple[dict[str, list[str]], dict[str, int]]:
    """Map test directories to primary operators and count expanded coverage.

    Returns:
        op_to_tests: {operator_name: [test_dir_name, ...]} — primary mapping
        expanded_uses: {operator_name: count} — how many expanded tests use each op
    """
    op_to_tests: dict[str, list[str]] = defaultdict(list)
    expanded_uses: dict[str, int] = defaultdict(int)

    for test_dir in sorted(NODE_DIR.iterdir()):
        if not test_dir.is_dir():
            continue
        model_path = test_dir / "model.onnx"
        if not model_path.exists():
            continue

        name = test_dir.name
        try:
            model = onnx.load(model_path)
            node_ops = [n.op_type for n in model.graph.node]
        except Exception:
            continue

        # _expanded tests use multiple ops to implement one
        if name.endswith("_expanded"):
            # Count each unique op used in this expanded test
            for op in set(node_ops):
                expanded_uses[op] += 1

            # Map to primary operator via the base (non-expanded) test
            base = name.removesuffix("_expanded")
            base_model = NODE_DIR / base / "model.onnx"
            if base_model.exists():
                try:
                    base_m = onnx.load(str(base_model))
                    base_ops = {n.op_type for n in base_m.graph.node}
                    if len(base_ops) == 1:
                        op_to_tests[next(iter(base_ops))].append(name)
                        continue
                except Exception:
                    pass
            op_to_tests["_expanded"].append(name)
            continue

        if len(set(node_ops)) == 1:
            op_to_tests[node_ops[0]].append(name)
        elif len(node_ops) > 0:
            # Multi-op test: use the dominant non-Constant op
            from collections import Counter

            non_const = [o for o in node_ops if o != "Constant"]
            if non_const:
                op = Counter(non_const).most_common(1)[0][0]
                op_to_tests[op].append(name)
            else:
                op_to_tests[node_ops[0]].append(name)

    return dict(op_to_tests), dict(expanded_uses)


def run_tests() -> dict[str, str]:
    """Run ONNX node tests and parse results in-memory.

    Returns: {test_name: "ok"|"failed"|"ignored"}
    """
    print("Running ONNX node tests (this may take a while)...", file=sys.stderr)
    proc = subprocess.run(
        [
            "rustup",
            "run",
            "nightly",
            "cargo",
            "test",
            "-p",
            "morok-onnx",
            "test::node",
            "--",
            "--include-ignored",
            "-Z",
            "unstable-options",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if proc.returncode not in (0, 101):  # 101 = some tests failed
        print(f"Test run failed (exit {proc.returncode}):", file=sys.stderr)
        print(
            proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr,
            file=sys.stderr,
        )
        sys.exit(1)

    results: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "test":
            continue
        event = ev.get("event")
        if event not in ("ok", "failed", "ignored"):
            continue
        name = ev.get("name", "")
        parts = name.split("::")
        if len(parts) >= 3 and parts[1] == "node":
            results[parts[2]] = event

    print(f"Parsed {len(results)} test results.", file=sys.stderr)
    return results


def is_extended(test_name: str) -> bool:
    """Check if a test name is from the extended test set.

    Extended tests typically have dtype suffixes, broadcast variants, etc.
    Standard tests are the basic/example ones.
    """
    # Extended patterns: dtype suffixes, broadcast, random, negative axis, etc.
    # But _example tests are standard
    # The ONNX test suite doesn't have a formal standard/extended split,
    # so we use the _expanded suffix as a separate axis and treat everything else
    # as either "base" (short names matching the op) or "extended" (variant tests)
    return "_expanded" in test_name


def count_results(
    tests: list[str], results: dict[str, str]
) -> tuple[int, int, int, int, int, int]:
    """Count pass/fail/skip for standard and extended tests.

    Returns: (std_pass, std_fail, std_skip, ext_pass, ext_fail, ext_skip)
    """
    sp = sf = ss = ep = ef = es = 0
    for t in tests:
        r = results.get(t)
        if is_extended(t):
            if r == "ok":
                ep += 1
            elif r == "failed":
                ef += 1
            else:
                es += 1
        else:
            if r == "ok":
                sp += 1
            elif r == "failed":
                sf += 1
            else:
                ss += 1
    return sp, sf, ss, ep, ef, es


def fmt_count(p: int, f: int, s: int) -> str:
    """Format pass/fail/skip counts as a compact string."""
    total = p + f + s
    if total == 0:
        return "-"
    parts = []
    if p:
        parts.append(f"{p} pass")
    if f:
        parts.append(f"{f} fail")
    if s:
        parts.append(f"{s} skip")
    return ", ".join(parts)


def generate_parity(results: dict[str, str]) -> tuple[str, dict]:
    """Generate the full PARITY.md content and stats for the badge."""
    implemented = extract_implemented_ops()
    op_tests, expanded_uses = scan_tests()

    # Collect all ops from categories
    all_categorized = set()
    for ops in CATEGORIES.values():
        all_categorized.update(ops)

    # Stats
    total_ops = len(all_categorized)
    impl_count = len(implemented & all_categorized)

    total_pass = sum(1 for v in results.values() if v == "ok")
    total_fail = sum(1 for v in results.values() if v == "failed")
    total_skip = sum(1 for v in results.values() if v == "ignored")
    total_tests = total_pass + total_fail + total_skip

    lines = [
        "# ONNX Operator Parity",
        "",
        f"**{impl_count} / {total_ops}** standard operators implemented ({impl_count * 100 // total_ops}%).",
        "",
        "Test results from the ONNX backend node test suite: "
        f"**{total_pass}** pass, **{total_fail}** fail, **{total_skip}** skip "
        f"(out of {total_tests} tests).",
        "",
        "The *expanded uses* column counts how many `_expanded` tests exercise each",
        "operator as a building block (indirect coverage beyond direct tests).",
        "",
    ]

    for cat_name, cat_ops in CATEGORIES.items():
        lines.append(f"## {cat_name}")
        lines.append("")
        lines.append(
            "| Operator | Impl | Tests (standard) | Tests (expanded) | Expanded uses |"
        )
        lines.append(
            "|----------|------|-------------------|-------------------|---------------|"
        )

        for op in cat_ops:
            impl = "Y" if op in implemented else "-"
            tests = op_tests.get(op, [])
            std_tests = [t for t in tests if not is_extended(t)]
            ext_tests = [t for t in tests if is_extended(t)]

            sp, sf, ss, _, _, _ = count_results(std_tests, results)
            _, _, _, ep, ef, es = count_results(ext_tests, results)

            std_str = fmt_count(sp, sf, ss)
            ext_str = fmt_count(ep, ef, es)
            uses = expanded_uses.get(op, 0)
            uses_str = str(uses) if uses else "-"

            lines.append(f"| {op} | {impl} | {std_str} | {ext_str} | {uses_str} |")

        lines.append("")

    # Microsoft extensions (no ONNX conformance tests for these)
    for cat_name, cat_ops in MS_EXTENSIONS.items():
        lines.append(f"## {cat_name}")
        lines.append("")
        lines.append("| Operator | Impl |")
        lines.append("|----------|------|")

        for op in cat_ops:
            lines.append(f"| {op} | Y |")

        lines.append("")

    stats = dict(
        impl_count=impl_count,
        total_ops=total_ops,
        total_pass=total_pass,
        total_tests=total_tests,
    )
    return "\n".join(lines), stats


def generate_badge(impl_count: int, total_ops: int, total_pass: int, total_tests: int):
    """Generate a compact SVG badge showing ops and test coverage."""
    ops_pct = impl_count * 100 // total_ops
    test_pct = total_pass * 100 // total_tests if total_tests else 0

    label = "ONNX"
    value = f"{impl_count}/{total_ops} ops  {test_pct}% tests"

    # Measure text widths (approximate: 6.5px per char at 11px font)
    char_w = 6.5
    pad = 10
    label_w = int(len(label) * char_w + pad * 2)
    value_w = int(len(value) * char_w + pad * 2)
    total_w = label_w + value_w

    # Color based on ops coverage
    if ops_pct >= 90:
        color = "#4c1"      # bright green
    elif ops_pct >= 70:
        color = "#97ca00"   # green
    elif ops_pct >= 50:
        color = "#dfb317"   # yellow
    else:
        color = "#e05d44"   # red

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="a"><rect width="{total_w}" height="20" rx="3"/></clipPath>
  <g clip-path="url(#a)">
    <rect width="{label_w}" height="20" fill="#555"/>
    <rect x="{label_w}" width="{value_w}" height="20" fill="{color}"/>
    <rect width="{total_w}" height="20" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_w / 2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_w / 2}" y="14">{label}</text>
    <text x="{label_w + value_w / 2}" y="15" fill="#010101" fill-opacity=".3">{value}</text>
    <text x="{label_w + value_w / 2}" y="14">{value}</text>
  </g>
</svg>'''
    BADGE_PATH.write_text(svg)


def main():
    results = run_tests()
    content, stats = generate_parity(results)
    OUTPUT_PATH.write_text(content)
    generate_badge(**stats)
    print(f"Written to {OUTPUT_PATH} and {BADGE_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
