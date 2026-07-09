//! Check the tile-IR node count + timing of each lowering stage, to isolate where the
//! blowup/hang is. Usage: `cargo run -p svod-tk2 --example dump_ir`

use std::collections::{HashMap, HashSet};
use std::time::Instant;
use svod_dtype::AmdArch;
use svod_tk2::{Node, TileId, TileIr, VectorizePass, matmul_lds_kblock_mw_pipe2};

fn main() {
    let (m, n, k) = (256usize, 256, 256);
    eprintln!("building pipe2 tile-IR at {m}³...");
    let t0 = Instant::now();
    let prog = matmul_lds_kblock_mw_pipe2(m, n, k, 64, 64, 1, 1, 64).apply(VectorizePass);
    eprintln!("  tile-IR built: {} nodes in {:?}", prog.ir.len(), t0.elapsed());
    print_histogram(&prog.ir);

    // Time the lower step (tile-IR → UOp sink)
    eprintln!("lowering...");
    let t1 = Instant::now();
    let sink = svod_tk2::lower::lower(&prog.ir, prog.sink, &prog.name);
    eprintln!("  lowered in {:?} — {} UOp nodes", t1.elapsed(), count_uops(&sink));

    // Time lower_and_prepare (includes the two graph_rewrite fixpoint passes)
    eprintln!("lower_and_prepare (fixpoint passes)...");
    let t2 = Instant::now();
    let sink = svod_tk2::lower::lower_and_prepare(&prog);
    eprintln!("  prepared in {:?} — {} UOp nodes", t2.elapsed(), count_uops(&sink));

    // Time the decompose
    eprintln!("decompose (AMD SLEEF patterns)...");
    let t3 = Instant::now();
    let matcher = svod_ir::decompositions::amd_decomposition_patterns();
    let sink = svod_ir::decompositions::decompose_with(&sink, &matcher);
    eprintln!("  decomposed in {:?} — {} UOp nodes", t3.elapsed(), count_uops(&sink));

    // Time the render (with a timeout-ish guard: we already know this hangs)
    eprintln!("render (LlvmTextRenderer)...");
    let t4 = Instant::now();
    let renderer = svod_codegen::llvm::text::LlvmTextRenderer::amd(AmdArch::Gfx942);
    let rendered = svod_codegen::Renderer::render(&renderer, &sink, Some(&prog.name)).expect("render");
    eprintln!("  rendered: {} bytes in {:?}", rendered.code.len(), t4.elapsed());

    eprintln!("done.");
}

/// Count UOp nodes reachable from a sink (via toposort).
fn count_uops(sink: &svod_ir::UOp) -> usize {
    sink.toposort().len()
}

/// Print a histogram of tile-IR node types so we can see if anything is unrolled/blown up.
fn print_histogram(ir: &TileIr) {
    let mut hist: HashMap<&str, usize> = HashMap::new();
    for id in 0..ir.len() {
        let id = TileId(id as u32);
        let kind = match ir.node(id) {
            Node::Range { .. } => "Range",
            Node::End { .. } => "End",
            Node::After { .. } => "After",
            Node::Barrier { .. } | Node::BareBarrier { .. } => "Barrier",
            Node::Mma { .. } => "Mma",
            Node::LoadGlobal { .. } => "LoadGlobal",
            Node::StoreGlobal { .. } => "StoreGlobal",
            Node::DsReadB64 { .. } => "DsRead",
            Node::DsWriteB64 { .. } => "DsWrite",
            Node::IndexAlu { .. } => "IndexAlu",
            Node::Const { .. } => "Const",
            Node::LoadVecAt { .. } => "LoadVec",
            Node::StoreVecAt { .. } => "StoreVec",
            Node::LoadRegVec { .. } => "LoadRegVec",
            Node::StoreRegVec { .. } => "StoreRegVec",
            Node::LdsPtrAs3 { .. } => "LdsPtrAs3",
            Node::SchedFence { .. } => "SchedFence",
            Node::SetPrio { .. } => "SetPrio",
            Node::WaveBarrier { .. } => "WaveBarrier",
            Node::SWaitLgkmcnt { .. } => "SWaitLgkmcnt",
            Node::SchedWallMarker => "SchedWallMarker",
            _ => "other",
        };
        *hist.entry(kind).or_default() += 1;
    }
    let mut sorted: Vec<_> = hist.into_iter().collect();
    sorted.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    eprintln!("  node histogram: {sorted:?}");
}
