//! Direct device launch — compile a lowered tile-IR SINK and dispatch it against
//! concrete buffers, populating the output(s) in place. A faithful, tk-free port
//! of `tk/src/launch.rs::compile` reusing the same `codegen`/`device`/`runtime`
//! building blocks (never tk's wrappers).

use std::collections::HashMap;

use snafu::{OptionExt, ResultExt};
use svod_codegen::program_pipeline::{self, ProgramTarget};
use svod_device::Buffer;
use svod_device::device::ProgramSpec;

use crate::error::{self, Result};
use crate::kernels::Program;
use crate::lower;

/// Compile `program` for the device backing `buffers` and dispatch it once,
/// synchronously. `buffers` are ordered output(s)-first then inputs — the ABI slot
/// order the builder's `global()` calls established (outputs bound before inputs).
///
/// # Safety
/// The buffers must stay allocated for the call; the synchronous dispatch
/// (`wait=true`) holds them for its duration.
pub fn run(program: &Program, buffers: &[Buffer]) -> Result<()> {
    snafu::ensure!(!buffers.is_empty(), error::BufferMissingSnafu { slot: 0usize, supplied: 0usize });

    // Resolve the concrete Device (renderer/compiler/runtime) for the buffers' device.
    let device_spec = buffers[0].allocator().device_spec();
    let device = svod_runtime::DEVICE_FACTORIES
        .device(&device_spec, svod_device::registry::registry())
        .context(error::DeviceResolveSnafu { spec: format!("{device_spec:?}") })?;

    // Lower + the pre-render rewrites, then the backend-mandatory decomposition
    // (AMD f32→bf16 casts, SLEEF transcendentals) the direct path must apply itself.
    let sink = lower::lower_and_prepare(program);
    let sink = match device.renderer.decompositor() {
        Some(matcher) => svod_ir::decompositions::decompose_with(&sink, &matcher),
        None => sink,
    };

    // PROGRAM(sink) → SOURCE (runs type_verify inside do_render) → BINARY.
    let uop_program = program_pipeline::program_from_sink(sink, device.device.clone());
    let name = program.name.clone();
    let rendered = program_pipeline::get_program(
        &uop_program,
        device.renderer.as_ref(),
        device.compiler.as_ref(),
        Some(&name),
        ProgramTarget::Source,
    )
    .context(error::CompileSnafu { name: name.clone() })?;
    let (compiled_program, compiled) = program_pipeline::do_compile(&rendered, device.compiler.as_ref())
        .context(error::CompileSnafu { name: name.clone() })?;

    let spec = ProgramSpec::from_uop(&compiled_program).context(error::CompileSnafu { name: name.clone() })?;
    let prog = (device.runtime)(&compiled).context(error::CompileSnafu { name: name.clone() })?;

    // Resolve buffer pointers in the compiled ABI order (sorted PARAM slots).
    let mut ptrs: Vec<*mut u8> = Vec::with_capacity(spec.globals.len());
    for &slot in &spec.globals {
        let buf = buffers.get(slot).context(error::BufferMissingSnafu { slot, supplied: buffers.len() })?;
        buf.ensure_allocated().context(error::BufferSnafu { slot })?;
        // SAFETY: allocated above; the caller keeps the buffers alive for this call.
        ptrs.push(unsafe { buf.as_raw_ptr() });
    }

    // No symbolic vars in a hand-built kernel: concrete grid/block from the SPECIALs.
    let var_vals: HashMap<&str, i64> = HashMap::new();
    let dims = spec.launch_dims(&var_vals).context(error::CompileSnafu { name: name.clone() })?;
    let vals: Vec<i64> = spec.var_names.iter().map(|n| var_vals.get(n.as_str()).copied().unwrap_or(0)).collect();

    // SAFETY: pointers are allocated + sized to the kernel's expectations and held
    // alive by the caller; synchronous dispatch drains before return.
    unsafe {
        prog.execute(&ptrs, &vals, Some(dims.global_size), dims.local_size, true).context(error::DispatchSnafu { name })
    }
}

/// Compile `program` for the given device spec and return `(llvm_ir_source, code_object_bytes)` —
/// the rendered amdgcn LLVM IR and the compiled ELF code object, WITHOUT dispatching. The ISA
/// validation route: dump the `.co` and `llvm-objdump-20 -d` it (the exact runtime ISA), or clang
/// the `.ll`. Shares the lower → decompose → render → compile prefix of [`run`].
pub fn compile_artifacts(program: &Program, device_spec: &svod_dtype::DeviceSpec) -> Result<(String, Vec<u8>)> {
    let device = svod_runtime::DEVICE_FACTORIES
        .device(device_spec, svod_device::registry::registry())
        .context(error::DeviceResolveSnafu { spec: format!("{device_spec:?}") })?;

    let sink = lower::lower_and_prepare(program);
    let sink = match device.renderer.decompositor() {
        Some(matcher) => svod_ir::decompositions::decompose_with(&sink, &matcher),
        None => sink,
    };

    let uop_program = program_pipeline::program_from_sink(sink, device.device.clone());
    let name = program.name.clone();
    let rendered = program_pipeline::get_program(
        &uop_program,
        device.renderer.as_ref(),
        device.compiler.as_ref(),
        Some(&name),
        ProgramTarget::Source,
    )
    .context(error::CompileSnafu { name: name.clone() })?;
    let src = ProgramSpec::from_uop(&rendered).context(error::CompileSnafu { name: name.clone() })?.src;
    let (_compiled_program, compiled) =
        program_pipeline::do_compile(&rendered, device.compiler.as_ref()).context(error::CompileSnafu { name })?;
    Ok((src, compiled.bytes))
}

/// **Headless render** (no device, no clang, no KFD): lower + decompose + linearize + render the
/// program's amdgcn LLVM IR text for `arch`, the same IR [`compile_artifacts`] would hand to
/// clang — but stopping before the clang compile. This is the no-hardware ISA-comparison route:
/// render the `.ll`, then `clang -O3 --target=amdgcn -mcpu=gfx942` it offline (or compare the IR).
///
/// Mirrors the real pipeline exactly (`lower_and_prepare` → decompose → `program_from_sink` →
/// `do_linearize` → `LlvmTextRenderer::render`), just without the device/clang/KFD. The
/// decompositor is reconstructed from the arch directly (the only device-sourced piece), so the
/// rendered IR is byte-identical to what a real AMD device's renderer would produce.
pub fn render_amd_ir(program: &Program, arch: svod_dtype::AmdArch) -> Result<String> {
    use svod_codegen::program_pipeline;
    use svod_dtype::DeviceSpec;

    let sink = lower::lower_and_prepare(program);
    let matcher = svod_ir::decompositions::amd_decomposition_patterns();
    let sink = svod_ir::decompositions::decompose_with(&sink, &matcher);

    // Wrap in a PROGRAM envelope + linearize (CFG + toposort + control-flow). The renderer
    // requires `Op::Linear` input (it rejects a raw `Op::Sink`), and its error path `{:?}`s the
    // full op — which exponentially expands the shared DAG → OOM. So we MUST linearize first.
    let uop_program = program_pipeline::program_from_sink(sink, DeviceSpec::Amd { device_id: 0 });
    let linearized = program_pipeline::do_linearize(&uop_program)
        .map_err(|e| error::Error::Render { name: program.name.clone(), reason: e.to_string() })?;

    // Extract the Op::Linear out of the PROGRAM envelope (the renderer matches on Linear, not
    // Program). The real `do_render` does this at program_pipeline.rs:152.
    let svod_ir::Op::Program { linear: Some(lin), .. } = linearized.op() else {
        return Err(error::Error::Render {
            name: program.name.clone(),
            reason: "do_linearize did not produce a Linear stage".into(),
        });
    };
    let linear_uop = lin.clone();

    let renderer = svod_codegen::llvm::text::LlvmTextRenderer::amd(arch);
    let rendered = svod_codegen::Renderer::render(&renderer, &linear_uop, Some(&program.name))
        .map_err(|e| error::Error::Render { name: program.name.clone(), reason: e.to_string() })?;
    Ok(rendered.code)
}
