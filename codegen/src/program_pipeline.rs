use std::sync::Arc;

use morok_device::device::{CompiledSpec, Compiler, ProgramSpec, Renderer};
use morok_device::{Error, Result};
use morok_dtype::DeviceSpec;
use morok_ir::{Op, UOp};
use morok_schedule::linearize::line_rewrite_cleanups;

type ProgramParts = (Arc<UOp>, Arc<UOp>, Option<Arc<UOp>>, Option<Arc<UOp>>, Option<Arc<UOp>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramTarget {
    Linear,
    Source,
    Binary,
}

fn invalid_program_state(details: impl Into<String>) -> Error {
    Error::Runtime { message: details.into() }
}

fn unpack_program(program: &Arc<UOp>) -> Result<ProgramParts> {
    let Op::Program { sink, device, linear, source, binary } = program.op() else {
        return Err(invalid_program_state(format!("expected PROGRAM op, got {:?}", program.op())));
    };
    Ok((sink.clone(), device.clone(), linear.clone(), source.clone(), binary.clone()))
}

fn validate_program_shape(program: &Arc<UOp>) -> Result<()> {
    let (sink, device, linear, source, binary) = unpack_program(program)?;

    if !matches!(sink.op(), Op::Sink { .. }) {
        return Err(invalid_program_state(format!("PROGRAM sink must be SINK op, got {:?}", sink.op())));
    }

    if !matches!(device.op(), Op::Device(_)) {
        return Err(invalid_program_state(format!("PROGRAM device must be DEVICE op, got {:?}", device.op())));
    }

    if let Some(linear) = &linear
        && !matches!(linear.op(), Op::Linear { .. })
    {
        return Err(invalid_program_state(format!("PROGRAM linear stage must be LINEAR op, got {:?}", linear.op())));
    }

    if let Some(source) = &source
        && !matches!(source.op(), Op::Source { .. })
    {
        return Err(invalid_program_state(format!("PROGRAM source stage must be SOURCE op, got {:?}", source.op())));
    }

    if let Some(binary) = &binary
        && !matches!(binary.op(), Op::ProgramBinary { .. })
    {
        return Err(invalid_program_state(format!(
            "PROGRAM binary stage must be ProgramBinary op, got {:?}",
            binary.op()
        )));
    }

    if source.is_some() && linear.is_none() {
        return Err(invalid_program_state("malformed PROGRAM state: SOURCE requires LINEAR stage"));
    }
    if binary.is_some() && source.is_none() {
        return Err(invalid_program_state("malformed PROGRAM state: BINARY requires SOURCE stage"));
    }

    Ok(())
}

fn preserve_program_context(new_program: Arc<UOp>, old_program: &Arc<UOp>) -> Arc<UOp> {
    let mut out = new_program.rtag(old_program.tag().clone());
    if let Some(meta) = old_program.metadata_raw() {
        out = out.with_metadata_raw(meta);
    }
    out
}

fn rebuild_program(
    base_program: &Arc<UOp>,
    linear: Option<Arc<UOp>>,
    source: Option<Arc<UOp>>,
    binary: Option<Arc<UOp>>,
) -> Result<Arc<UOp>> {
    let (sink, device, _, _, _) = unpack_program(base_program)?;
    let rebuilt = UOp::program(sink, device, linear, source, binary);
    Ok(preserve_program_context(rebuilt, base_program))
}

/// Create initial PROGRAM(sink, device) state.
pub fn program_from_sink(sink: Arc<UOp>, device: DeviceSpec) -> Arc<UOp> {
    let sink = if matches!(sink.op(), Op::Sink { .. }) { sink } else { UOp::sink(vec![sink]) };
    UOp::program(sink, UOp::device(device), None, None, None)
}

/// PROGRAM -> LINEAR stage.
pub fn do_linearize(program: &Arc<UOp>) -> Result<Arc<UOp>> {
    validate_program_shape(program)?;
    let (sink, _device, linear, source, binary) = unpack_program(program)?;
    if linear.is_some() {
        return Ok(program.clone());
    }

    let linear_ops = morok_schedule::linearize_with_cfg(sink);
    let linear_clean = line_rewrite_cleanups(linear_ops);
    let linear_uop = UOp::linear(linear_clean.into());
    rebuild_program(program, Some(linear_uop), source, binary)
}

/// PROGRAM(+LINEAR) -> SOURCE stage via Renderer.
pub fn do_render(program: &Arc<UOp>, renderer: &dyn Renderer, name: Option<&str>) -> Result<(Arc<UOp>, ProgramSpec)> {
    let linearized = do_linearize(program)?;
    let (_sink, _device, linear, source, binary) = unpack_program(&linearized)?;

    if source.is_some() || binary.is_some() {
        return Err(invalid_program_state(format!(
            "do_render expects PROGRAM stage with LINEAR only (source=None,binary=None), got source_present={}, binary_present={}",
            source.is_some(),
            binary.is_some()
        )));
    }

    let linear_uop = linear.clone().ok_or_else(|| invalid_program_state("PROGRAM has no LINEAR stage"))?;

    let spec = renderer.render(&linear_uop, name)?;
    let source_uop = UOp::source(spec.src.clone());
    let mut rendered = rebuild_program(&linearized, linear, Some(source_uop), None)?;
    rendered = rendered.with_metadata(spec.clone());
    Ok((rendered, spec))
}

/// PROGRAM(+SOURCE) -> BINARY stage via Compiler.
pub fn do_compile(program: &Arc<UOp>, compiler: &dyn Compiler) -> Result<(Arc<UOp>, CompiledSpec)> {
    validate_program_shape(program)?;
    let (sink, _device, linear, source, binary) = unpack_program(program)?;

    if let Some(binary_uop) = binary {
        let Op::ProgramBinary { bytes } = binary_uop.op() else {
            return Err(invalid_program_state("PROGRAM binary stage is not a ProgramBinary UOp"));
        };

        let spec = ProgramSpec::from_uop(program)?;

        let mut compiled = CompiledSpec::from_bytes(spec.name.clone(), bytes.clone(), sink);
        if !spec.src.is_empty() {
            compiled.src = Some(spec.src.clone());
        }
        compiled.var_names = spec.var_names.clone();
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        compiled.buf_count = spec.buf_count;
        return Ok((program.clone(), compiled));
    }

    if source.is_none() {
        return Err(invalid_program_state("PROGRAM has no SOURCE stage"));
    }

    let spec = ProgramSpec::from_uop(program)?;
    if spec.src.is_empty() {
        return Err(invalid_program_state("PROGRAM has empty SOURCE stage"));
    }

    let compiled = compiler.compile(&spec)?;

    let binary_uop = UOp::binary(compiled.bytes.clone());
    let mut compiled_program = rebuild_program(program, linear, source, Some(binary_uop))?;
    compiled_program = compiled_program.with_metadata(spec);
    Ok((compiled_program, compiled))
}

/// Progressively advance SINK/PROGRAM input to a requested PROGRAM stage.
pub fn get_program(
    input: &Arc<UOp>,
    renderer: &dyn Renderer,
    compiler: &dyn Compiler,
    name: Option<&str>,
    target: ProgramTarget,
) -> Result<Arc<UOp>> {
    let mut program = match input.op() {
        Op::Program { .. } => {
            validate_program_shape(input)?;
            input.clone()
        }
        other => return Err(invalid_program_state(format!("expected PROGRAM input, got {other:?}"))),
    };

    if matches!(target, ProgramTarget::Linear | ProgramTarget::Source | ProgramTarget::Binary) {
        let (_, _, linear, _, _) = unpack_program(&program)?;
        if linear.is_none() {
            program = do_linearize(&program)?;
        }
    }

    if matches!(target, ProgramTarget::Source | ProgramTarget::Binary) {
        let (_, _, _, source, _) = unpack_program(&program)?;
        if source.is_none() {
            let (rendered, _) = do_render(&program, renderer, name)?;
            program = rendered;
        }
    }

    if matches!(target, ProgramTarget::Binary) {
        let (_, _, _, _, binary) = unpack_program(&program)?;
        if binary.is_none() {
            let (compiled, _) = do_compile(&program, compiler)?;
            program = compiled;
        }
    }

    validate_program_shape(&program)?;
    Ok(program)
}
