//! Unit tests for the `NT_AMDGPU_METADATA` kernarg-layout parser.
//!
//! Three tiers, each self-gating so `cargo test` is green on any host:
//!   1. [`msgpack_decodes_interleaved_layout`] — a hand-built msgpack payload
//!      replicating a vendor-style *interleaved* layout (pointer, pad, pointer,
//!      scalar, hidden). Pure decode, always runs.
//!   2. [`parse_generated_co_matches_signature`] — compiles a small kernel with
//!      `clang` and checks the full ELF-note → msgpack path against the known
//!      svod buffers-first/scalars-after ABI. Skips if `clang` is unavailable.
//!   3. [`parse_real_aiter_bf16gemm_co`] — parses aiter's committed
//!      `bf16gemm*.co` when the `submodules/aiter` checkout is present, asserting
//!      its interleaved layout. Skips otherwise.

use crate::amd::metadata::{ValueKind, parse_amdgpu_metadata, parse_metadata_msgpack};

/// Minimal msgpack encoder for the metadata subset — just enough to hand-build a
/// kernel-metadata payload in-memory (fixmap/fixarray/fixstr + uint 8/16).
#[derive(Default)]
struct Pack(Vec<u8>);

impl Pack {
    fn str(&mut self, s: &str) -> &mut Self {
        assert!(s.len() < 32, "test strings stay in fixstr range");
        self.0.push(0xa0 | s.len() as u8);
        self.0.extend_from_slice(s.as_bytes());
        self
    }

    fn map(&mut self, n: usize) -> &mut Self {
        assert!(n < 16, "test maps stay in fixmap range");
        self.0.push(0x80 | n as u8);
        self
    }

    fn arr(&mut self, n: usize) -> &mut Self {
        assert!(n < 16, "test arrays stay in fixarray range");
        self.0.push(0x90 | n as u8);
        self
    }

    fn uint(&mut self, v: u64) -> &mut Self {
        match v {
            0..=0x7f => self.0.push(v as u8),
            0x80..=0xff => self.0.extend_from_slice(&[0xcc, v as u8]),
            0x100..=0xffff => {
                self.0.push(0xcd);
                self.0.extend_from_slice(&(v as u16).to_be_bytes());
            }
            _ => unreachable!("test values stay <= u16"),
        }
        self
    }

    /// `.name/.offset/.size/.value_kind` (+ optional `.address_space`) arg map.
    fn arg(&mut self, name: &str, offset: u64, size: u64, kind: &str, addr_space: Option<&str>) -> &mut Self {
        let entries = 4 + addr_space.is_some() as usize;
        self.map(entries);
        self.str(".name").str(name);
        self.str(".offset").uint(offset);
        self.str(".size").uint(size);
        self.str(".value_kind").str(kind);
        if let Some(a) = addr_space {
            self.str(".address_space").str(a);
        }
        self
    }
}

#[test]
fn msgpack_decodes_interleaved_layout() {
    // A vendor-style interleaved kernarg layout: pointers are NOT contiguous at
    // the front — a by_value pad follows each, a scalar sits at a >127 offset
    // (exercises the uint8 path), and a hidden arg trails. kernarg_segment_size
    // 352 exercises the uint16 path.
    let mut p = Pack::default();
    p.map(1).str("amdhsa.kernels").arr(1);
    p.map(4);
    p.str(".name").str("k");
    p.str(".symbol").str("k.kd");
    p.str(".kernarg_segment_size").uint(352);
    p.str(".args").arr(5);
    p.arg("D", 0, 8, "global_buffer", Some("global"));
    p.arg("pad", 8, 8, "by_value", None);
    p.arg("A", 16, 8, "global_buffer", Some("global"));
    p.arg("alpha", 128, 4, "by_value", None);
    p.arg("grid_x", 136, 8, "hidden_global_offset_x", None);

    let kernels = parse_metadata_msgpack(&p.0).expect("decode");
    assert_eq!(kernels.len(), 1);
    let k = &kernels[0];
    assert_eq!(k.name, "k");
    assert_eq!(k.symbol, "k.kd");
    assert_eq!(k.kernarg_segment_size, 352);
    assert_eq!(k.args.len(), 5);

    let d = &k.args[0];
    assert_eq!(d.name.as_deref(), Some("D"));
    assert_eq!((d.offset, d.size), (0, 8));
    assert_eq!(d.value_kind, ValueKind::GlobalBuffer);
    assert_eq!(d.address_space.as_deref(), Some("global"));
    assert!(d.value_kind.is_pointer() && !d.value_kind.is_hidden());

    let pad = &k.args[1];
    assert_eq!(pad.name.as_deref(), Some("pad"));
    assert_eq!((pad.offset, pad.size), (8, 8));
    assert_eq!(pad.value_kind, ValueKind::ByValue);
    assert_eq!(pad.address_space, None);
    assert!(!pad.value_kind.is_pointer());

    assert_eq!(k.args[2].offset, 16);
    assert_eq!(k.args[2].value_kind, ValueKind::GlobalBuffer);

    let alpha = &k.args[3];
    assert_eq!((alpha.offset, alpha.size), (128, 4)); // uint8 offset round-trips
    assert_eq!(alpha.value_kind, ValueKind::ByValue);

    let hidden = &k.args[4];
    assert_eq!(hidden.value_kind, ValueKind::Hidden("hidden_global_offset_x".to_string()));
    assert!(hidden.value_kind.is_hidden() && !hidden.value_kind.is_pointer());

    // The layout is genuinely interleaved: a by_value arg precedes a later
    // global_buffer arg (svod's buffers-first assumption would be wrong here).
    let first_by_value = k.args.iter().position(|a| a.value_kind == ValueKind::ByValue).unwrap();
    let last_buffer = k.args.iter().rposition(|a| a.value_kind == ValueKind::GlobalBuffer).unwrap();
    assert!(first_by_value < last_buffer, "expected interleaved pointers/scalars");
}

#[test]
fn rejects_non_elf_and_missing_note() {
    assert!(parse_amdgpu_metadata(b"not an elf").is_err());
    // A truncated msgpack map with the right top key but no kernel fields errors
    // rather than panicking.
    let mut p = Pack::default();
    p.map(1).str("amdhsa.kernels").arr(1).map(0);
    assert!(parse_metadata_msgpack(&p.0).is_err());
}

/// Shell out to `clang` to compile amdgcn IR → code object (ELF). Mirrors
/// `program.rs`'s test helper; returns `None` if clang/the target is missing.
fn clang_amdgcn(ir: &str, mcpu: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let child = std::process::Command::new("clang")
        .args(["-x", "ir", "-c", "-O2", "--target=amdgcn-amd-amdhsa"])
        .arg(format!("-mcpu={mcpu}"))
        .args(["-mcumode", "-nogpulib", "-nogpuinc", "-Wno-override-module", "-", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.as_ref()?.write_all(ir.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() || out.stdout.len() < 4 || &out.stdout[..4] != b"\x7fELF" {
        return None;
    }
    Some(out.stdout)
}

#[test]
fn parse_generated_co_matches_signature() {
    // Two buffers + one i32 scalar: svod's buffers-first/scalars-after ABI, and a
    // controlled object whose layout we can predict exactly.
    let ir = r#"target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @meta_probe(ptr noalias %buf0, ptr noalias %buf1, i32 %n) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %e = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %e
  store float 0.0, ptr %p
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;
    let Some(bytes) = clang_amdgcn(ir, "gfx942") else {
        eprintln!("skipping parse_generated_co: clang amdgcn (gfx942) unavailable");
        return;
    };
    let kernels = parse_amdgpu_metadata(&bytes).expect("parse metadata");
    let k = kernels.iter().find(|k| k.name == "meta_probe").expect("meta_probe kernel present");
    assert_eq!(k.symbol, "meta_probe.kd");
    assert_eq!(k.kernarg_segment_size, 20);

    let by_name = |n: &str| k.args.iter().find(|a| a.name.as_deref() == Some(n)).expect("named arg");
    let buf0 = by_name("buf0");
    assert_eq!((buf0.offset, buf0.size), (0, 8));
    assert_eq!(buf0.value_kind, ValueKind::GlobalBuffer);
    let buf1 = by_name("buf1");
    assert_eq!((buf1.offset, buf1.size), (8, 8));
    assert_eq!(buf1.value_kind, ValueKind::GlobalBuffer);
    let n = by_name("n");
    assert_eq!((n.offset, n.size), (16, 4));
    assert_eq!(n.value_kind, ValueKind::ByValue);
}

#[test]
fn parse_real_aiter_bf16gemm_co() {
    // aiter's committed vendor GEMM object (present only when the submodule is
    // checked out). Confirms the parser on a real vendor `.co` and that its
    // layout is interleaved, unlike svod's buffers-first convention.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../submodules/aiter/hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk_clean.co"
    );
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("skipping parse_real_aiter_bf16gemm_co: aiter submodule not checked out");
        return;
    };
    let kernels = parse_amdgpu_metadata(&bytes).expect("parse aiter metadata");
    assert_eq!(kernels.len(), 1);
    let k = &kernels[0];
    assert!(k.name.contains("bf16gemm"), "unexpected kernel name {}", k.name);
    assert!(k.symbol.ends_with(".kd"));
    assert_eq!(k.kernarg_segment_size, 352);

    let d = &k.args[0];
    assert_eq!(d.name.as_deref(), Some("D"));
    assert_eq!((d.offset, d.size), (0, 8));
    assert_eq!(d.value_kind, ValueKind::GlobalBuffer);
    assert_eq!(d.address_space.as_deref(), Some("global"));

    // Each pointer is followed by an 8-byte by_value pad (the interleave).
    assert_eq!(k.args[1].name.as_deref(), Some("pad"));
    assert_eq!((k.args[1].offset, k.args[1].size), (8, 8));
    assert_eq!(k.args[1].value_kind, ValueKind::ByValue);
    assert_eq!(k.args[2].name.as_deref(), Some("C"));
    assert_eq!(k.args[2].value_kind, ValueKind::GlobalBuffer);

    // A trailing pointer proves buffers are NOT all front-loaded.
    let sem = k.args.iter().find(|a| a.name.as_deref() == Some("ptr_semaphore")).expect("ptr_semaphore arg");
    assert_eq!(sem.offset, 336);
    assert_eq!(sem.value_kind, ValueKind::GlobalBuffer);
    let first_by_value = k.args.iter().position(|a| a.value_kind == ValueKind::ByValue).unwrap();
    let last_buffer = k.args.iter().rposition(|a| a.value_kind == ValueKind::GlobalBuffer).unwrap();
    assert!(first_by_value < last_buffer, "aiter layout must be interleaved");
}
