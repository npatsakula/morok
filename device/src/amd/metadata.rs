//! Parser for the AMDGPU code-object metadata note (`NT_AMDGPU_METADATA`).
//!
//! An `amdgcn-amd-amdhsa` ELF carries an ELF note (`.note` section or `PT_NOTE`
//! segment) with note type `NT_AMDGPU_METADATA` (= 32) and name `"AMDGPU"`. Its
//! descriptor is a msgpack map whose `"amdhsa.kernels"` key holds an array of
//! per-kernel maps, each with the exact kernarg `.args` table: every argument's
//! byte `.offset`, `.size`, `.value_kind`, and `.address_space`.
//!
//! Unlike the 64-byte kernel *descriptor* (`AmdHsaKernelDescriptor` in
//! [`super::sys::hsa`]) — which only reports the total kernarg *size* — this note
//! gives the true per-argument *layout*. svod's own kernels lay buffers first
//! (u64 each) then scalars (i32 each), but an external vendor object (e.g.
//! aiter's `bf16gemm*.co`) may interleave pointers, scalars, hidden args, and
//! explicit padding — so loading such a `.co` requires these exact offsets.
//!
//! The parser is intentionally free of GPU/FFI dependencies: it is pure byte
//! decoding over the [`object`] ELF reader plus a minimal msgpack reader for the
//! exact subset the metadata uses, so it compiles and tests on any host.

use object::LittleEndian;
use object::elf::FileHeader64;
use object::read::elf::{FileHeader, ProgramHeader, SectionHeader};
use snafu::{OptionExt, ResultExt, Snafu};

/// ELF note type for the AMDGPU code-object metadata (`NT_AMDGPU_METADATA`).
/// Not exported by the [`object`] crate, so defined here per the AMDHSA ABI.
pub const NT_AMDGPU_METADATA: u32 = 32;

/// Errors from [`parse_amdgpu_metadata`].
#[derive(Debug, Snafu)]
pub enum MetadataError {
    /// Input is not an AMDGPU ELF (bad magic, wrong class, etc.).
    #[snafu(display("not an AMDGPU ELF code object: {reason}"))]
    NotElf { reason: String },

    /// The [`object`] ELF reader failed to walk the header/sections/notes.
    #[snafu(display("ELF parse error: {source}"))]
    ElfParse { source: object::read::Error },

    /// No `NT_AMDGPU_METADATA` note with name `"AMDGPU"` was present.
    #[snafu(display("no NT_AMDGPU_METADATA note found in code object"))]
    NoMetadataNote,

    /// The msgpack payload ended mid-value.
    #[snafu(display("truncated msgpack: need {needed} bytes at offset {offset} (payload len {len})"))]
    Truncated { offset: usize, needed: usize, len: usize },

    /// A msgpack marker byte was malformed or unsupported.
    #[snafu(display("invalid msgpack at offset {offset}: {reason}"))]
    BadMsgpack { offset: usize, reason: String },

    /// A msgpack string was not valid UTF-8.
    #[snafu(display("invalid UTF-8 in msgpack string: {source}"))]
    Utf8 { source: std::str::Utf8Error },

    /// A required metadata key was absent.
    #[snafu(display("metadata missing required key '{key}'"))]
    MissingKey { key: String },

    /// A metadata key held an unexpected msgpack type.
    #[snafu(display("metadata key '{key}' has unexpected type (expected {expected})"))]
    WrongType { key: String, expected: &'static str },
}

/// Result alias for the metadata parser.
pub type Result<T> = std::result::Result<T, MetadataError>;

/// The `.value_kind` of a kernel argument. Concrete pointer/scalar kinds are
/// named; every compiler-injected `hidden_*` argument collapses into
/// [`ValueKind::Hidden`] (retaining its exact spelling), and anything
/// unrecognized falls back to [`ValueKind::Other`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueKind {
    /// A global-memory pointer argument (`global_buffer`).
    GlobalBuffer,
    /// A pass-by-value scalar/struct argument (`by_value`).
    ByValue,
    /// A pointer into LDS (`dynamic_shared_pointer`).
    DynamicSharedPointer,
    /// A sampler handle (`sampler`).
    Sampler,
    /// An image handle (`image`).
    Image,
    /// A pipe handle (`pipe`).
    Pipe,
    /// A device-queue handle (`queue`).
    Queue,
    /// A compiler-injected hidden argument (`hidden_*`); the full kind string is
    /// preserved so callers can distinguish e.g. `hidden_none` from
    /// `hidden_global_offset_x`.
    Hidden(String),
    /// An unrecognized `.value_kind` string (forward compatibility).
    Other(String),
}

impl ValueKind {
    /// Whether this argument is a compiler-injected hidden argument.
    pub fn is_hidden(&self) -> bool {
        matches!(self, ValueKind::Hidden(_))
    }

    /// Whether this argument occupies an 8-byte handle/pointer slot the host
    /// fills with a GPU address (as opposed to a by-value scalar or hidden arg).
    pub fn is_pointer(&self) -> bool {
        matches!(
            self,
            ValueKind::GlobalBuffer
                | ValueKind::DynamicSharedPointer
                | ValueKind::Image
                | ValueKind::Sampler
                | ValueKind::Pipe
                | ValueKind::Queue
        )
    }
}

impl From<&str> for ValueKind {
    fn from(s: &str) -> Self {
        match s {
            "global_buffer" => ValueKind::GlobalBuffer,
            "by_value" => ValueKind::ByValue,
            "dynamic_shared_pointer" => ValueKind::DynamicSharedPointer,
            "sampler" => ValueKind::Sampler,
            "image" => ValueKind::Image,
            "pipe" => ValueKind::Pipe,
            "queue" => ValueKind::Queue,
            hidden if hidden.starts_with("hidden_") => ValueKind::Hidden(hidden.to_string()),
            other => ValueKind::Other(other.to_string()),
        }
    }
}

/// One kernel argument's kernarg-segment layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelArg {
    /// Source-level argument name, when present (`.name`).
    pub name: Option<String>,
    /// Byte offset of this argument within the kernarg segment (`.offset`).
    pub offset: u64,
    /// Byte size of this argument (`.size`).
    pub size: u64,
    /// The argument's kind (`.value_kind`).
    pub value_kind: ValueKind,
    /// Address space for pointer arguments, when present (`.address_space`).
    pub address_space: Option<String>,
}

/// A single kernel's identity + full ordered kernarg layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelMeta {
    /// Kernel name (`.name`).
    pub name: String,
    /// Kernel-descriptor symbol, e.g. `"<name>.kd"` (`.symbol`).
    pub symbol: String,
    /// Total kernarg-segment size in bytes (`.kernarg_segment_size`).
    pub kernarg_segment_size: u64,
    /// Arguments in declaration order — the exact kernarg layout.
    pub args: Vec<KernelArg>,
}

/// Parse the `NT_AMDGPU_METADATA` note of an AMDGPU code object and return the
/// per-kernel kernarg layout (identity + ordered `.args`).
pub fn parse_amdgpu_metadata(co_bytes: &[u8]) -> Result<Vec<KernelMeta>> {
    let payload = extract_metadata_note(co_bytes)?;
    parse_metadata_msgpack(payload)
}

/// Locate the AMDGPU metadata note's msgpack descriptor bytes inside `co_bytes`.
///
/// Walks section notes first (`.note` in an `ET_REL` object) then program-header
/// notes (`PT_NOTE` in an `ET_DYN`/`ET_EXEC` vendor object), matching
/// `n_type == NT_AMDGPU_METADATA` and name `"AMDGPU"`.
fn extract_metadata_note(co_bytes: &[u8]) -> Result<&[u8]> {
    if co_bytes.len() < 4 || &co_bytes[..4] != b"\x7fELF" {
        return NotElfSnafu { reason: "missing ELF magic".to_string() }.fail();
    }
    let header = FileHeader64::<LittleEndian>::parse(co_bytes).context(ElfParseSnafu)?;
    let endian = header.endian().context(ElfParseSnafu)?;

    for section in header.section_headers(endian, co_bytes).context(ElfParseSnafu)? {
        if let Some(mut notes) = section.notes(endian, co_bytes).context(ElfParseSnafu)? {
            while let Some(note) = notes.next().context(ElfParseSnafu)? {
                if note.n_type(endian) == NT_AMDGPU_METADATA && note.name() == b"AMDGPU" {
                    return Ok(note.desc());
                }
            }
        }
    }
    for segment in header.program_headers(endian, co_bytes).context(ElfParseSnafu)? {
        if let Some(mut notes) = segment.notes(endian, co_bytes).context(ElfParseSnafu)? {
            while let Some(note) = notes.next().context(ElfParseSnafu)? {
                if note.n_type(endian) == NT_AMDGPU_METADATA && note.name() == b"AMDGPU" {
                    return Ok(note.desc());
                }
            }
        }
    }
    NoMetadataNoteSnafu.fail()
}

/// Decode the msgpack metadata payload into the per-kernel layout. Exposed to the
/// crate so the note-extraction path and unit tests share one decoder.
pub(crate) fn parse_metadata_msgpack(payload: &[u8]) -> Result<Vec<KernelMeta>> {
    let root = Decoder::new(payload).read_value()?;
    let kernels = root
        .map_get("amdhsa.kernels")
        .context(MissingKeySnafu { key: "amdhsa.kernels" })?
        .as_array()
        .context(WrongTypeSnafu { key: "amdhsa.kernels", expected: "array" })?;
    kernels.iter().map(decode_kernel).collect()
}

fn decode_kernel(kernel: &Value) -> Result<KernelMeta> {
    let args = match kernel.map_get(".args") {
        Some(value) => value
            .as_array()
            .context(WrongTypeSnafu { key: ".args", expected: "array" })?
            .iter()
            .map(decode_arg)
            .collect::<Result<Vec<_>>>()?,
        None => Vec::new(),
    };
    Ok(KernelMeta {
        name: get_str(kernel, ".name")?.to_string(),
        symbol: get_str(kernel, ".symbol")?.to_string(),
        kernarg_segment_size: get_u64(kernel, ".kernarg_segment_size")?,
        args,
    })
}

fn decode_arg(arg: &Value) -> Result<KernelArg> {
    Ok(KernelArg {
        name: arg.map_get(".name").and_then(Value::as_str).map(str::to_string),
        offset: get_u64(arg, ".offset")?,
        size: get_u64(arg, ".size")?,
        value_kind: ValueKind::from(get_str(arg, ".value_kind")?),
        address_space: arg.map_get(".address_space").and_then(Value::as_str).map(str::to_string),
    })
}

fn get_str<'a>(map: &'a Value, key: &'static str) -> Result<&'a str> {
    map.map_get(key).context(MissingKeySnafu { key })?.as_str().context(WrongTypeSnafu { key, expected: "string" })
}

fn get_u64(map: &Value, key: &'static str) -> Result<u64> {
    map.map_get(key).context(MissingKeySnafu { key })?.as_u64().context(WrongTypeSnafu { key, expected: "uint" })
}

// ── Minimal msgpack reader ───────────────────────────────────────────────────
//
// Covers exactly the subset AMDGPU metadata emits (maps, arrays, strings, uints,
// ints, bool, nil) and skips every other well-formed value (float/bin/ext) so an
// unknown key never derails the parse. Strings borrow from the payload.

/// A decoded msgpack value. Non-load-bearing types (float/bin/ext) collapse to
/// [`Value::Skipped`]: still consumed exactly, but not retained.
enum Value<'a> {
    Nil,
    Uint(u64),
    Int(i64),
    Str(&'a str),
    Array(Vec<Value<'a>>),
    Map(Vec<(Value<'a>, Value<'a>)>),
    Skipped,
}

impl<'a> Value<'a> {
    fn as_str(&self) -> Option<&'a str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Uint(u) => Some(*u),
            // AMDHSA encodes offsets/sizes as unsigned, but accept a
            // non-negative signed int too so a small-int encoding still decodes.
            Value::Int(i) if *i >= 0 => Some(*i as u64),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Value<'a>]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }

    fn map_get(&self, key: &str) -> Option<&Value<'a>> {
        match self {
            Value::Map(entries) => entries.iter().find_map(|(k, v)| match k {
                Value::Str(s) if *s == key => Some(v),
                _ => None,
            }),
            _ => None,
        }
    }
}

struct Decoder<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.saturating_add(n);
        let slice =
            self.buf.get(self.pos..end).context(TruncatedSnafu { offset: self.pos, needed: n, len: self.buf.len() })?;
        self.pos = end;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn be_u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.take(2)?.try_into().expect("2 bytes")))
    }

    fn be_u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().expect("4 bytes")))
    }

    fn be_u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().expect("8 bytes")))
    }

    fn read_str(&mut self, n: usize) -> Result<Value<'a>> {
        Ok(Value::Str(std::str::from_utf8(self.take(n)?).context(Utf8Snafu)?))
    }

    fn read_array(&mut self, n: usize) -> Result<Value<'a>> {
        let mut items = Vec::with_capacity(n.min(256));
        for _ in 0..n {
            items.push(self.read_value()?);
        }
        Ok(Value::Array(items))
    }

    fn read_map(&mut self, n: usize) -> Result<Value<'a>> {
        let mut entries = Vec::with_capacity(n.min(256));
        for _ in 0..n {
            let key = self.read_value()?;
            let value = self.read_value()?;
            entries.push((key, value));
        }
        Ok(Value::Map(entries))
    }

    fn read_value(&mut self) -> Result<Value<'a>> {
        let at = self.pos;
        let marker = self.byte()?;
        Ok(match marker {
            // positive fixint / negative fixint
            0x00..=0x7f => Value::Uint(marker as u64),
            0xe0..=0xff => Value::Int((marker as i8) as i64),
            // fixmap / fixarray / fixstr
            0x80..=0x8f => self.read_map((marker & 0x0f) as usize)?,
            0x90..=0x9f => self.read_array((marker & 0x0f) as usize)?,
            0xa0..=0xbf => self.read_str((marker & 0x1f) as usize)?,
            // nil / bool (bools are consumed but never queried)
            0xc0 => Value::Nil,
            0xc2 | 0xc3 => Value::Skipped,
            // bin 8/16/32 (skipped)
            0xc4 => {
                let n = self.byte()? as usize;
                self.take(n)?;
                Value::Skipped
            }
            0xc5 => {
                let n = self.be_u16()? as usize;
                self.take(n)?;
                Value::Skipped
            }
            0xc6 => {
                let n = self.be_u32()? as usize;
                self.take(n)?;
                Value::Skipped
            }
            // ext 8/16/32 (skipped): length + 1 type byte
            0xc7 => {
                let n = self.byte()? as usize;
                self.take(n + 1)?;
                Value::Skipped
            }
            0xc8 => {
                let n = self.be_u16()? as usize;
                self.take(n + 1)?;
                Value::Skipped
            }
            0xc9 => {
                let n = self.be_u32()? as usize;
                self.take(n + 1)?;
                Value::Skipped
            }
            // float 32/64 (skipped)
            0xca => {
                self.take(4)?;
                Value::Skipped
            }
            0xcb => {
                self.take(8)?;
                Value::Skipped
            }
            // uint 8/16/32/64
            0xcc => Value::Uint(self.byte()? as u64),
            0xcd => Value::Uint(self.be_u16()? as u64),
            0xce => Value::Uint(self.be_u32()? as u64),
            0xcf => Value::Uint(self.be_u64()?),
            // int 8/16/32/64
            0xd0 => Value::Int((self.byte()? as i8) as i64),
            0xd1 => Value::Int((self.be_u16()? as i16) as i64),
            0xd2 => Value::Int((self.be_u32()? as i32) as i64),
            0xd3 => Value::Int(self.be_u64()? as i64),
            // fixext 1/2/4/8/16 (skipped): 1 type byte + (1 << (marker-0xd4)) data
            0xd4..=0xd8 => {
                self.take(1 + (1usize << (marker - 0xd4)))?;
                Value::Skipped
            }
            // str 8/16/32
            0xd9 => {
                let n = self.byte()? as usize;
                self.read_str(n)?
            }
            0xda => {
                let n = self.be_u16()? as usize;
                self.read_str(n)?
            }
            0xdb => {
                let n = self.be_u32()? as usize;
                self.read_str(n)?
            }
            // array 16/32
            0xdc => {
                let n = self.be_u16()? as usize;
                self.read_array(n)?
            }
            0xdd => {
                let n = self.be_u32()? as usize;
                self.read_array(n)?
            }
            // map 16/32
            0xde => {
                let n = self.be_u16()? as usize;
                self.read_map(n)?
            }
            0xdf => {
                let n = self.be_u32()? as usize;
                self.read_map(n)?
            }
            // 0xc1 is never a valid msgpack marker.
            0xc1 => return BadMsgpackSnafu { offset: at, reason: "reserved marker 0xc1".to_string() }.fail(),
        })
    }
}
