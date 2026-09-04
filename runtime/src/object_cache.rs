//! Persistent, content-addressed compiled-object storage.
//!
//! Entries are intentionally schema-specific: old formats are cache misses, not
//! migration inputs. The store owns no process-global state, so callers can
//! disable it or drop it without affecting compiler/runtime lifetime.
//!
//! Publication is atomic (temp file + rename) but never fsynced: this is a
//! build cache, so an entry torn by a power loss is just a miss — the payload
//! digest check in `decode_entry` rejects it and the file is removed. Nothing
//! synchronises concurrent compilers of one key: both publish, the last
//! rename wins, and the contents are identical.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use sha2::{Digest, Sha256};

use crate::{Error, Result};

pub const OBJECT_CACHE_SCHEMA: u32 = 1;
const MAGIC: &[u8; 16] = b"SVODOBJCACHE\0\0\0\0";
const HEADER_LEN: usize = MAGIC.len() + 4 + 32 + 32 + 8;
const DEFAULT_MAX_BYTES: u64 = 1024 * 1024 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Every compiler property that can change emitted object bytes or their ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilerIdentity {
    pub schema: u32,
    pub backend: String,
    pub target_architecture: String,
    pub toolchain: String,
    pub flags: Vec<String>,
    pub abi: String,
    pub object_format: String,
}

impl CompilerIdentity {
    pub fn cache_key(&self) -> String {
        let schema = self.schema.to_le_bytes();
        let mut fields = vec![schema.as_slice()];
        fields.extend(self.fields());
        format!("{}:{}", self.backend, hex(&digest_fields(fields)))
    }

    fn fields(&self) -> Vec<&[u8]> {
        let mut fields = vec![
            self.backend.as_bytes(),
            self.target_architecture.as_bytes(),
            self.toolchain.as_bytes(),
            self.abi.as_bytes(),
            self.object_format.as_bytes(),
        ];
        fields.extend(self.flags.iter().map(String::as_bytes));
        fields
    }
}

/// Content address for one compiler output. The source itself is never used as
/// a filename; only its digest participates in the canonical key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectCacheKey {
    pub source_digest: [u8; 32],
    pub compiler: CompilerIdentity,
}

impl ObjectCacheKey {
    pub fn new(source: &[u8], compiler: CompilerIdentity) -> Self {
        Self { source_digest: Sha256::digest(source).into(), compiler }
    }

    pub fn digest(&self) -> [u8; 32] {
        let schema = self.compiler.schema.to_le_bytes();
        let mut fields = vec![schema.as_slice(), self.source_digest.as_slice()];
        fields.extend(self.compiler.fields());
        digest_fields(fields)
    }
}

/// A host-owned cache handle. Dropping it closes all cache state; no worker or
/// global map survives the handle.
#[derive(Debug)]
pub struct ObjectCache {
    root: PathBuf,
    max_bytes: u64,
}

impl ObjectCache {
    pub fn open(root: impl Into<PathBuf>, max_bytes: u64) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|e| cache_io("create cache directory", e))?;
        Ok(Self { root, max_bytes })
    }

    /// Open the default cache. `SVOD_OBJECT_CACHE=0` is the explicit host-side
    /// off switch. `SVOD_OBJECT_CACHE_DIR` and `SVOD_OBJECT_CACHE_MAX_BYTES`
    /// override location and byte budget.
    pub fn from_env() -> Result<Option<Self>> {
        if std::env::var("SVOD_OBJECT_CACHE").as_deref() == Ok("0") {
            return Ok(None);
        }
        let root = if let Some(path) = std::env::var_os("SVOD_OBJECT_CACHE_DIR") {
            PathBuf::from(path)
        } else if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
            PathBuf::from(path).join("svod/objects")
        } else if let Some(path) = std::env::var_os("HOME") {
            PathBuf::from(path).join(".cache/svod/objects")
        } else {
            return Ok(None);
        };
        let max_bytes = match std::env::var("SVOD_OBJECT_CACHE_MAX_BYTES") {
            Ok(value) => value.parse().map_err(|_| Error::JitCompilation {
                reason: format!("invalid SVOD_OBJECT_CACHE_MAX_BYTES={value:?}"),
            })?,
            Err(_) => DEFAULT_MAX_BYTES,
        };
        match Self::open(root, max_bytes) {
            Ok(cache) => Ok(Some(cache)),
            // A store we cannot even create is a missing store, not a broken
            // compiler: tinygrad's `diskcache_get` (helpers.py:415-424)
            // swallows the store's own errors and reports a miss.
            Err(error) => {
                warn_degraded("open object cache", &error);
                Ok(None)
            }
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Read or produce one validated object. Both cached and newly compiled
    /// bytes pass the backend validator before they can reach a runtime loader.
    pub fn get_or_compile<V, C>(&self, key: &ObjectCacheKey, validate: V, compile: C) -> Result<Vec<u8>>
    where
        V: Fn(&[u8]) -> Result<()>,
        C: FnOnce() -> Result<Vec<u8>>,
    {
        let digest = key.digest();
        let path = self.entry_path(&digest);
        if let Some(bytes) = self.read_validated(&path, &digest, &validate) {
            return Ok(bytes);
        }
        let bytes = compile()?;
        validate(&bytes)?;
        let encoded = encode_entry(&digest, &bytes);
        if self.max_bytes > 0 && encoded.len() as u64 <= self.max_bytes {
            if let Err(error) = atomic_write(&path, &encoded) {
                warn_degraded("publish object cache entry", &error);
            }
            if let Err(error) = self.evict_to_budget(path.file_name()) {
                warn_degraded("evict object cache entries", &error);
            }
        }
        Ok(bytes)
    }

    /// Persist deterministic compiler probes separately from evictable object
    /// entries. This lets a warm process reconstruct a versioned object key
    /// without invoking the compiler just to run `--version` or `-###`.
    pub(crate) fn get_or_create_probe<C>(&self, namespace: &str, input: &[u8], create: C) -> Result<Vec<u8>>
    where
        C: FnOnce() -> Result<Vec<u8>>,
    {
        let digest = digest_fields([namespace.as_bytes(), input]);
        let path = self.root.join(format!("probe-{}-{}.data", sanitize(namespace), hex(&digest)));
        let cached = |path: &Path| {
            read_entry(path, &digest).unwrap_or_else(|error| {
                warn_degraded("read compiler probe", &error);
                None
            })
        };
        if let Some(bytes) = cached(&path) {
            return Ok(bytes);
        }
        let bytes = create()?;
        if bytes.is_empty() {
            return Err(Error::JitCompilation { reason: format!("empty {namespace} compiler probe") });
        }
        if let Err(error) = atomic_write(&path, &encode_entry(&digest, &bytes)) {
            warn_degraded("publish compiler probe", &error);
        }
        Ok(bytes)
    }

    pub(crate) fn entry_path(&self, digest: &[u8; 32]) -> PathBuf {
        self.root.join(format!("{}.obj", hex(digest)))
    }

    fn read_validated<V>(&self, path: &Path, digest: &[u8; 32], validate: &V) -> Option<Vec<u8>>
    where
        V: Fn(&[u8]) -> Result<()>,
    {
        let bytes = read_entry(path, digest).unwrap_or_else(|error| {
            warn_degraded("read object cache entry", &error);
            None
        })?;
        if validate(&bytes).is_ok() {
            return Some(bytes);
        }
        let _ = fs::remove_file(path);
        None
    }

    fn evict_to_budget(&self, protected: Option<&std::ffi::OsStr>) -> Result<()> {
        if self.max_bytes == 0 {
            return Ok(());
        }
        // Concurrent sweeps race benignly: a file the other one already
        // removed is `NotFound` below, which counts as evicted.
        let mut entries = Vec::new();
        let mut total = 0u64;
        for item in fs::read_dir(&self.root).map_err(|e| cache_io("scan cache for eviction", e))? {
            let item = item.map_err(|e| cache_io("read cache directory entry", e))?;
            let path = item.path();
            match path.extension().and_then(|value| value.to_str()) {
                Some("obj") => {}
                // Publication locks from earlier schemas; nothing holds them now.
                Some("lock") => {
                    let _ = fs::remove_file(&path);
                    continue;
                }
                _ => continue,
            }
            let metadata = match item.metadata() {
                Ok(metadata) if metadata.is_file() => metadata,
                _ => continue,
            };
            total = total.saturating_add(metadata.len());
            entries.push((metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH), metadata.len(), path));
        }
        entries.sort_by_key(|(modified, _, _)| *modified);
        for (_, size, path) in entries {
            if total <= self.max_bytes {
                break;
            }
            if protected.is_some_and(|name| path.file_name() == Some(name)) {
                continue;
            }
            match fs::remove_file(&path) {
                Ok(()) => total = total.saturating_sub(size),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => total = total.saturating_sub(size),
                Err(error) => return Err(cache_io("evict cache entry", error)),
            }
        }
        Ok(())
    }
}

fn read_entry(path: &Path, expected_key: &[u8; 32]) -> Result<Option<Vec<u8>>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(cache_io("open cache entry", error)),
    };
    let mut encoded = Vec::new();
    if file.read_to_end(&mut encoded).is_err() {
        let _ = fs::remove_file(path);
        return Ok(None);
    }
    let Some(payload) = decode_entry(&encoded, expected_key) else {
        let _ = fs::remove_file(path);
        return Ok(None);
    };
    Ok(Some(payload.to_vec()))
}

fn encode_entry(key: &[u8; 32], payload: &[u8]) -> Vec<u8> {
    let payload_digest: [u8; 32] = Sha256::digest(payload).into();
    let mut encoded = Vec::with_capacity(HEADER_LEN + payload.len());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(&OBJECT_CACHE_SCHEMA.to_le_bytes());
    encoded.extend_from_slice(key);
    encoded.extend_from_slice(&payload_digest);
    encoded.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    encoded.extend_from_slice(payload);
    encoded
}

fn decode_entry<'a>(encoded: &'a [u8], expected_key: &[u8; 32]) -> Option<&'a [u8]> {
    if encoded.len() < HEADER_LEN || &encoded[..MAGIC.len()] != MAGIC {
        return None;
    }
    let schema = u32::from_le_bytes(encoded[16..20].try_into().ok()?);
    if schema != OBJECT_CACHE_SCHEMA || &encoded[20..52] != expected_key {
        return None;
    }
    let expected_payload_digest = &encoded[52..84];
    let len = usize::try_from(u64::from_le_bytes(encoded[84..92].try_into().ok()?)).ok()?;
    let payload = encoded.get(HEADER_LEN..HEADER_LEN.checked_add(len)?)?;
    if HEADER_LEN + len != encoded.len() || Sha256::digest(payload).as_slice() != expected_payload_digest {
        return None;
    }
    Some(payload)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::JitCompilation { reason: format!("cache path has no parent: {}", path.display()) })?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name().and_then(|name| name.to_str()).unwrap_or("entry"),
        std::process::id(),
        sequence
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| cache_io("create cache temp", e))?;
        file.write_all(bytes).map_err(|e| cache_io("write cache temp", e))?;
        drop(file);
        fs::rename(&temp, path).map_err(|e| cache_io("publish cache entry", e))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn digest_fields<'a>(fields: impl IntoIterator<Item = &'a [u8]>) -> [u8; 32] {
    let mut digest = Sha256::new();
    for field in fields {
        digest.update((field.len() as u64).to_le_bytes());
        digest.update(field);
    }
    digest.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0xf) as usize] as char);
    }
    output
}

fn sanitize(value: &str) -> String {
    value.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '-' { ch } else { '_' }).collect()
}

/// The object cache is advisory storage: every failure to read, publish or
/// evict degrades to a miss instead of failing the caller's compile, the way
/// tinygrad's `diskcache_get` (`helpers.py:415-424`) reports a miss when the
/// store itself errors.
fn warn_degraded(action: &str, error: &Error) {
    tracing::warn!(target: "svod_runtime::object_cache", action, %error, "object cache degraded to a miss");
}

fn cache_io(action: &'static str, source: std::io::Error) -> Error {
    Error::Jit { source: Box::new(source.into()), context: action }
}
