//! MSL → metallib through Apple's private `MTLCodeGenService` (the same path
//! tinygrad uses), with the public `newLibraryWithSource:options:error:` as
//! the fallback when the private framework is unavailable.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::null_mut;
use std::sync::OnceLock;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use super::device::MetalDevice;
use super::objc::{
    AutoreleasePool, BlockLiteral, Id, MTL_MATH_MODE_SAFE, NSInteger, Objc, ObjcBool, ObjcId, mtl_compiler,
    ns_error_message, ns_string, objc,
};
use crate::{Error, Result};

/// Undocumented request type for "compile MSL source to a metallib".
const REQUEST_TYPE_COMPILE: c_int = 13;
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(60);
const METALLIB_MAGIC: &[u8] = b"MTLB";
const METALLIB_TRAILER: &[u8] = b"ENDT";

fn runtime(message: String) -> Error {
    Error::Runtime { message }
}

/// `kern.osproductversion`, e.g. `"26.6.2"`; `None` off Darwin.
pub fn macos_product_version() -> Option<String> {
    objc().ok()?.sysctl_string(c"kern.osproductversion")
}

fn macos_major() -> u32 {
    macos_product_version().and_then(|version| version.split('.').next()?.parse().ok()).unwrap_or(0)
}

/// Newest MSL standard the running macOS accepts; compute semantics are
/// unchanged across 2.0–2.4, so 2.0 is the floor.
pub(crate) fn std_for_major(major: u32) -> &'static str {
    match major {
        26.. => "metal4.0",
        14..=25 => "metal3.1",
        13 => "metal3.0",
        _ => "macos-metal2.0",
    }
}

pub fn metal_std_flag() -> &'static str {
    std_for_major(macos_major())
}

/// Whether the private compiler service is loadable (fast path).
pub fn codegen_service_available() -> bool {
    objc().is_ok() && mtl_compiler().is_some()
}

/// `<u64 src len><u64 params len><src, NUL-padded to 4><params, NUL>`.
pub(crate) fn pack_request(source: &[u8], params: &[u8]) -> Vec<u8> {
    let padded_len = (source.len() + 1).next_multiple_of(4);
    let mut request = Vec::with_capacity(16 + padded_len + params.len() + 1);
    request.extend_from_slice(&(padded_len as u64).to_le_bytes());
    request.extend_from_slice(&((params.len() + 1) as u64).to_le_bytes());
    request.extend_from_slice(source);
    request.resize(16 + padded_len, 0);
    request.extend_from_slice(params);
    request.push(0);
    request
}

/// The metallib follows a header and a warnings blob whose sizes sit at bytes
/// 8 and 12 of the reply.
pub(crate) fn extract_metallib(reply: &[u8]) -> std::result::Result<Vec<u8>, String> {
    if reply.len() < 16 {
        return Err(format!("MTLCodeGenService reply is {} bytes, shorter than its header", reply.len()));
    }
    let word = |at: usize| u32::from_le_bytes(reply[at..at + 4].try_into().expect("4 bytes")) as usize;
    let start = word(8) + word(12);
    reply
        .get(start..)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| format!("MTLCodeGenService reply payload offset {start} exceeds reply length {}", reply.len()))
}

/// Format and entry-point plausibility check for both payload forms. The
/// metallib function table stores names as raw strings, so a byte scan is a
/// necessary condition; `newFunctionWithName:` at load is authoritative.
pub fn validate_metallib(bytes: &[u8], kernel_name: &str) -> Result<()> {
    if bytes.starts_with(METALLIB_MAGIC) {
        if bytes.len() < 8 || !bytes.ends_with(METALLIB_TRAILER) {
            return Err(runtime("metallib does not end with the ENDT trailer".into()));
        }
        let name = kernel_name.as_bytes();
        if !bytes.windows(name.len()).any(|window| window == name) {
            return Err(runtime(format!("metallib does not contain the kernel entry point {kernel_name:?}")));
        }
        return Ok(());
    }
    let source =
        std::str::from_utf8(bytes).map_err(|_| runtime("Metal payload is neither a metallib nor UTF-8 MSL".into()))?;
    if !source.contains(kernel_name) {
        return Err(runtime(format!("MSL payload does not define the kernel entry point {kernel_name:?}")));
    }
    Ok(())
}

struct CodeGenService {
    /// The service handle; the framework's thread-safety is undocumented, so
    /// requests are serialized.
    handle: Mutex<usize>,
}

static SERVICE: OnceLock<Option<CodeGenService>> = OnceLock::new();

fn service() -> Option<&'static CodeGenService> {
    SERVICE
        .get_or_init(|| {
            let api = mtl_compiler()?;
            // SAFETY: NUL-terminated service name.
            let handle = unsafe { (api.create)(c"svod".as_ptr()) };
            (!handle.is_null()).then(|| CodeGenService { handle: Mutex::new(handle as usize) })
        })
        .as_ref()
}

#[derive(Default)]
struct CompileReply {
    result: Mutex<Option<std::result::Result<Vec<u8>, String>>>,
    done: Condvar,
}

/// The block's `invoke`; the block's context is the `CompileReply` to fill.
unsafe extern "C" fn on_compiled(block: *mut c_void, error: i32, data: *const u8, len: usize, message: *const c_char) {
    // SAFETY: `block` is the `BlockLiteral<*const CompileReply>` built in `compile_msl`.
    let reply = unsafe { &*(*(block as *const BlockLiteral<*const CompileReply>)).context };
    let result = if error == 0 {
        if data.is_null() {
            Err("MTLCodeGenService reported success without data".to_string())
        } else {
            // SAFETY: the service hands us `len` readable bytes for the duration of the callback.
            extract_metallib(unsafe { std::slice::from_raw_parts(data, len) })
        }
    } else if message.is_null() {
        Err(format!("MTLCodeGenService failed with code {error} and no message"))
    } else {
        // SAFETY: NUL-terminated diagnostic owned by the service.
        Err(unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned())
    };
    *reply.result.lock() = Some(result);
    reply.done.notify_all();
}

/// Compile MSL to a metallib with the private compiler service. `params` is
/// the verbatim compiler flag string.
pub fn compile_msl(source: &str, params: &str) -> Result<Vec<u8>> {
    let objc = objc()?;
    let api = mtl_compiler()
        .ok_or_else(|| Error::DeviceUnavailable { reason: "MTLCompiler.framework unavailable".into() })?;
    let service =
        service().ok_or_else(|| Error::DeviceUnavailable { reason: "MTLCodeGenServiceCreate failed".into() })?;
    let request = pack_request(source.as_bytes(), params.as_bytes());

    let reply = CompileReply::default();
    let descriptor = BlockLiteral::<*const CompileReply>::descriptor();
    let block = BlockLiteral::new(objc, on_compiled as *const c_void, &descriptor, &raw const reply);
    {
        let handle = service.handle.lock();
        // SAFETY: the request buffer and the block outlive the call; the block
        // layout follows clang's Block-ABI-Apple.
        unsafe {
            (api.build_request)(
                *handle as *mut c_void,
                null_mut(),
                REQUEST_TYPE_COMPILE,
                request.as_ptr().cast(),
                request.len(),
                (&raw const block).cast(),
            )
        };
    }
    let mut result = reply.result.lock();
    if result.is_none() {
        reply.done.wait_for(&mut result, CALLBACK_TIMEOUT);
    }
    let bytes = result
        .take()
        .ok_or_else(|| runtime("MTLCodeGenService did not invoke its callback".into()))?
        .map_err(|diagnostic| runtime(format!("Metal compile failed: {diagnostic}")))?;
    if !bytes.starts_with(METALLIB_MAGIC) || !bytes.ends_with(METALLIB_TRAILER) {
        return Err(runtime("MTLCodeGenService returned an invalid metallib".into()));
    }
    Ok(bytes)
}

/// `MTLCompileOptions` with fast math disabled (IEEE semantics, as `-fno-fast-math`).
fn compile_options(objc: &Objc) -> Result<ObjcId> {
    let options = objc.new_object(objc.classes.compile_options)?;
    // SAFETY: `setMathMode:` (macOS 15+) takes NSInteger; `setFastMathEnabled:` takes BOOL.
    unsafe {
        if macos_major() >= 15 {
            objc.send1::<NSInteger, ()>(options.as_raw(), objc.sels.set_math_mode, MTL_MATH_MODE_SAFE);
        } else {
            objc.send1::<ObjcBool, ()>(options.as_raw(), objc.sels.set_fast_math_enabled, 0);
        }
    }
    Ok(options)
}

/// `newLibraryWithSource:options:error:` on the device, retained.
pub(crate) fn new_library_from_source(dev: &MetalDevice, source: &str) -> Result<ObjcId> {
    let objc = dev.objc();
    let _pool = AutoreleasePool::push(objc);
    let source = ns_string(objc, source)?;
    let options = compile_options(objc)?;
    let mut error: Id = null_mut();
    // SAFETY: (NSString, MTLCompileOptions, NSError**) → +1 library or nil.
    let library = unsafe {
        objc.send3::<Id, Id, *mut Id, Id>(
            dev.mtl(),
            objc.sels.new_library_with_source_options_error,
            source.as_raw(),
            options.as_raw(),
            &mut error,
        )
    };
    // SAFETY: +1 or nil; autoreleased NSError or nil.
    unsafe { ObjcId::adopt(library) }.ok_or_else(|| {
        runtime(format!("Metal compile failed: {}", unsafe { ns_error_message(objc, error) }.unwrap_or_default()))
    })
}

/// Public-API fallback: compiles for real so diagnostics surface, then returns
/// the source itself as the payload the loader recompiles from.
pub fn compile_msl_public(dev: &MetalDevice, source: &str) -> Result<Vec<u8>> {
    new_library_from_source(dev, source)?;
    Ok(source.as_bytes().to_vec())
}
