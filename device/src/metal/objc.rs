//! Runtime-bound Objective-C / Metal shim.
//!
//! Nothing here links against Apple frameworks: `libobjc`, `libSystem`,
//! `CoreGraphics`, `Metal.framework` and the private `MTLCompiler.framework`
//! are `dlopen`ed on first use, so the module compiles on every host and
//! simply reports [`crate::Error::DeviceUnavailable`] where they are absent.
//!
//! Message sends go through `objc_msgSend` cast to the concrete C signature of
//! each selector (`send0..send4`); rustc then emits the platform C ABI,
//! including by-value `MTLSize` arguments. **No selector used here returns a
//! struct** — that would need `objc_msgSend_stret` on x86-64 — so every limit
//! is read through its scalar accessor (`maxTotalThreadsPerThreadgroup`, not
//! `maxThreadsPerThreadgroup`).

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::OnceLock;

use libloading::Library;

use crate::{Error, Result};

pub(crate) type Id = *mut c_void;
pub(crate) type Sel = *mut c_void;
pub(crate) type Class = *mut c_void;
pub(crate) type NSUInteger = u64;
pub(crate) type NSInteger = i64;
/// `BOOL` is `signed char` on 64-bit Darwin; the upper register bits are
/// unspecified, so it must not be read as a wider integer.
pub(crate) type ObjcBool = c_char;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MTLSize {
    pub width: NSUInteger,
    pub height: NSUInteger,
    pub depth: NSUInteger,
}

impl From<[usize; 3]> for MTLSize {
    fn from([width, height, depth]: [usize; 3]) -> Self {
        Self { width: width as u64, height: height as u64, depth: depth as u64 }
    }
}

/// `NSRange`, passed by value (two words in registers; never returned).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NSRange {
    pub location: NSUInteger,
    pub length: NSUInteger,
}

pub(crate) const MTL_RESOURCE_STORAGE_MODE_SHARED: NSUInteger = 0;
/// `MTLResourceCPUCacheModeDefaultCache`, the only option an indirect command buffer needs.
pub(crate) const MTL_RESOURCE_OPTIONS_DEFAULT: NSUInteger = 0;
/// `MTLIndirectCommandTypeConcurrentDispatch`.
pub(crate) const MTL_INDIRECT_COMMAND_TYPE_CONCURRENT_DISPATCH: NSUInteger = 1 << 5;
/// `MTLResourceUsageRead | MTLResourceUsageWrite`.
pub(crate) const MTL_RESOURCE_USAGE_READ_WRITE: NSUInteger = 1 | 2;
pub(crate) const MTL_PIPELINE_OPTION_NONE: NSUInteger = 0;
pub(crate) const MTL_COMMAND_BUFFER_STATUS_COMPLETED: NSUInteger = 4;
pub(crate) const MTL_MATH_MODE_SAFE: NSInteger = 0;
/// `MTLGPUFamilyApple1..Apple9`, highest first.
pub(crate) const MTL_GPU_FAMILY_APPLE: [(NSInteger, &str); 9] = [
    (1009, "Apple9"),
    (1008, "Apple8"),
    (1007, "Apple7"),
    (1006, "Apple6"),
    (1005, "Apple5"),
    (1004, "Apple4"),
    (1003, "Apple3"),
    (1002, "Apple2"),
    (1001, "Apple1"),
];
pub(crate) const MTL_GPU_FAMILY_MAC2: (NSInteger, &str) = (2002, "Mac2");

const LIBOBJC: &str = "/usr/lib/libobjc.A.dylib";
const LIBSYSTEM: &str = "/usr/lib/libSystem.B.dylib";
const CORE_GRAPHICS: &str = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics";
const METAL: &str = "/System/Library/Frameworks/Metal.framework/Metal";
const MTL_COMPILER: &str = "/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler";

fn unavailable(reason: String) -> Error {
    Error::DeviceUnavailable { reason }
}

fn load(path: &str) -> Result<Library> {
    // SAFETY: these are Apple system libraries whose initializers are safe to
    // run from any thread; the paths resolve through the dyld shared cache.
    unsafe { Library::new(path) }.map_err(|error| unavailable(format!("cannot load {path}: {error}")))
}

/// Resolve `name` as a function pointer of type `T`.
fn sym<T: Copy>(lib: &Library, path: &str, name: &CStr) -> Result<T> {
    // SAFETY: every call site declares `T` from the symbol's C prototype.
    let symbol = unsafe { lib.get::<T>(name.to_bytes_with_nul()) }
        .map_err(|error| unavailable(format!("{path} has no symbol {name:?}: {error}")))?;
    Ok(*symbol)
}

/// Address of a data symbol (the symbol itself, not the value stored there).
fn data_sym(lib: &Library, path: &str, name: &CStr) -> Result<*const c_void> {
    // SAFETY: only the symbol's address is taken; nothing is read through it.
    // `Symbol<T>` dereferences to the symbol location itself, so the reference's
    // address is the symbol's address (the class object of the block).
    let symbol = unsafe { lib.get::<*const c_void>(name.to_bytes_with_nul()) }
        .map_err(|error| unavailable(format!("{path} has no symbol {name:?}: {error}")))?;
    Ok((&*symbol as *const *const c_void).cast())
}

macro_rules! selectors {
    ($($field:ident = $name:literal;)*) => {
        /// Every selector the backend sends, registered once.
        pub(crate) struct Sels { $(pub(crate) $field: Sel,)* }

        impl Sels {
            fn register(register_name: unsafe extern "C" fn(*const c_char) -> Sel) -> Self {
                // SAFETY: `sel_registerName` takes a NUL-terminated C string.
                Self { $($field: unsafe { register_name(concat!($name, "\0").as_ptr().cast()) },)* }
            }
        }
    };
}

selectors! {
    new = "new";
    string_with_utf8 = "stringWithUTF8String:";
    utf8_string = "UTF8String";
    localized_description = "localizedDescription";
    name = "name";
    label = "label";
    set_label = "setLabel:";
    supports_family = "supportsFamily:";
    new_command_queue_with_max = "newCommandQueueWithMaxCommandBufferCount:";
    new_buffer_with_length_options = "newBufferWithLength:options:";
    contents = "contents";
    command_buffer = "commandBuffer";
    compute_command_encoder = "computeCommandEncoder";
    commit = "commit";
    wait_until_completed = "waitUntilCompleted";
    status = "status";
    error = "error";
    set_compute_pipeline_state = "setComputePipelineState:";
    set_buffer_offset_at_index = "setBuffer:offset:atIndex:";
    set_bytes_length_at_index = "setBytes:length:atIndex:";
    dispatch_threadgroups = "dispatchThreadgroups:threadsPerThreadgroup:";
    end_encoding = "endEncoding";
    new_library_with_data_error = "newLibraryWithData:error:";
    new_library_with_source_options_error = "newLibraryWithSource:options:error:";
    new_function_with_name = "newFunctionWithName:";
    set_compute_function = "setComputeFunction:";
    set_support_indirect_command_buffers = "setSupportIndirectCommandBuffers:";
    new_compute_pipeline_state = "newComputePipelineStateWithDescriptor:options:reflection:error:";
    max_total_threads_per_threadgroup = "maxTotalThreadsPerThreadgroup";
    thread_execution_width = "threadExecutionWidth";
    static_threadgroup_memory_length = "staticThreadgroupMemoryLength";
    set_fast_math_enabled = "setFastMathEnabled:";
    set_math_mode = "setMathMode:";
    gpu_start_time = "GPUStartTime";
    gpu_end_time = "GPUEndTime";
    // Indirect command buffers (graph replay).
    set_command_types = "setCommandTypes:";
    set_inherit_buffers = "setInheritBuffers:";
    set_inherit_pipeline_state = "setInheritPipelineState:";
    set_max_kernel_buffer_bind_count = "setMaxKernelBufferBindCount:";
    new_indirect_command_buffer = "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:";
    indirect_compute_command_at_index = "indirectComputeCommandAtIndex:";
    set_kernel_buffer_offset_at_index = "setKernelBuffer:offset:atIndex:";
    concurrent_dispatch_threadgroups = "concurrentDispatchThreadgroups:threadsPerThreadgroup:";
    set_barrier = "setBarrier";
    use_resources_count_usage = "useResources:count:usage:";
    execute_commands_in_buffer_with_range = "executeCommandsInBuffer:withRange:";
}

/// The Objective-C classes the backend instantiates.
pub(crate) struct Classes {
    pub(crate) ns_string: Class,
    pub(crate) compute_pipeline_descriptor: Class,
    pub(crate) compile_options: Class,
    pub(crate) indirect_command_buffer_descriptor: Class,
}

/// The loaded runtime: libobjc, libSystem, CoreGraphics and Metal.
pub(crate) struct Objc {
    objc_msg_send: unsafe extern "C" fn(),
    retain: unsafe extern "C" fn(Id) -> Id,
    release: unsafe extern "C" fn(Id),
    pool_push: unsafe extern "C" fn() -> *mut c_void,
    pool_pop: unsafe extern "C" fn(*mut c_void),
    pub(crate) create_system_default_device: unsafe extern "C" fn() -> Id,
    pub(crate) dispatch_data_create: unsafe extern "C" fn(*const c_void, usize, *mut c_void, *mut c_void) -> Id,
    sysctlbyname: unsafe extern "C" fn(*const c_char, *mut c_void, *mut usize, *mut c_void, usize) -> c_int,
    /// `&_NSConcreteStackBlock`, the `isa` of a stack-allocated block literal.
    pub(crate) stack_block_isa: *const c_void,
    pub(crate) sels: Sels,
    pub(crate) classes: Classes,
    // Declared last so the function pointers above never outlive their
    // libraries; process-global anyway.
    _libraries: Vec<Library>,
}

// SAFETY: immutable after construction; the pointers are into libraries that
// live for the rest of the process, and the Metal objects created through them
// are documented thread-safe (device, queue, buffers, pipeline states).
unsafe impl Send for Objc {}
unsafe impl Sync for Objc {}

impl Objc {
    fn load() -> Result<Self> {
        let objc = load(LIBOBJC)?;
        let system = load(LIBSYSTEM)?;
        // CoreGraphics must be resident before `MTLCreateSystemDefaultDevice`.
        let core_graphics = load(CORE_GRAPHICS)?;
        let metal = load(METAL)?;

        let get_class: unsafe extern "C" fn(*const c_char) -> Class = sym(&objc, LIBOBJC, c"objc_getClass")?;
        let register_name: unsafe extern "C" fn(*const c_char) -> Sel = sym(&objc, LIBOBJC, c"sel_registerName")?;
        let class = |name: &CStr| -> Result<Class> {
            // SAFETY: NUL-terminated class name.
            let class = unsafe { get_class(name.as_ptr()) };
            if class.is_null() { Err(unavailable(format!("Objective-C class {name:?} not found"))) } else { Ok(class) }
        };
        Ok(Self {
            objc_msg_send: sym(&objc, LIBOBJC, c"objc_msgSend")?,
            retain: sym(&objc, LIBOBJC, c"objc_retain")?,
            release: sym(&objc, LIBOBJC, c"objc_release")?,
            pool_push: sym(&objc, LIBOBJC, c"objc_autoreleasePoolPush")?,
            pool_pop: sym(&objc, LIBOBJC, c"objc_autoreleasePoolPop")?,
            create_system_default_device: sym(&metal, METAL, c"MTLCreateSystemDefaultDevice")?,
            dispatch_data_create: sym(&system, LIBSYSTEM, c"dispatch_data_create")?,
            sysctlbyname: sym(&system, LIBSYSTEM, c"sysctlbyname")?,
            stack_block_isa: data_sym(&system, LIBSYSTEM, c"_NSConcreteStackBlock")?,
            sels: Sels::register(register_name),
            classes: Classes {
                ns_string: class(c"NSString")?,
                compute_pipeline_descriptor: class(c"MTLComputePipelineDescriptor")?,
                compile_options: class(c"MTLCompileOptions")?,
                indirect_command_buffer_descriptor: class(c"MTLIndirectCommandBufferDescriptor")?,
            },
            _libraries: vec![objc, system, core_graphics, metal],
        })
    }

    /// # Safety
    ///
    /// `R` must be the selector's return type and must not be a struct.
    pub(crate) unsafe fn send0<R>(&self, receiver: Id, sel: Sel) -> R {
        // SAFETY: fn-pointer transmute to the selector's C signature.
        let f: unsafe extern "C" fn(Id, Sel) -> R = unsafe { std::mem::transmute(self.objc_msg_send) };
        unsafe { f(receiver, sel) }
    }

    /// # Safety
    ///
    /// Argument and return types must match the selector; `R` must not be a struct.
    pub(crate) unsafe fn send1<A, R>(&self, receiver: Id, sel: Sel, a: A) -> R {
        // SAFETY: as `send0`.
        let f: unsafe extern "C" fn(Id, Sel, A) -> R = unsafe { std::mem::transmute(self.objc_msg_send) };
        unsafe { f(receiver, sel, a) }
    }

    /// # Safety
    ///
    /// Argument and return types must match the selector; `R` must not be a struct.
    pub(crate) unsafe fn send2<A, B, R>(&self, receiver: Id, sel: Sel, a: A, b: B) -> R {
        // SAFETY: as `send0`.
        let f: unsafe extern "C" fn(Id, Sel, A, B) -> R = unsafe { std::mem::transmute(self.objc_msg_send) };
        unsafe { f(receiver, sel, a, b) }
    }

    /// # Safety
    ///
    /// Argument and return types must match the selector; `R` must not be a struct.
    pub(crate) unsafe fn send3<A, B, C, R>(&self, receiver: Id, sel: Sel, a: A, b: B, c: C) -> R {
        // SAFETY: as `send0`.
        let f: unsafe extern "C" fn(Id, Sel, A, B, C) -> R = unsafe { std::mem::transmute(self.objc_msg_send) };
        unsafe { f(receiver, sel, a, b, c) }
    }

    /// # Safety
    ///
    /// Argument and return types must match the selector; `R` must not be a struct.
    pub(crate) unsafe fn send4<A, B, C, D, R>(&self, receiver: Id, sel: Sel, a: A, b: B, c: C, d: D) -> R {
        // SAFETY: as `send0`.
        let f: unsafe extern "C" fn(Id, Sel, A, B, C, D) -> R = unsafe { std::mem::transmute(self.objc_msg_send) };
        unsafe { f(receiver, sel, a, b, c, d) }
    }

    /// `[Class new]`, adopted.
    pub(crate) fn new_object(&self, class: Class) -> Result<ObjcId> {
        // SAFETY: `new` returns a +1 object or nil.
        unsafe { ObjcId::adopt(self.send0::<Id>(class, self.sels.new)) }
            .ok_or_else(|| Error::Runtime { message: "Objective-C `new` returned nil".into() })
    }

    /// Read a `sysctl` string such as `kern.osproductversion`.
    pub(crate) fn sysctl_string(&self, name: &CStr) -> Option<String> {
        let mut buffer = [0u8; 64];
        let mut len = buffer.len();
        // SAFETY: `len` bounds the writable buffer; the name is NUL-terminated.
        let status = unsafe {
            (self.sysctlbyname)(name.as_ptr(), buffer.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0)
        };
        if status != 0 {
            return None;
        }
        let text = &buffer[..len.min(buffer.len())];
        let text = text.split(|byte| *byte == 0).next().unwrap_or(text);
        std::str::from_utf8(text).ok().map(str::to_string)
    }
}

static OBJC: OnceLock<std::result::Result<Objc, String>> = OnceLock::new();

/// The process-wide Objective-C/Metal runtime, or why Metal is unavailable
/// (Linux, or a Mac without the frameworks).
pub(crate) fn objc() -> Result<&'static Objc> {
    OBJC.get_or_init(|| Objc::load().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|reason| unavailable(reason.clone()))
}

/// The private Metal compiler service (`MTLCompiler.framework`).
pub(crate) struct MtlCompilerApi {
    pub(crate) create: unsafe extern "C" fn(*const c_char) -> *mut c_void,
    pub(crate) build_request:
        unsafe extern "C" fn(*mut c_void, *mut c_void, c_int, *const c_void, usize, *const c_void),
    _library: Library,
}

// SAFETY: immutable after construction; callers serialize use of the service handle.
unsafe impl Send for MtlCompilerApi {}
unsafe impl Sync for MtlCompilerApi {}

static MTL_COMPILER_API: OnceLock<Option<MtlCompilerApi>> = OnceLock::new();

/// `None` when the private framework cannot be loaded; callers fall back to
/// the public `newLibraryWithSource:` path.
pub(crate) fn mtl_compiler() -> Option<&'static MtlCompilerApi> {
    MTL_COMPILER_API
        .get_or_init(|| {
            // MTLCompiler.framework loads its own libLLVM RTLD_GLOBAL, which
            // collides with the CPU backend's in-process libLLVM. Load it only
            // if this process's in-process LLVM slot is free (or already ours).
            if !crate::claim_inprocess_llvm("metal") {
                return None;
            }
            let library = load(MTL_COMPILER).ok()?;
            Some(MtlCompilerApi {
                create: sym(&library, MTL_COMPILER, c"MTLCodeGenServiceCreate").ok()?,
                build_request: sym(&library, MTL_COMPILER, c"MTLCodeGenServiceBuildRequest").ok()?,
                _library: library,
            })
        })
        .as_ref()
}

/// An owned (+1) Objective-C object reference; `Drop` releases it. Transparent
/// over `id`, so a `[ObjcId]` doubles as the C array selectors like
/// `useResources:count:usage:` expect.
#[repr(transparent)]
pub struct ObjcId(Id);

impl ObjcId {
    /// Adopt a +1 reference (`new*`, `alloc`, `MTLCreateSystemDefaultDevice`,
    /// `dispatch_data_create`). `None` for nil.
    ///
    /// # Safety
    ///
    /// `raw` must be nil or an object the caller owns one reference to.
    pub(crate) unsafe fn adopt(raw: Id) -> Option<Self> {
        (!raw.is_null()).then_some(Self(raw))
    }

    /// Retain a +0 (autoreleased) reference. `None` for nil.
    ///
    /// # Safety
    ///
    /// `raw` must be nil or a live object.
    pub(crate) unsafe fn retain(objc: &Objc, raw: Id) -> Option<Self> {
        if raw.is_null() {
            return None;
        }
        // SAFETY: `objc_retain` on a live object.
        unsafe { (objc.retain)(raw) };
        Some(Self(raw))
    }

    pub(crate) fn as_raw(&self) -> Id {
        self.0
    }
}

impl Clone for ObjcId {
    fn clone(&self) -> Self {
        // An `ObjcId` exists only after `objc()` succeeded.
        let objc = objc().expect("Objective-C runtime loaded");
        // SAFETY: retaining a live object we own a reference to.
        unsafe { (objc.retain)(self.0) };
        Self(self.0)
    }
}

impl Drop for ObjcId {
    fn drop(&mut self) {
        if let Ok(objc) = objc() {
            // SAFETY: releasing the reference this handle owns.
            unsafe { (objc.release)(self.0) };
        }
    }
}

impl std::fmt::Debug for ObjcId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ObjcId({:p})", self.0)
    }
}

// SAFETY: every object wrapped here (device, queue, buffer, library, function,
// pipeline state, string, completed command buffer) is documented as safe to
// use from any thread; encoders are created and finished within one call.
unsafe impl Send for ObjcId {}
unsafe impl Sync for ObjcId {}

/// An autorelease pool for the current thread; dispatch runs on worker
/// threads that have none.
pub(crate) struct AutoreleasePool {
    objc: &'static Objc,
    token: *mut c_void,
}

impl AutoreleasePool {
    pub(crate) fn push(objc: &'static Objc) -> Self {
        // SAFETY: plain runtime call.
        Self { objc, token: unsafe { (objc.pool_push)() } }
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        // SAFETY: pops the pool this value pushed, on the same thread (`!Send`).
        unsafe { (self.objc.pool_pop)(self.token) };
    }
}

/// `[NSString stringWithUTF8String:]`, retained.
pub(crate) fn ns_string(objc: &Objc, text: &str) -> Result<ObjcId> {
    let text = CString::new(text)
        .map_err(|_| Error::Runtime { message: "string with interior NUL cannot become an NSString".into() })?;
    // SAFETY: class method with a NUL-terminated argument; returns +0.
    unsafe {
        ObjcId::retain(
            objc,
            objc.send1::<*const c_char, Id>(objc.classes.ns_string, objc.sels.string_with_utf8, text.as_ptr()),
        )
    }
    .ok_or_else(|| Error::Runtime { message: "stringWithUTF8String: returned nil".into() })
}

/// `-[NSString UTF8String]`; empty for nil.
///
/// # Safety
///
/// `string` must be nil or a live `NSString`.
pub(crate) unsafe fn ns_string_to_string(objc: &Objc, string: Id) -> String {
    if string.is_null() {
        return String::new();
    }
    // SAFETY: guaranteed by the caller.
    let chars = unsafe { objc.send0::<*const c_char>(string, objc.sels.utf8_string) };
    if chars.is_null() {
        return String::new();
    }
    // SAFETY: `UTF8String` returns a NUL-terminated buffer owned by the string.
    unsafe { CStr::from_ptr(chars) }.to_string_lossy().into_owned()
}

/// `-[NSError localizedDescription]`; `None` for nil.
///
/// # Safety
///
/// `error` must be nil or a live `NSError`.
pub(crate) unsafe fn ns_error_message(objc: &Objc, error: Id) -> Option<String> {
    if error.is_null() {
        return None;
    }
    // SAFETY: guaranteed by the caller.
    let description = unsafe { objc.send0::<Id>(error, objc.sels.localized_description) };
    Some(unsafe { ns_string_to_string(objc, description) })
}

/// Block descriptor per clang's Block-ABI-Apple (no copy/dispose helpers).
#[repr(C)]
pub(crate) struct BlockDescriptor {
    pub reserved: u64,
    pub size: u64,
}

/// A stack block literal carrying one captured pointer-sized context value.
#[repr(C)]
pub(crate) struct BlockLiteral<C> {
    pub isa: *const c_void,
    pub flags: i32,
    pub reserved: i32,
    pub invoke: *const c_void,
    pub descriptor: *const BlockDescriptor,
    pub context: C,
}

impl<C> BlockLiteral<C> {
    /// `descriptor` must outlive the block (keep it on the same stack frame).
    pub(crate) fn new(objc: &Objc, invoke: *const c_void, descriptor: &BlockDescriptor, context: C) -> Self {
        Self { isa: objc.stack_block_isa, flags: 0, reserved: 0, invoke, descriptor, context }
    }

    pub(crate) fn descriptor() -> BlockDescriptor {
        BlockDescriptor { reserved: 0, size: std::mem::size_of::<Self>() as u64 }
    }
}
