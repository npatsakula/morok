//! LLVM IR renderer implementation using inkwell.

use crate::{RenderedKernel, Renderer, Result, with_context};
use inkwell::context::Context;
use inkwell::module::Module;
use morok_ir::{Op, UOp};
use std::rc::Rc;

use super::helpers::ValueMap;

/// Render a UOp graph to LLVM IR using the thread-local context.
pub fn render(uop: &Rc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
    with_context(|context| {
        let renderer = LlvmRenderer::new(context);
        renderer.render(uop, name)
    })
}

/// Collect all buffer parameters from a UOp graph.
///
/// After rangeify, BUFFER operations have been converted to DEFINE_GLOBAL/DEFINE_LOCAL.
/// This function collects both patterns:
/// - Op::Buffer (for non-rangeified graphs)
/// - Op::DefineGlobal/DefineLocal (for rangeified kernels)
///
/// Returns buffers in a consistent order (sorted by UOp ID) for deterministic
/// function signatures.
fn collect_buffers(root: &Rc<UOp>) -> Vec<Rc<UOp>> {
    let mut buffers = Vec::new();
    let nodes = root.toposort();

    for node in &nodes {
        match node.op() {
            Op::Buffer { .. } | Op::DefineGlobal(_) | Op::DefineLocal(_) => {
                buffers.push(node.clone());
            }
            _ => {}
        }
    }

    // Sort by ID for deterministic ordering
    buffers.sort_by_key(|b| b.id);
    buffers
}

/// LLVM IR code generator for CPU execution using inkwell.
pub struct LlvmRenderer<'ctx> {
    context: &'ctx Context,
}

impl<'ctx> LlvmRenderer<'ctx> {
    /// Create a new LLVM renderer with a given context.
    ///
    /// The context must outlive the renderer, typically the context
    /// is owned by the calling code and passed by reference.
    pub fn new(context: &'ctx Context) -> Self {
        Self { context }
    }

    /// Get the inkwell context.
    pub fn context(&self) -> &'ctx Context {
        self.context
    }

    /// Render a UOp graph into LLVM IR.
    ///
    /// This creates a module with:
    /// 1. The actual kernel function taking individual buffer pointers
    /// 2. A bootstrap function that unpacks an array of pointers and calls the kernel
    fn render_to_module(&self, uop: &Rc<UOp>, name: &str) -> Result<Module<'ctx>> {
        let module = self.context.create_module(name);
        let builder = self.context.create_builder();

        // Collect all buffers from the graph
        let buffers = collect_buffers(uop);

        // Create kernel function signature: void kernel(ptr %buf0, ptr %buf1, ...)
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let param_types: Vec<_> = buffers.iter().map(|_| ptr_type.into()).collect();
        let fn_type = self.context.void_type().fn_type(&param_types, false);
        let kernel_name = format!("{}_impl", name);
        let kernel_function = module.add_function(&kernel_name, fn_type, None);

        // Create entry block for kernel
        let entry_block = self.context.append_basic_block(kernel_function, "entry");
        builder.position_at_end(entry_block);

        // Create ValueMap and populate with buffer parameters
        let mut values = ValueMap::new();
        for (i, buffer_uop) in buffers.iter().enumerate() {
            let param = kernel_function.get_nth_param(i as u32).unwrap();
            param.set_name(&format!("buf{}", i));
            values.insert(buffer_uop.id, param);
        }

        // Walk the UOp graph in topological order and generate code
        let nodes = uop.toposort();
        for node in &nodes {
            // Generate code for this node
            super::ops::codegen_uop(node, self.context, &module, &builder, &mut values)?;
        }

        // Return void
        builder.build_return(None).map_err(|e| crate::Error::LlvmError { reason: format!("build_return: {}", e) })?;

        // Create bootstrap function: void kernel_bootstrap(ptr %args)
        // This unpacks the array of pointers and calls the kernel
        let bootstrap_fn_type = self.context.void_type().fn_type(&[ptr_type.into()], false);
        let bootstrap_function = module.add_function(name, bootstrap_fn_type, None);

        let bootstrap_entry = self.context.append_basic_block(bootstrap_function, "entry");
        builder.position_at_end(bootstrap_entry);

        let args_array = bootstrap_function.get_first_param().unwrap().into_pointer_value();
        args_array.set_name("args");

        // Extract each buffer pointer from the array
        let mut buffer_ptrs = Vec::new();
        for i in 0..buffers.len() {
            // GEP to get pointer to args[i]
            let index = self.context.i64_type().const_int(i as u64, false);
            let ptr_to_ptr = unsafe {
                builder
                    .build_gep(ptr_type, args_array, &[index], &format!("arg{}_ptr", i))
                    .map_err(|e| crate::Error::LlvmError { reason: format!("build_gep for arg {}: {}", i, e) })?
            };

            // Load the actual buffer pointer
            let buffer_ptr = builder
                .build_load(ptr_type, ptr_to_ptr, &format!("arg{}", i))
                .map_err(|e| crate::Error::LlvmError { reason: format!("build_load for arg {}: {}", i, e) })?;

            buffer_ptrs.push(buffer_ptr.into());
        }

        // Call the kernel with unpacked arguments
        builder
            .build_call(kernel_function, &buffer_ptrs, "")
            .map_err(|e| crate::Error::LlvmError { reason: format!("build_call kernel: {}", e) })?;

        // Return void
        builder
            .build_return(None)
            .map_err(|e| crate::Error::LlvmError { reason: format!("build_return bootstrap: {}", e) })?;

        // Verify the module
        if let Err(err) = module.verify() {
            return Err(crate::Error::LlvmError { reason: format!("Module verification failed: {}", err) });
        }

        Ok(module)
    }
}

impl<'ctx> Renderer for LlvmRenderer<'ctx> {
    fn render(&self, uop: &Rc<UOp>, name: Option<&str>) -> Result<RenderedKernel> {
        let kernel_name = name.unwrap_or("kernel");

        // Generate LLVM IR module
        let module = self.render_to_module(uop, kernel_name)?;

        // Get LLVM IR as string
        let ir_string = module.print_to_string().to_string();

        Ok(RenderedKernel::new(ir_string, kernel_name.to_string(), kernel_name.to_string()))
    }

    fn backend_name(&self) -> &str {
        "llvm"
    }

    fn supports_op(&self, _op: &Op) -> bool {
        // TODO: Implement proper op support checking
        true
    }
}
