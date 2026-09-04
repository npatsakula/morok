use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Error, Expr, Ident, Result, Token, Type, braced,
    parse::{Parse, ParseStream},
    token::Comma,
};

pub(crate) struct JitWrapper {
    name: Ident,
    model_ty: Type,
    inputs: Vec<Input>,
    vars: Vec<VarDecl>,
    /// Declared output names. Empty = the classic single-output form (the
    /// `build` closure returns one `Tensor`). Non-empty = the `build` closure
    /// returns a tuple of that many `Tensor`s, in this order, each exposed as
    /// its own `output_buffer_at(i)`-backed accessor.
    outputs: Vec<Ident>,
    build_args: Vec<Ident>,
    build_body: TokenStream,
}

struct Input {
    name: Ident,
}

struct VarDecl {
    name: Ident,
    min: Expr,
    max: Expr,
}

impl Parse for JitWrapper {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        let content;
        syn::parenthesized!(content in input);
        let model_ty: Type = content.parse()?;

        let body;
        braced!(body in input);

        let mut inputs = Vec::new();
        let mut vars = Vec::new();
        let mut outputs = Vec::new();
        let mut build_args = Vec::new();
        let mut build_body = None;

        while !body.is_empty() {
            let first: Ident = body.parse()?;

            if first == "build" {
                let args;
                syn::parenthesized!(args in body);
                build_args = args.parse_terminated(Ident::parse, Comma)?.into_iter().collect();

                let block;
                braced!(block in body);
                build_body = Some(block.parse()?);
            } else if first == "vars" {
                let vars_block;
                braced!(vars_block in body);
                while !vars_block.is_empty() {
                    let name: Ident = vars_block.parse()?;
                    vars_block.parse::<Token![:]>()?;

                    let bounds;
                    syn::parenthesized!(bounds in vars_block);
                    let min: Expr = bounds.parse()?;
                    bounds.parse::<Comma>()?;
                    let max: Expr = bounds.parse()?;
                    if !bounds.is_empty() {
                        return Err(Error::new(bounds.span(), "expected bounds as (min, max)"));
                    }

                    vars.push(VarDecl { name, min, max });

                    if vars_block.peek(Comma) {
                        vars_block.parse::<Comma>()?;
                    }
                }
            } else if first == "outputs" {
                let outs_block;
                braced!(outs_block in body);
                while !outs_block.is_empty() {
                    let out_name: Ident = outs_block.parse()?;
                    outputs.push(out_name);
                    if outs_block.peek(Comma) {
                        outs_block.parse::<Comma>()?;
                    }
                }
                // Tolerate a trailing comma after the `outputs { .. }` block so it
                // reads like the input declarations it sits beside.
                if body.peek(Comma) {
                    body.parse::<Comma>()?;
                }
            } else {
                // Accept (and discard) an optional `: Tensor` for DSL clarity;
                // the macro now allocates placeholder buffers from `InputSpec`
                // so the declared type is informational.
                if body.peek(Token![:]) {
                    body.parse::<Token![:]>()?;
                    let _: Type = body.parse()?;
                }
                inputs.push(Input { name: first });
                if body.peek(Comma) {
                    body.parse::<Comma>()?;
                }
            }
        }

        let build_body = build_body.ok_or_else(|| Error::new(name.span(), "missing `build(...) { ... }` block"))?;

        Ok(JitWrapper { name, model_ty, inputs, vars, outputs, build_args, build_body })
    }
}

pub(crate) fn generate(jit: JitWrapper) -> Result<TokenStream> {
    use std::collections::HashSet;

    let name = &jit.name;
    let model_ty = &jit.model_ty;
    let state_name = format_ident!("{}State", name);

    let input_names: Vec<&Ident> = jit.inputs.iter().map(|i| &i.name).collect();
    let var_names: Vec<&Ident> = jit.vars.iter().map(|v| &v.name).collect();
    let var_min_exprs: Vec<&Expr> = jit.vars.iter().map(|v| &v.min).collect();
    let var_max_exprs: Vec<&Expr> = jit.vars.iter().map(|v| &v.max).collect();
    let var_field_names: Vec<Ident> = jit.vars.iter().map(|v| format_ident!("__var_{}", v.name)).collect();
    let input_id_fields: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_idx", i.name)).collect();
    let input_accessor_names: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_mut", i.name)).collect();
    let copy_helper_names: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("copy_output_to_{}", i.name)).collect();
    let input_buffer_id_fields: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_buffer_id", i.name)).collect();
    let input_ast_id_locals: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_ast_id", i.name)).collect();
    let input_realized_locals: Vec<Ident> =
        jit.inputs.iter().map(|i| format_ident!("__jit_input_{}", i.name)).collect();

    let build_args = &jit.build_args;
    let build_body = &jit.build_body;

    let input_name_set: HashSet<String> = jit.inputs.iter().map(|i| i.name.to_string()).collect();
    let var_name_set: HashSet<String> = jit.vars.iter().map(|v| v.name.to_string()).collect();

    for var in &jit.vars {
        if input_name_set.contains(&var.name.to_string()) {
            return Err(Error::new(var.name.span(), "variable name conflicts with input name"));
        }
    }

    let output_names: Vec<&Ident> = jit.outputs.iter().collect();
    let multi_output = !output_names.is_empty();
    let n_outputs = output_names.len();

    // Fixed method names the generated impl always contains; a declared
    // output with one of these names would emit a duplicate definition.
    const GENERATED_METHODS: &[&str] = &[
        "new",
        "prepare",
        "prepare_with_config",
        "replicate",
        "output",
        "buffers",
        "output_buffers",
        "input_buffer_ids",
        "prepared_kernels",
        "execute",
        "execute_profiled",
        "execute_profiled_static",
        "execute_with_vars",
        "execute_with_vars_profiled",
    ];
    for out in &output_names {
        let out_str = out.to_string();
        if input_name_set.contains(&out_str) || var_name_set.contains(&out_str) {
            return Err(Error::new(out.span(), "output name conflicts with an input or variable name"));
        }
        if GENERATED_METHODS.contains(&out_str.as_str()) {
            return Err(Error::new(out.span(), "output name collides with a generated method"));
        }
    }

    for arg in build_args {
        let arg_name = arg.to_string();
        if !input_name_set.contains(&arg_name) && !var_name_set.contains(&arg_name) {
            return Err(Error::new(arg.span(), "build arg must match an input or a declared variable"));
        }
    }

    let build_arg_sources: Vec<TokenStream> = build_args.iter().map(|arg| quote! { #arg }).collect();

    let prepare_params: Vec<TokenStream> =
        input_names.iter().map(|n| quote! { #n: svod_model::jit::InputSpec }).collect();

    let var_inits =
        var_names.iter().zip(var_field_names.iter()).zip(var_min_exprs.iter().zip(var_max_exprs.iter())).map(
            |((var_name, field_name), (min_expr, max_expr))| {
                quote! {
                    let #field_name = svod_tensor::Variable::new(
                        stringify!(#var_name),
                        (#min_expr) as i64,
                        (#max_expr) as i64,
                    );
                }
            },
        );

    // For each declared `vars { name: (min, max), ... }` entry, emit three
    // builders:
    //   * `with_<name>_bound(max)`     — override only the upper bound
    //   * `with_<name>_min_bound(min)` — override only the lower bound
    //   * `with_<name>_fixed(value)`   — pin both bounds to one value, making
    //     the variable a JIT-time constant (specializable kernels, single
    //     valid value at execute time)
    //
    // All three panic if the resulting `[min, max]` is empty so misuse fails
    // loud at construction instead of at bind/execute time. Variable names
    // are checked at compile time via the generated method names. Must be
    // chained before `prepare` — the JIT plan captures the bounds when the
    // build closure runs.
    let with_var_bound_methods = var_names.iter().zip(var_field_names.iter()).flat_map(|(var_name, field_name)| {
        let max_setter = format_ident!("with_{}_bound", var_name);
        let min_setter = format_ident!("with_{}_min_bound", var_name);
        let fixed_setter = format_ident!("with_{}_fixed", var_name);
        let max_doc = format!(
            "Override the upper bound for the `{var_name}` symbolic variable. \
             Must be called before `prepare`/`prepare_with_config`. Panics if \
             `max < min`."
        );
        let min_doc = format!(
            "Override the lower bound for the `{var_name}` symbolic variable. \
             Must be called before `prepare`/`prepare_with_config`. Panics if \
             `min > max`."
        );
        let fixed_doc = format!(
            "Pin `{var_name}` to a single value, making it a JIT-time \
             constant. Sets both bounds to `value` so only `value` is \
             accepted at execute time. Must be called before \
             `prepare`/`prepare_with_config`. Panics on `value == 0`."
        );
        let name_str = format!("{var_name}");
        std::iter::empty()
            .chain(std::iter::once(quote! {
                #[doc = #max_doc]
                pub fn #max_setter(mut self, max: usize) -> Self {
                    let (min, _) = self.#field_name.bounds();
                    let max_i64 = max as i64;
                    assert!(
                        max_i64 >= min,
                        "{}: with_{}_bound({max}) creates empty range (min={min})",
                        #name_str, #name_str,
                    );
                    self.#field_name = svod_tensor::Variable::new(stringify!(#var_name), min, max_i64);
                    self
                }
            }))
            .chain(std::iter::once(quote! {
                #[doc = #min_doc]
                pub fn #min_setter(mut self, min: usize) -> Self {
                    let (_, max) = self.#field_name.bounds();
                    let min_i64 = min as i64;
                    assert!(
                        min_i64 <= max,
                        "{}: with_{}_min_bound({min}) exceeds upper bound max={max}",
                        #name_str, #name_str,
                    );
                    self.#field_name = svod_tensor::Variable::new(stringify!(#var_name), min_i64, max);
                    self
                }
            }))
            .chain(std::iter::once(quote! {
                #[doc = #fixed_doc]
                pub fn #fixed_setter(mut self, value: usize) -> Self {
                    assert!(value > 0, "{}: with_{}_fixed(0) is not allowed", #name_str, #name_str);
                    let v = value as i64;
                    self.#field_name = svod_tensor::Variable::new(stringify!(#var_name), v, v);
                    self
                }
            }))
    });

    let prepare_var_bindings = var_names.iter().zip(var_field_names.iter()).map(|(var_name, field_name)| {
        quote! {
            let #var_name = self.#field_name
                .bind(self.#field_name.bounds().1)
                .map_err(|e| svod_model::jit::JitError::Tensor { source: Box::new(e) })?;
        }
    });

    // The build closure runs once, at capture, on the caller's thread, so it
    // inherits the caller's origin scope; the wrapper's name roots every kernel
    // the plan ends up owning.
    let name_str = name.to_string();
    let build_closure = quote! {
        (|| {
            let __jit_origin = svod_ir::origin::OriginScope::label(#name_str);
            let model: &#model_ty = &self.model;
            let (#(#build_args),*) = (#(#build_arg_sources),*);
            #build_body
        })()
    };

    // Placeholder inputs are allocated directly (`from_bytes_shaped_spec`
    // mints a unique-slot BUFFER UOp) rather than realized from
    // `Tensor::zeros`: a zeros graph is pure and hash-consed, so every
    // same-shape placeholder in the process would share ONE UOp identity —
    // and a concurrent prepare's `apply_map_to_tensors` could then rebind
    // this JIT's input to a foreign buffer. Per-tensor identity closes that
    // race at the source. Trade-off (accepted): allocation resolves through
    // the global device registry and panics on allocator failure, bypassing
    // `PrepareConfig`'s resolver — the pre-existing `device_local` behavior,
    // now uniform for all inputs.
    let input_realizations = input_names.iter().zip(input_realized_locals.iter()).map(|(input_name, local)| {
        quote! {
            let #local = {
                let numel: usize = #input_name.shape.iter().product();
                svod_tensor::Tensor::from_bytes_shaped_spec(
                    &vec![0u8; numel * #input_name.dtype.bytes()],
                    &#input_name.shape,
                    #input_name.dtype.clone(),
                    svod_dtype::default_device::default_device(),
                    // Host-mapped unless declared device-local: inputs are
                    // host-written every execute (`as_array_mut` pack).
                    svod_device::BufferSpec { cpu_access: !#input_name.device_local, ..Default::default() },
                )
            };
            let #input_name = &#local;
        }
    });

    let buffer_id_extractions =
        input_names.iter().zip(input_buffer_id_fields.iter()).zip(input_ast_id_locals.iter()).map(
            |((input_name, buf_field), ast_field)| {
                quote! {
                    let #buf_field = #input_name.buffer().ok_or(svod_model::jit::JitError::NotPrepared)?.id();
                    let #ast_field = #input_name.uop().id;
                }
            },
        );

    let duplicate_input_checks = input_names.iter().zip(input_buffer_id_fields.iter()).enumerate().flat_map(
        |(left_idx, (left_name, left_buf_field))| {
            input_names.iter().zip(input_buffer_id_fields.iter()).skip(left_idx + 1).map(
                move |(right_name, right_buf_field)| {
                    let left_name_str = left_name.to_string();
                    let right_name_str = right_name.to_string();
                    quote! {
                        if #left_buf_field == #right_buf_field {
                            return Err(svod_model::jit::JitError::DuplicateInputBuffer {
                                name: #right_name_str,
                                duplicate_of: #left_name_str,
                                buffer_id: #right_buf_field,
                            });
                        }
                    }
                },
            )
        },
    );

    // Eagerly resolve each input's plan buffer index at prepare time. A
    // missing input fails loud here (instead of at first accessor use), and
    // the stored plain `usize` survives replication, where buffer handle ids
    // are re-minted but indices are preserved.
    let index_resolution = input_id_fields
        .iter()
        .zip(input_buffer_id_fields.iter())
        .zip(input_ast_id_locals.iter())
        .zip(input_names.iter())
        .map(|(((idx_field, buf_id_field), ast_id_field), input_name)| {
            let name_str = input_name.to_string();
            quote! {
                let #idx_field = plan
                    .ast_to_buffer_map()
                    .get(&#ast_id_field)
                    .copied()
                    .or_else(|| plan.buffers().iter().position(|b| b.id() == #buf_id_field))
                    .ok_or(svod_model::jit::JitError::InputBufferNotFound { name: #name_str })?;
            }
        });

    // Declare each input on the plan right after resolving its index: the
    // plan's write analysis cannot see host writes (`copyin`, the
    // `copy_output_to_*` state recycling), and `replicate` snapshot-forks
    // declared inputs.
    let input_declarations = if input_id_fields.is_empty() {
        quote! {}
    } else {
        let declares = input_id_fields.iter().map(|idx_field| {
            quote! {
                plan.declare_input(#idx_field).map_err(|e| svod_model::jit::JitError::Runtime { source: e })?;
            }
        });
        quote! {
            let mut plan = plan;
            #(#declares)*
        }
    };

    // Post-resolution identity checks: indices must be pairwise distinct and
    // must resolve back to the realized input buffers. This guards the uop
    // channel (`ast_to_buffer`) — the one cross-plan aliasing corrupts —
    // where the pre-plan `DuplicateInputBuffer` check only sees the local
    // buffer handles.
    let mut index_conflict_checks: Vec<TokenStream> = Vec::new();
    for (left_pos, (left_idx, left_name)) in input_id_fields.iter().zip(input_names.iter()).enumerate() {
        for ((right_idx, right_buf), right_name) in
            input_id_fields.iter().zip(input_buffer_id_fields.iter()).zip(input_names.iter()).skip(left_pos + 1)
        {
            let left_str = left_name.to_string();
            let right_str = right_name.to_string();
            index_conflict_checks.push(quote! {
                if #left_idx == #right_idx {
                    return Err(svod_model::jit::JitError::DuplicateInputBuffer {
                        name: #right_str,
                        duplicate_of: #left_str,
                        buffer_id: #right_buf,
                    });
                }
            });
        }
    }
    let identity_checks = input_id_fields.iter().zip(input_buffer_id_fields.iter()).zip(input_names.iter()).map(
        |((idx_field, buf_id_field), input_name)| {
            let name_str = input_name.to_string();
            quote! {
                {
                    let resolved = plan.buffers()[#idx_field].id();
                    if resolved != #buf_id_field {
                        return Err(svod_model::jit::JitError::InputAliased {
                            name: #name_str,
                            expected: #buf_id_field,
                            actual: resolved,
                        });
                    }
                }
            }
        },
    );

    let idx_fields: Vec<&Ident> = input_id_fields.iter().collect();
    let buf_id_fields: Vec<&Ident> = input_buffer_id_fields.iter().collect();
    let state_init = quote! {
        #state_name {
            plan,
            #( #idx_fields, )*
            #( #buf_id_fields, )*
        }
    };

    let accessor_impls = input_accessor_names.iter().zip(input_id_fields.iter()).zip(input_names.iter()).map(
        |((accessor, idx_field), input_name)| {
            let name_str = input_name.to_string();
            quote! {
                fn #accessor(&mut self) -> svod_model::jit::Result<&mut svod_device::Buffer> {
                    self.plan
                        .buffer_at_mut(self.#idx_field)
                        .ok_or(svod_model::jit::JitError::InputBufferNotFound { name: #name_str })
                }
            }
        },
    );

    // Per-input on-device copy helpers: copy a region of declared output
    // `out_pos` into the input's buffer with NO host round-trip (the plan owns
    // both buffers; the split borrow lives in the runtime). Used to recycle
    // recurrent state output→input.
    let copy_helper_impls = copy_helper_names.iter().zip(input_id_fields.iter()).map(|(helper, idx_field)| {
        quote! {
            fn #helper(
                &mut self,
                out_pos: usize,
                dst_off: usize,
                src_off: usize,
                len: usize,
            ) -> svod_model::jit::Result<()> {
                self.plan
                    .copy_output_region_to_buffer(out_pos, self.#idx_field, dst_off, src_off, len)
                    .map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }
        }
    });

    // Build the output tensor(s) and compile the plan. The single-output form
    // (no `outputs` clause) keeps the original `output: Tensor` codegen verbatim;
    // the multi-output form destructures the build closure's tuple in declared
    // order and feeds all of them to `prepare_batch_with` (which preserves order),
    // then asserts the plan kept exactly that many outputs.
    let build_and_compile = if multi_output {
        quote! {
            let (#(#output_names,)*) = #build_closure
                .map_err(|e| svod_model::jit::JitError::Build { source: Box::new(e) as _ })?;
            let mut __jit_outputs: [svod_tensor::Tensor; #n_outputs] = [#(#output_names,)*];
            let plan = svod_tensor::Tensor::prepare_batch_with(__jit_outputs.iter_mut(), config)
                .map_err(|e| svod_model::jit::JitError::Tensor { source: Box::new(e) })?;
            if plan.num_outputs() != #n_outputs {
                return Err(svod_model::jit::JitError::OutputCountMismatch {
                    declared: #n_outputs,
                    actual: plan.num_outputs(),
                });
            }
        }
    } else {
        quote! {
            let output: svod_tensor::Tensor = #build_closure
                .map_err(|e| svod_model::jit::JitError::Build { source: Box::new(e) as _ })?;
            let mut output = output;
            let plan = svod_tensor::Tensor::prepare_batch_with(std::iter::once(&mut output), config)
                .map_err(|e| svod_model::jit::JitError::Tensor { source: Box::new(e) })?;
        }
    };

    // One accessor per declared output, backed by positional `output_buffer_at(i)`
    // (i = declared order = `prepare_batch_with` order). Empty for single-output.
    let output_named_accessors = output_names.iter().enumerate().map(|(i, out_name)| {
        quote! {
            fn #out_name(&self) -> svod_model::jit::Result<&svod_device::Buffer> {
                self.plan.output_buffer_at(#i).ok_or(svod_model::jit::JitError::NotPrepared)
            }
        }
    });

    // `replicate` on the state: the plan itself knows every declared input
    // (see the `declare_input` calls emitted into `prepare_with_config`), so
    // replication is a single argument-less call; only the replica's buffer
    // ids need re-resolving at the preserved indices.
    let replicate_copy_indices = idx_fields.iter().map(|idx| quote! { #idx: self.#idx, });
    let replicate_rebinds = idx_fields.iter().zip(buf_id_fields.iter()).map(|(idx, buf)| {
        quote! { #buf: plan.buffers()[self.#idx].id(), }
    });

    // The full post-prepare API, implemented once on the state struct and
    // forwarded verbatim by the wrapper through its `Option<state>`.
    let forward_api = |state_ref: TokenStream, state_mut: TokenStream| -> TokenStream {
        let input_accessors = input_accessor_names.iter().map(|method| {
            quote! {
                pub fn #method(&mut self) -> svod_model::jit::Result<&mut svod_device::Buffer> {
                    #state_mut.#method()
                }
            }
        });
        let copy_helpers = copy_helper_names.iter().map(|method| {
            quote! {
                pub fn #method(
                    &mut self,
                    out_pos: usize,
                    dst_off: usize,
                    src_off: usize,
                    len: usize,
                ) -> svod_model::jit::Result<()> {
                    #state_mut.#method(out_pos, dst_off, src_off, len)
                }
            }
        });
        let named_outputs = output_names.iter().map(|method| {
            quote! {
                pub fn #method(&self) -> svod_model::jit::Result<&svod_device::Buffer> {
                    #state_ref.#method()
                }
            }
        });
        quote! {
            #(#input_accessors)*
            #(#copy_helpers)*
            #(#named_outputs)*

            pub fn output(&self) -> svod_model::jit::Result<&svod_device::Buffer> {
                #state_ref.output()
            }

            pub fn buffers(&self) -> svod_model::jit::Result<&[svod_device::Buffer]> {
                #state_ref.buffers()
            }

            pub fn output_buffers(&self) -> svod_model::jit::Result<Vec<&svod_device::Buffer>> {
                #state_ref.output_buffers()
            }

            pub fn input_buffer_ids(&self) -> svod_model::jit::Result<Vec<svod_device::BufferId>> {
                #state_ref.input_buffer_ids()
            }

            pub fn prepared_kernels(&self) -> svod_model::jit::Result<Vec<&svod_runtime::PreparedKernel>> {
                #state_ref.prepared_kernels()
            }

            pub fn execute(&mut self) -> svod_model::jit::Result<()> {
                #state_mut.execute()
            }

            pub fn execute_profiled(&mut self) -> svod_model::jit::Result<Vec<svod_runtime::KernelProfile>> {
                #state_mut.execute_profiled()
            }

            pub fn execute_profiled_static(&mut self) -> svod_model::jit::Result<Vec<svod_runtime::KernelProfile>> {
                #state_mut.execute_profiled_static()
            }

            pub fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> svod_model::jit::Result<()> {
                #state_mut.execute_with_vars(vars)
            }

            pub fn execute_with_vars_profiled(
                &mut self,
                vars: &[(&str, i64)],
            ) -> svod_model::jit::Result<Vec<svod_runtime::KernelProfile>> {
                #state_mut.execute_with_vars_profiled(vars)
            }
        }
    };
    let wrapper_api = forward_api(
        quote! { self.state.as_ref().ok_or(svod_model::jit::JitError::NotPrepared)? },
        quote! { self.state.as_mut().ok_or(svod_model::jit::JitError::NotPrepared)? },
    );

    let expanded = quote! {
        pub struct #name {
            /// `Arc` so `replicate` can return `Self` without an `M: Clone`
            /// bound; the model only feeds `&`-access graph building. Note
            /// the wrapper is therefore `Send`/`Sync` iff `M: Send + Sync`.
            model: std::sync::Arc<#model_ty>,
            state: Option<#state_name>,
            #( #var_field_names: svod_tensor::Variable, )*
        }

        struct #state_name {
            plan: svod_runtime::ExecutionPlan,
            #( #input_id_fields: usize, )*
            #( #input_buffer_id_fields: svod_device::BufferId, )*
        }

        impl #state_name {
            #(#accessor_impls)*
            #(#copy_helper_impls)*
            #(#output_named_accessors)*

            fn output(&self) -> svod_model::jit::Result<&svod_device::Buffer> {
                self.plan.output_buffer().ok_or(svod_model::jit::JitError::NotPrepared)
            }

            fn buffers(&self) -> svod_model::jit::Result<&[svod_device::Buffer]> {
                Ok(self.plan.buffers())
            }

            fn output_buffers(&self) -> svod_model::jit::Result<Vec<&svod_device::Buffer>> {
                Ok(self.plan.output_buffers())
            }

            fn input_buffer_ids(&self) -> svod_model::jit::Result<Vec<svod_device::BufferId>> {
                Ok(vec![#( self.#buf_id_fields ),*])
            }

            fn prepared_kernels(&self) -> svod_model::jit::Result<Vec<&svod_runtime::PreparedKernel>> {
                Ok(self.plan.prepared_kernels())
            }

            fn execute(&mut self) -> svod_model::jit::Result<()> {
                self.plan.execute().map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }

            fn execute_profiled(&mut self) -> svod_model::jit::Result<Vec<svod_runtime::KernelProfile>> {
                self.plan.execute_profiled().map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }

            fn execute_profiled_static(&mut self) -> svod_model::jit::Result<Vec<svod_runtime::KernelProfile>> {
                self.plan.profile(&svod_runtime::ProfileOptions::default())
                    .map(|mut profile| profile.stages.pop().map_or_else(Vec::new, |stage| stage.kernels))
                    .map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }

            fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> svod_model::jit::Result<()> {
                self.plan.execute_with_vars(vars).map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }

            fn execute_with_vars_profiled(
                &mut self,
                vars: &[(&str, i64)],
            ) -> svod_model::jit::Result<Vec<svod_runtime::KernelProfile>> {
                self.plan
                    .execute_with_vars_profiled(vars)
                    .map_err(|e| svod_model::jit::JitError::Runtime { source: e })
            }

            fn replicate(&self) -> svod_model::jit::Result<Self> {
                let plan = self
                    .plan
                    .replicate()
                    .map_err(|e| svod_model::jit::JitError::Runtime { source: e })?;
                Ok(Self {
                    #( #replicate_copy_indices )*
                    #( #replicate_rebinds )*
                    plan,
                })
            }
        }

        impl #name {
            pub fn new(model: #model_ty) -> Self {
                #(#var_inits)*
                Self {
                    model: std::sync::Arc::new(model),
                    state: None,
                    #( #var_field_names, )*
                }
            }

            #(#with_var_bound_methods)*

            pub fn prepare(&mut self, #(#prepare_params),*) -> svod_model::jit::Result<()> {
                let config = svod_tensor::PrepareConfig::from_env();
                self.prepare_with_config(#(#input_names,)* &config)
            }

            pub fn prepare_with_config(
                &mut self,
                #(#prepare_params,)*
                config: &svod_tensor::PrepareConfig,
            ) -> svod_model::jit::Result<()> {
                #(#input_realizations)*
                #(#buffer_id_extractions)*
                #(#duplicate_input_checks)*

                #(#prepare_var_bindings)*

                #build_and_compile

                #(#index_resolution)*
                #(#index_conflict_checks)*
                #(#identity_checks)*

                #input_declarations

                self.state = Some(#state_init);
                Ok(())
            }

            /// Deep-copy the prepared JIT for concurrent execution: forks
            /// every written storage (bare) and the declared input buffers
            /// (with a snapshot of their current contents), shares the model,
            /// weights and compiled kernels, and gives the replica its own
            /// backend queue. Replicate only while the JIT is idle.
            pub fn replicate(&self) -> svod_model::jit::Result<Self> {
                let state = self.state.as_ref().ok_or(svod_model::jit::JitError::NotPrepared)?;
                Ok(Self {
                    model: std::sync::Arc::clone(&self.model),
                    state: Some(state.replicate()?),
                    #( #var_field_names: self.#var_field_names.clone(), )*
                })
            }

            #wrapper_api
        }
    };

    Ok(expanded)
}
