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
    build_args: Vec<Ident>,
    build_body: TokenStream,
}

struct Input {
    name: Ident,
    ty: Type,
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
            } else {
                body.parse::<Token![:]>()?;
                let input_ty: Type = body.parse()?;
                inputs.push(Input { name: first, ty: input_ty });
                if body.peek(Comma) {
                    body.parse::<Comma>()?;
                }
            }
        }

        let build_body = build_body.ok_or_else(|| Error::new(name.span(), "missing `build(...) { ... }` block"))?;

        Ok(JitWrapper { name, model_ty, inputs, vars, build_args, build_body })
    }
}

pub(crate) fn generate(jit: JitWrapper) -> Result<TokenStream> {
    use std::collections::HashSet;

    let name = &jit.name;
    let model_ty = &jit.model_ty;
    let state_name = format_ident!("{}State", name);

    let input_names: Vec<&Ident> = jit.inputs.iter().map(|i| &i.name).collect();
    let input_types: Vec<&Type> = jit.inputs.iter().map(|i| &i.ty).collect();
    let var_names: Vec<&Ident> = jit.vars.iter().map(|v| &v.name).collect();
    let var_min_exprs: Vec<&Expr> = jit.vars.iter().map(|v| &v.min).collect();
    let var_max_exprs: Vec<&Expr> = jit.vars.iter().map(|v| &v.max).collect();
    let var_field_names: Vec<Ident> = jit.vars.iter().map(|v| format_ident!("__var_{}", v.name)).collect();
    let input_id_fields: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_idx", i.name)).collect();
    let input_accessor_names: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_mut", i.name)).collect();
    let input_buffer_id_fields: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_buffer_id", i.name)).collect();
    let input_ast_id_locals: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_ast_id", i.name)).collect();

    let build_args = &jit.build_args;
    let build_body = &jit.build_body;

    let input_name_set: HashSet<String> = jit.inputs.iter().map(|i| i.name.to_string()).collect();
    let var_name_set: HashSet<String> = jit.vars.iter().map(|v| v.name.to_string()).collect();

    for var in &jit.vars {
        if input_name_set.contains(&var.name.to_string()) {
            return Err(Error::new(var.name.span(), "variable name conflicts with input name"));
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
        input_names.iter().zip(input_types.iter()).map(|(n, t)| quote! { #n: &#t }).collect();

    let var_inits =
        var_names.iter().zip(var_field_names.iter()).zip(var_min_exprs.iter().zip(var_max_exprs.iter())).map(
            |((var_name, field_name), (min_expr, max_expr))| {
                quote! {
                    let #field_name = morok_tensor::Variable::new(
                        stringify!(#var_name),
                        (#min_expr) as i64,
                        (#max_expr) as i64,
                    );
                }
            },
        );

    let prepare_var_bindings = var_names.iter().zip(var_field_names.iter()).map(|(var_name, field_name)| {
        quote! {
            let #var_name = self.#field_name
                .bind(self.#field_name.bounds().1)
                .map_err(|e| morok_model::jit::JitError::Tensor { source: e })?;
        }
    });

    let build_closure = quote! {
        (|| {
            let model = &self.model;
            let (#(#build_args),*) = (#(#build_arg_sources),*);
            #build_body
        })()
    };

    let buffer_id_extractions =
        input_names.iter().zip(input_buffer_id_fields.iter()).zip(input_ast_id_locals.iter()).map(
            |((input_name, buf_field), ast_field)| {
                quote! {
                    let #buf_field = #input_name.buffer().ok_or(morok_model::jit::JitError::NotPrepared)?.id();
                    let #ast_field = #input_name.uop().id;
                }
            },
        );

    let index_resolution =
        input_id_fields.iter().zip(input_buffer_id_fields.iter()).zip(input_ast_id_locals.iter()).map(
            |((idx_field, buf_id_field), ast_id_field)| {
                quote! {
                    let #idx_field = plan
                        .ast_to_buffer_map()
                        .get(&#ast_id_field)
                        .copied()
                        .or_else(|| plan.buffers().iter().position(|b| b.id() == #buf_id_field));
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

    let accessor_impls = input_accessor_names
        .iter()
        .zip(input_id_fields.iter())
        .zip(input_buffer_id_fields.iter())
        .zip(input_names.iter())
        .map(|(((accessor, idx_field), buf_id_field), input_name)| {
            let name_str = input_name.to_string();
            quote! {
                pub fn #accessor(&mut self) -> morok_model::jit::Result<&mut morok_device::Buffer> {
                    let state = self.state.as_mut().ok_or(morok_model::jit::JitError::NotPrepared)?;
                    let idx = match state.#idx_field {
                        Some(idx) => idx,
                        None => {
                            let idx = state
                                .plan
                                .buffers()
                                .iter()
                                .position(|b| b.id() == state.#buf_id_field)
                                .ok_or(morok_model::jit::JitError::InputBufferNotFound { name: #name_str })?;
                            state.#idx_field = Some(idx);
                            idx
                        }
                    };
                    state.plan.buffer_at_mut(idx)
                        .ok_or(morok_model::jit::JitError::InputBufferNotFound { name: #name_str })
                }
            }
        });

    let expanded = quote! {
        pub struct #name {
            model: #model_ty,
            state: Option<#state_name>,
            #( #var_field_names: morok_tensor::Variable, )*
        }

        struct #state_name {
            plan: morok_runtime::ExecutionPlan,
            #( #input_id_fields: Option<usize>, )*
            #( #input_buffer_id_fields: morok_device::BufferId, )*
        }

        impl #name {
            pub fn new(model: #model_ty) -> Self {
                #(#var_inits)*
                Self {
                    model,
                    state: None,
                    #( #var_field_names, )*
                }
            }

            pub fn prepare(&mut self, #(#prepare_params),*) -> morok_model::jit::Result<()> {
                let config = morok_tensor::PrepareConfig::from_env();
                self.prepare_with_config(#(#input_names,)* &config)
            }

            pub fn prepare_with_config(
                &mut self,
                #(#prepare_params,)*
                config: &morok_tensor::PrepareConfig,
            ) -> morok_model::jit::Result<()> {
                #(#prepare_var_bindings)*
                let output: morok_tensor::Tensor = #build_closure
                    .map_err(|e| morok_model::jit::JitError::Build { source: Box::new(e) as _ })?;

                #(#buffer_id_extractions)*

                let mut output = output;
                let plan = morok_tensor::Tensor::prepare_batch_with(std::iter::once(&mut output), config)
                    .map_err(|e| morok_model::jit::JitError::Tensor { source: e })?;

                #(#index_resolution)*

                self.state = Some(#state_init);
                Ok(())
            }

            #(#accessor_impls)*

            pub fn output(&self) -> morok_model::jit::Result<&morok_device::Buffer> {
                let state = self.state.as_ref().ok_or(morok_model::jit::JitError::NotPrepared)?;
                Ok(state.plan.output_buffer())
            }

            pub fn buffers(&self) -> morok_model::jit::Result<&[morok_device::Buffer]> {
                let state = self.state.as_ref().ok_or(morok_model::jit::JitError::NotPrepared)?;
                Ok(state.plan.buffers())
            }

            pub fn output_buffers(&self) -> morok_model::jit::Result<Vec<&morok_device::Buffer>> {
                let state = self.state.as_ref().ok_or(morok_model::jit::JitError::NotPrepared)?;
                Ok(state.plan.output_buffers())
            }

            pub fn input_buffer_ids(&self) -> morok_model::jit::Result<Vec<morok_device::BufferId>> {
                let state = self.state.as_ref().ok_or(morok_model::jit::JitError::NotPrepared)?;
                Ok(vec![#( state.#input_buffer_id_fields ),*])
            }

            pub fn prepared_kernels(&self) -> morok_model::jit::Result<&[morok_runtime::PreparedKernel]> {
                let state = self.state.as_ref().ok_or(morok_model::jit::JitError::NotPrepared)?;
                Ok(state.plan.prepared_kernels())
            }

            pub fn execute(&mut self) -> morok_model::jit::Result<()> {
                let state = self.state.as_mut().ok_or(morok_model::jit::JitError::NotPrepared)?;
                state.plan.execute()
                    .map_err(|e| morok_model::jit::JitError::Runtime { source: e })
            }

            pub fn execute_profiled(&mut self) -> morok_model::jit::Result<Vec<morok_runtime::KernelProfile>> {
                let state = self.state.as_mut().ok_or(morok_model::jit::JitError::NotPrepared)?;
                state.plan.execute_profiled()
                    .map_err(|e| morok_model::jit::JitError::Runtime { source: e })
            }

            pub fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> morok_model::jit::Result<()> {
                let state = self.state.as_mut().ok_or(morok_model::jit::JitError::NotPrepared)?;
                state.plan.execute_with_vars(vars)
                    .map_err(|e| morok_model::jit::JitError::Runtime { source: e })
            }

            pub fn execute_with_vars_profiled(
                &mut self,
                vars: &[(&str, i64)],
            ) -> morok_model::jit::Result<Vec<morok_runtime::KernelProfile>> {
                let state = self.state.as_mut().ok_or(morok_model::jit::JitError::NotPrepared)?;
                state.plan.execute_with_vars_profiled(vars)
                    .map_err(|e| morok_model::jit::JitError::Runtime { source: e })
            }
        }
    };

    Ok(expanded)
}
