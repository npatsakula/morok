use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Error, Ident, Result, Token, Type, braced,
    parse::{Parse, ParseStream},
    token::Comma,
};

pub(crate) struct JitWrapper {
    name: Ident,
    model_ty: Type,
    inputs: Vec<Input>,
    build_args: Vec<Ident>,
    build_body: TokenStream,
}

struct Input {
    name: Ident,
    ty: Type,
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

        Ok(JitWrapper { name, model_ty, inputs, build_args, build_body })
    }
}

pub(crate) fn generate(jit: JitWrapper) -> Result<TokenStream> {
    let name = &jit.name;
    let model_ty = &jit.model_ty;
    let state_name = format_ident!("{}State", name);

    let input_names: Vec<&Ident> = jit.inputs.iter().map(|i| &i.name).collect();
    let input_types: Vec<&Type> = jit.inputs.iter().map(|i| &i.ty).collect();
    let input_id_fields: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_idx", i.name)).collect();
    let input_accessor_names: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_mut", i.name)).collect();
    let input_buffer_id_fields: Vec<Ident> = jit.inputs.iter().map(|i| format_ident!("{}_buffer_id", i.name)).collect();

    let build_args = &jit.build_args;
    let build_body = &jit.build_body;

    let prepare_params = input_names.iter().zip(input_types.iter()).map(|(n, t)| quote! { #n: &#t });

    let build_closure = quote! {
        (|| {
            let model = &self.model;
            let (#(#build_args),*) = (#(#input_names),*);
            #build_body
        })()
    };

    let buffer_id_extractions = input_names.iter().zip(input_buffer_id_fields.iter()).map(|(input_name, field)| {
        quote! { let #field = #input_name.buffer().ok_or(morok_model::jit::JitError::NotPrepared)?.id(); }
    });

    let index_resolution = input_id_fields.iter().zip(input_buffer_id_fields.iter()).zip(input_names.iter()).map(
        |((idx_field, buf_id_field), _input_name)| {
            quote! {
                let #idx_field = plan.buffers().iter().position(|b| b.id() == #buf_id_field)
                    .ok_or(morok_model::jit::JitError::InputBufferNotFound { name: stringify!(#idx_field) })?;
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
                pub fn #accessor(&mut self) -> morok_model::jit::Result<&mut morok_device::Buffer> {
                    let state = self.state.as_mut().ok_or(morok_model::jit::JitError::NotPrepared)?;
                    state.plan.buffer_at_mut(state.#idx_field)
                        .ok_or(morok_model::jit::JitError::InputBufferNotFound { name: #name_str })
                }
            }
        },
    );

    let expanded = quote! {
        pub struct #name {
            model: #model_ty,
            state: Option<#state_name>,
        }

        struct #state_name {
            plan: morok_runtime::ExecutionPlan,
            #( #input_id_fields: usize, )*
            #( #input_buffer_id_fields: morok_device::BufferId, )*
        }

        impl #name {
            pub fn new(model: #model_ty) -> Self {
                Self { model, state: None }
            }

            pub fn prepare(&mut self, #(#prepare_params),*) -> morok_model::jit::Result<()> {
                let output: morok_tensor::Tensor = #build_closure
                    .map_err(|e| morok_model::jit::JitError::Build { source: Box::new(e) as _ })?;

                #(#buffer_id_extractions)*

                let mut output = output;
                let plan = morok_tensor::Tensor::prepare_batch(std::iter::once(&mut output))
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

            pub fn execute(&mut self) -> morok_model::jit::Result<()> {
                let state = self.state.as_mut().ok_or(morok_model::jit::JitError::NotPrepared)?;
                state.plan.execute()
                    .map_err(|e| morok_model::jit::JitError::Runtime { source: e })
            }

            pub fn execute_with_vars(&mut self, vars: &[(&str, i64)]) -> morok_model::jit::Result<()> {
                let state = self.state.as_mut().ok_or(morok_model::jit::JitError::NotPrepared)?;
                state.plan.execute_with_vars(vars)
                    .map_err(|e| morok_model::jit::JitError::Runtime { source: e })
            }
        }
    };

    Ok(expanded)
}
