//! `#[op_enum]`: one struct per struct-like variant, so each operation can carry
//! its own `impl` block instead of living as an anonymous field list inside `Op`.
//!
//! The enum is rewritten so every named-field variant wraps its struct
//! (`Index { buffer, indices }` becomes `Index(ops::Index)`), the structs live in
//! a sibling `ops` module, and `Op` gains `From<ops::X>` plus `as_x()` accessors.
//! Tuple and unit variants are left untouched. The pattern-matching
//! infrastructure (`OpKey`, `pattern_metadata`) is derived from the original
//! field layout, so the DSL keeps seeing named children.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Attribute, DeriveInput, Fields, FieldsUnnamed, ItemEnum, Result, Variant, parse_quote};

pub fn expand(mut item: ItemEnum) -> Result<TokenStream> {
    let derived = super::generate(&DeriveInput::from(item.clone()))?;
    let enum_name = item.ident.clone();

    let mut structs = Vec::new();
    let mut accessors = Vec::new();
    for variant in &mut item.variants {
        let Fields::Named(fields) = &variant.fields else {
            variant.attrs.retain(|attr| !attr.path().is_ident("pattern"));
            continue;
        };
        let name = &variant.ident;
        let docs: Vec<&Attribute> = variant.attrs.iter().filter(|attr| attr.path().is_ident("doc")).collect();
        let members = fields.named.iter().map(|field| {
            let docs = field.attrs.iter().filter(|attr| attr.path().is_ident("doc"));
            let (ident, ty) = (&field.ident, &field.ty);
            quote! { #(#docs)* pub #ident: #ty }
        });
        let accessor = format_ident!("as_{}", snake(&name.to_string()));
        structs.push(quote! {
            #(#docs)*
            #[derive(Debug, Clone, Hash, PartialEq, Eq)]
            pub struct #name { #(#members),* }

            impl From<#name> for super::#enum_name {
                fn from(op: #name) -> Self {
                    Self::#name(op)
                }
            }
        });
        accessors.push(quote! {
            pub fn #accessor(&self) -> Option<&ops::#name> {
                match self {
                    Self::#name(op) => Some(op),
                    _ => None,
                }
            }
        });

        let unnamed: FieldsUnnamed = parse_quote!((ops::#name));
        *variant = Variant {
            attrs: variant.attrs.iter().filter(|attr| !attr.path().is_ident("pattern")).cloned().collect(),
            ident: variant.ident.clone(),
            fields: Fields::Unnamed(unnamed),
            discriminant: None,
        };
    }
    item.attrs.retain(|attr| !attr.path().is_ident("pattern"));

    Ok(quote! {
        #item

        /// Payload structs of the struct-like `Op` variants.
        pub mod ops {
            use super::*;
            #(#structs)*
        }

        impl #enum_name {
            #(#accessors)*
        }

        #derived
    })
}

fn snake(name: &str) -> String {
    use convert_case::{Case, Casing};
    name.to_case(Case::Snake)
}
