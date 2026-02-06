//! Code generation for PatternEnum derive macro.

use super::analyze::{AnalyzedVariant, VariantGroups, VariantKind, analyze_variants, group_by_kind};
use super::parse::parse_input;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Ident, Result};

/// Generate all code from the derive input.
pub fn generate(input: &DeriveInput) -> Result<TokenStream> {
    let (enum_attrs, variants) = parse_input(input)?;
    let analyzed = analyze_variants(&enum_attrs, variants);
    let groups = group_by_kind(&analyzed);

    let enum_name = &input.ident;

    // Generate OpKey enum and from_op method
    let op_key = generate_op_key(&analyzed, &groups, enum_name);
    let metadata = generate_metadata(&analyzed, &groups);

    // Wrap in module
    Ok(quote! {
        /// Generated pattern matching infrastructure for Op enum.
        pub mod pattern_derived {
            use super::*;

            #op_key
            #metadata
        }
    })
}

/// Generate the OpKey enum and from_op method.
fn generate_op_key(variants: &[AnalyzedVariant], _groups: &VariantGroups, enum_name: &Ident) -> TokenStream {
    // Generate OpKey variants (including skipped ones so from_op can return a valid key)
    let key_variants: Vec<_> = variants
        .iter()
        .map(|v| {
            let name = &v.name;
            if v.kind == VariantKind::Grouped {
                let filter_type = v.filter_enum_type.as_ref().unwrap();
                quote! { #name(#filter_type) }
            } else {
                quote! { #name }
            }
        })
        .collect();

    // Generate from_op match arms (ALL variants, including skipped)
    let from_op_arms: Vec<_> = variants
        .iter()
        .map(|v| {
            let name = &v.name;

            // For skipped variants, return their OpKey (no patterns are indexed under these keys)
            if v.kind == VariantKind::Skipped {
                if v.is_struct {
                    return quote! { #enum_name::#name { .. } => OpKey::#name };
                } else {
                    return quote! { #enum_name::#name => OpKey::#name };
                }
            }

            match v.kind {
                VariantKind::Grouped => {
                    if v.is_struct {
                        quote! { #enum_name::#name { .. } => unreachable!() }
                    } else {
                        quote! { #enum_name::#name(op, ..) => OpKey::#name(*op) }
                    }
                }
                _ => {
                    if v.is_struct {
                        quote! { #enum_name::#name { .. } => OpKey::#name }
                    } else if v.children.is_empty() && v.filters.is_empty() && v.variadic.is_none() {
                        // Unit variant with no fields
                        quote! { #enum_name::#name => OpKey::#name }
                    } else {
                        // Tuple variant with fields
                        quote! { #enum_name::#name(..) => OpKey::#name }
                    }
                }
            }
        })
        .collect();

    quote! {
        /// Operation key for pattern indexing.
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub enum OpKey {
            #(#key_variants),*
        }

        impl OpKey {
            /// Extract the operation key from an Op.
            pub fn from_op(op: &#enum_name) -> Self {
                match op {
                    #(#from_op_arms),*
                }
            }
        }
    }
}

/// Generate metadata constants.
fn generate_metadata(variants: &[AnalyzedVariant], groups: &VariantGroups) -> TokenStream {
    let grouped_names: Vec<_> = groups.grouped.iter().map(|v| v.name.to_string()).collect();

    let single_source: Vec<_> = variants
        .iter()
        .filter(|v| v.kind == VariantKind::Regular && v.fixed_arity() == 1)
        .map(|v| {
            let name = v.name.to_string();
            let snake = v.snake_name();
            quote! { (#name, #snake) }
        })
        .collect();

    let variadic: Vec<_> = variants
        .iter()
        .filter(|v| v.has_variadic())
        .map(|v| {
            let name = v.name.to_string();
            let fixed = v.fixed_arity();
            quote! { (#name, #fixed) }
        })
        .collect();

    let all_ops: Vec<_> =
        variants.iter().filter(|v| v.kind != VariantKind::Skipped).map(|v| v.name.to_string()).collect();

    // Generate child field metadata for struct variants
    let child_fields: Vec<_> = variants
        .iter()
        .filter(|v| v.is_struct && !v.children.is_empty())
        .map(|v| {
            let name = v.name.to_string();
            let fields: Vec<_> = v.children.iter().filter_map(|f| f.name.as_named()).map(|n| n.to_string()).collect();
            quote! { (#name, &[#(#fields),*]) }
        })
        .collect();

    quote! {
        /// Metadata for pattern DSL.
        pub mod pattern_metadata {
            /// Grouped operation names (Binary, Unary, Ternary).
            pub const GROUPED_OPS: &[&str] = &[#(#grouped_names),*];

            /// Single-source ops: (name, snake_name).
            pub const SINGLE_SOURCE_OPS: &[(&str, &str)] = &[#(#single_source),*];

            /// Variadic ops: (name, fixed_arity).
            pub const VARIADIC_OPS: &[(&str, usize)] = &[#(#variadic),*];

            /// All operation names.
            pub const ALL_OPS: &[&str] = &[#(#all_ops),*];

            /// Child field names for struct variants: (op_name, &[field_names]).
            pub const CHILD_FIELDS: &[(&str, &[&str])] = &[#(#child_fields),*];

            // Re-export variant names from sub-enums (requires strum::VariantNames)
            /// Binary operation variant names.
            pub const BINARY_OPS: &[&str] = <super::super::BinaryOp as strum::VariantNames>::VARIANTS;
            /// Unary operation variant names.
            pub const UNARY_OPS: &[&str] = <super::super::UnaryOp as strum::VariantNames>::VARIANTS;
            /// Ternary operation variant names.
            pub const TERNARY_OPS: &[&str] = <super::super::TernaryOp as strum::VariantNames>::VARIANTS;
        }
    }
}
