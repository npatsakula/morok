//! Code generation for the `Op` pattern infrastructure: `OpKey`, its dense index, and
//! the `alu` module that lets pattern code destructure grouped operations by kind
//! without any per-arity tables.

use super::analyze::{AnalyzedVariant, VariantKind, analyze_variants};
use super::parse::parse_input;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{DeriveInput, Ident, Result};

/// Generate all code from the derive input.
pub fn generate(input: &DeriveInput) -> Result<TokenStream> {
    let (enum_attrs, variants) = parse_input(input)?;
    let analyzed = analyze_variants(&enum_attrs, variants);
    let enum_name = &input.ident;

    let op_key = generate_op_key(&analyzed, enum_name);
    let alu = generate_alu(&analyzed, enum_name);

    Ok(quote! {
        /// Generated pattern matching infrastructure for Op enum.
        pub mod pattern_derived {
            use super::*;

            #op_key
        }

        #alu
    })
}

/// Generate the OpKey enum, `from_op` and the dense `index`.
fn generate_op_key(variants: &[AnalyzedVariant], enum_name: &Ident) -> TokenStream {
    let key_variants: Vec<_> = variants
        .iter()
        .map(|v| {
            let name = &v.name;
            match &v.filter_enum_type {
                Some(filter_type) => quote! { #name(#filter_type) },
                None => quote! { #name },
            }
        })
        .collect();

    let from_op_arms: Vec<_> = variants
        .iter()
        .map(|v| {
            let name = &v.name;
            match v.kind {
                VariantKind::Grouped => quote! { #enum_name::#name(op, ..) => OpKey::#name(*op) },
                _ if v.is_unit => quote! { #enum_name::#name => OpKey::#name },
                _ => quote! { #enum_name::#name(..) => OpKey::#name },
            }
        })
        .collect();

    // Dense index per OpKey, used as the bit position in `OpMask`. Ungrouped variants take
    // one slot each; a grouped variant reserves one slot per sub-enum variant so that
    // `Binary(Add)` and `Binary(Mul)` reject independently.
    let mut index_arms: Vec<TokenStream> = Vec::new();
    let mut next = 0usize;
    for v in variants.iter().filter(|v| v.kind != VariantKind::Grouped) {
        let name = &v.name;
        let index = next;
        next += 1;
        index_arms.push(quote! { OpKey::#name => #index });
    }
    let mut base_defs: Vec<TokenStream> = Vec::new();
    let mut count = quote! { #next };
    for v in variants.iter().filter(|v| v.kind == VariantKind::Grouped) {
        let name = &v.name;
        let ty = v.filter_enum_type.as_ref().expect("grouped variant has a filter enum");
        let base = format_ident!("OP_KEY_BASE_{}", name.to_string().to_uppercase());
        base_defs.push(quote! { const #base: usize = #count; });
        index_arms.push(quote! { OpKey::#name(sub) => #base + *sub as usize });
        count = quote! { #base + <#ty as ::strum::VariantArray>::VARIANTS.len() };
    }

    quote! {
        /// Operation key for pattern indexing.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum OpKey {
            #(#key_variants),*
        }

        #(#base_defs)*

        /// Number of distinct `OpKey` values — the bit width of `OpMask`.
        pub const OP_KEY_COUNT: usize = #count;

        impl OpKey {
            /// Extract the operation key from an Op.
            pub fn from_op(op: &#enum_name) -> Self {
                match op {
                    #(#from_op_arms),*
                }
            }

            /// Dense bit position of this key within an `OpMask`.
            #[inline]
            pub const fn index(&self) -> usize {
                match self {
                    #(#index_arms),*
                }
            }
        }
    }
}

/// Generate the `alu` module: every grouped kind's variants as values (`alu::Add`), one
/// type alias per grouped variant (`alu::Binary`), and the `AluOp` trait that keys and
/// destructures an op by kind. Pattern code names an ALU op through this module, so a
/// typo is a resolution error at the author's span and the arity is a tuple type.
fn generate_alu(variants: &[AnalyzedVariant], enum_name: &Ident) -> TokenStream {
    let grouped: Vec<_> = variants.iter().filter(|v| v.kind == VariantKind::Grouped).collect();
    let reexports = grouped.iter().map(|v| {
        let (name, ty) = (&v.name, v.filter_enum_type.as_ref().expect("grouped variant has a filter enum"));
        quote! {
            pub use #ty::*;
            pub type #name = #ty;
        }
    });
    let impls = grouped.iter().map(|v| {
        let (name, ty) = (&v.name, v.filter_enum_type.as_ref().expect("grouped variant has a filter enum"));
        let child_tys = v.children.iter().map(|f| &f.ty);
        let vars: Vec<Ident> = (0..v.children.len()).map(|i| format_ident!("child{i}")).collect();
        quote! {
            impl AluOp for #ty {
                type Children<'a> = (#(&'a #child_tys,)*);
                const ALL: &'static [Self] = <Self as ::strum::VariantArray>::VARIANTS;

                #[inline]
                fn key(self) -> pattern_derived::OpKey {
                    pattern_derived::OpKey::#name(self)
                }

                #[inline]
                fn destructure(self, op: &#enum_name) -> Option<Self::Children<'_>> {
                    match op {
                        #enum_name::#name(kind, #(#vars),*) if *kind == self => Some((#(#vars,)*)),
                        _ => None,
                    }
                }
            }
        }
    });

    quote! {
        /// Grouped (ALU) operation kinds as first-class values, plus [`alu::AluOp`] to
        /// dispatch and destructure an op by kind.
        pub mod alu {
            use super::*;

            #(#reexports)*

            /// An operation kind that selects a grouped `Op` variant.
            pub trait AluOp: Copy + PartialEq + Send + Sync + 'static {
                /// The children of a matching op, as a tuple of references.
                type Children<'a>;
                /// Every kind, in declaration order.
                const ALL: &'static [Self];
                /// The dispatch key of ops of this kind.
                fn key(self) -> pattern_derived::OpKey;
                /// The children of `op` when it is of this kind.
                fn destructure(self, op: &#enum_name) -> Option<Self::Children<'_>>;
            }

            #(#impls)*
        }
    }
}
