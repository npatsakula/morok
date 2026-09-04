//! Parsing of Op enum and pattern attributes.

use syn::{
    Attribute, Data, DataEnum, DeriveInput, Error, Expr, ExprArray, ExprPath, Fields, Ident, Meta, Result, Type,
};

/// Parsed enum-level attributes.
#[derive(Debug, Default)]
pub struct EnumAttrs {
    /// Variants marked as "grouped" (first field is sub-enum filter).
    pub grouped: Vec<Ident>,
}

/// Parsed variant-level attributes.
#[derive(Debug, Default)]
pub struct VariantAttrs {
    /// Skip pattern generation for this variant.
    pub skip: bool,
}

/// Parsed field information.
#[derive(Debug)]
pub struct FieldInfo {
    /// Field type.
    pub ty: Type,
    /// Classification based on type.
    pub classification: FieldClass,
}

/// Classification of a field based on its type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldClass {
    /// `Arc<UOp>` - fixed child.
    Child,
    /// `SmallVec<[Arc<UOp>; N]>` or `Vec<Arc<UOp>>` - variadic children.
    VariadicChildren,
    /// `Option<Arc<UOp>>` - optional child.
    OptionalChild,
    /// Other types - filter/metadata.
    Filter,
}

/// Parsed variant information.
#[derive(Debug)]
pub struct VariantInfo {
    pub name: Ident,
    pub attrs: VariantAttrs,
    pub fields: Vec<FieldInfo>,
}

/// Parse enum-level #[pattern(...)] attributes.
pub fn parse_enum_attrs(attrs: &[Attribute]) -> Result<EnumAttrs> {
    let mut result = EnumAttrs::default();

    for attr in attrs {
        if !attr.path().is_ident("pattern") {
            continue;
        }

        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("grouped") {
                let value: Expr = meta.value()?.parse()?;
                if let Expr::Array(ExprArray { elems, .. }) = value {
                    for elem in elems {
                        if let Expr::Path(ExprPath { path, .. }) = elem
                            && let Some(ident) = path.get_ident()
                        {
                            result.grouped.push(ident.clone());
                        }
                    }
                }
                Ok(())
            } else {
                Err(meta.error("unknown pattern attribute"))
            }
        })?;
    }

    Ok(result)
}

/// Parse variant-level #[pattern(...)] attributes.
pub fn parse_variant_attrs(attrs: &[Attribute]) -> Result<VariantAttrs> {
    let mut result = VariantAttrs::default();

    for attr in attrs {
        if !attr.path().is_ident("pattern") {
            continue;
        }

        match &attr.meta {
            Meta::List(list) => {
                list.parse_nested_meta(|meta| {
                    if meta.path.is_ident("skip") {
                        result.skip = true;
                        Ok(())
                    } else {
                        Err(meta.error("unknown pattern attribute"))
                    }
                })?;
            }
            Meta::Path(_) => {
                // #[pattern] without arguments - ignore
            }
            Meta::NameValue(nv) => {
                return Err(Error::new_spanned(nv, "expected #[pattern(...)]"));
            }
        }
    }

    Ok(result)
}

/// Classify a field type.
pub fn classify_field_type(ty: &Type) -> FieldClass {
    let type_str = quote::quote!(#ty).to_string();
    let normalized = type_str.replace(' ', "");

    // Check for Arc<UOp>
    if normalized.contains("Arc<UOp>") {
        // Check if wrapped in Option
        if normalized.starts_with("Option<") {
            return FieldClass::OptionalChild;
        }
        // Check if wrapped in SmallVec or Vec
        if normalized.starts_with("SmallVec<") || normalized.starts_with("Vec<") {
            return FieldClass::VariadicChildren;
        }
        return FieldClass::Child;
    }

    FieldClass::Filter
}

/// Parse fields from a variant.
pub fn parse_fields(fields: &Fields) -> Vec<FieldInfo> {
    fields.iter().map(|f| FieldInfo { ty: f.ty.clone(), classification: classify_field_type(&f.ty) }).collect()
}

/// Parse all variants from the enum.
pub fn parse_variants(data: &DataEnum) -> Result<Vec<VariantInfo>> {
    data.variants
        .iter()
        .map(|v| {
            let attrs = parse_variant_attrs(&v.attrs)?;
            Ok(VariantInfo { name: v.ident.clone(), attrs, fields: parse_fields(&v.fields) })
        })
        .collect()
}

/// Parse the entire derive input.
pub fn parse_input(input: &DeriveInput) -> Result<(EnumAttrs, Vec<VariantInfo>)> {
    let Data::Enum(data) = &input.data else {
        return Err(Error::new_spanned(input, "PatternEnum can only be derived for enums"));
    };

    let enum_attrs = parse_enum_attrs(&input.attrs)?;
    let variants = parse_variants(data)?;

    Ok((enum_attrs, variants))
}
