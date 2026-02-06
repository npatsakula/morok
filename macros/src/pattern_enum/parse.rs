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
    /// Skip matcher generation for this variant (but still generate OpKey).
    pub skip_matcher: bool,
}

/// Parsed field information.
#[derive(Debug)]
pub struct FieldInfo {
    /// Field name (for struct variants) or index (for tuple variants).
    pub name: FieldName,
    /// Field type.
    pub ty: Type,
    /// Classification based on type.
    pub classification: FieldClass,
}

/// Field name - either named (struct) or indexed (tuple).
#[derive(Debug, Clone)]
pub enum FieldName {
    Named(Ident),
    /// Tuple field (index not tracked, only used to distinguish from Named).
    Indexed,
}

impl FieldName {
    /// Returns the identifier if this is a named field.
    pub fn as_named(&self) -> Option<&Ident> {
        match self {
            FieldName::Named(ident) => Some(ident),
            FieldName::Indexed => None,
        }
    }
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
    pub is_struct: bool,
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
                    } else if meta.path.is_ident("skip_matcher") {
                        result.skip_matcher = true;
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
    match fields {
        Fields::Named(named) => named
            .named
            .iter()
            .map(|f| FieldInfo {
                name: FieldName::Named(f.ident.clone().unwrap()),
                ty: f.ty.clone(),
                classification: classify_field_type(&f.ty),
            })
            .collect(),
        Fields::Unnamed(unnamed) => unnamed
            .unnamed
            .iter()
            .map(|f| FieldInfo {
                name: FieldName::Indexed,
                ty: f.ty.clone(),
                classification: classify_field_type(&f.ty),
            })
            .collect(),
        Fields::Unit => vec![],
    }
}

/// Parse all variants from the enum.
pub fn parse_variants(data: &DataEnum) -> Result<Vec<VariantInfo>> {
    data.variants
        .iter()
        .map(|v| {
            let attrs = parse_variant_attrs(&v.attrs)?;
            let fields = parse_fields(&v.fields);
            let is_struct = matches!(v.fields, Fields::Named(_));
            Ok(VariantInfo { name: v.ident.clone(), attrs, fields, is_struct })
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
