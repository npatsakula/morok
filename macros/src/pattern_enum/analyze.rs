//! Analysis of parsed variants to determine pattern characteristics.

use super::parse::{EnumAttrs, FieldClass, FieldInfo, VariantInfo};
use syn::Ident;

/// Classification of a variant for pattern generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantKind {
    /// Grouped operation (Binary, Unary, Ternary) - first field is sub-enum filter.
    Grouped,
    /// Regular variant with children and filters.
    Regular,
    /// Nullary variant (no children).
    Nullary,
    /// Skipped variant (no pattern generation).
    Skipped,
}

/// Analyzed variant with computed pattern characteristics.
#[derive(Debug)]
pub struct AnalyzedVariant {
    pub name: Ident,
    pub kind: VariantKind,
    /// Fixed child fields (Arc<UOp>).
    pub children: Vec<FieldInfo>,
    /// Variadic child field if present (SmallVec<...> or Vec<...>).
    pub variadic: Option<FieldInfo>,
    /// Filter/metadata fields.
    pub filters: Vec<FieldInfo>,
    /// Whether this is a struct variant (vs tuple).
    pub is_struct: bool,
    /// For grouped variants, the name of the sub-enum filter type.
    pub filter_enum_type: Option<syn::Type>,
}

impl AnalyzedVariant {
    /// Number of fixed children (used for arity).
    pub fn fixed_arity(&self) -> usize {
        self.children.len()
    }

    /// Whether this variant has variadic children.
    pub fn has_variadic(&self) -> bool {
        self.variadic.is_some()
    }

    /// Snake_case name for use in function names.
    pub fn snake_name(&self) -> String {
        use convert_case::{Case, Casing};
        self.name.to_string().to_case(Case::Snake)
    }
}

/// Analyze all variants given the enum attributes.
pub fn analyze_variants(enum_attrs: &EnumAttrs, variants: Vec<VariantInfo>) -> Vec<AnalyzedVariant> {
    variants.into_iter().map(|v| analyze_variant(enum_attrs, v)).collect()
}

fn analyze_variant(enum_attrs: &EnumAttrs, variant: VariantInfo) -> AnalyzedVariant {
    // Check if skipped
    if variant.attrs.skip {
        return AnalyzedVariant {
            name: variant.name,
            kind: VariantKind::Skipped,
            children: vec![],
            variadic: None,
            filters: vec![],
            is_struct: variant.is_struct,
            filter_enum_type: None,
        };
    }

    // Check if grouped
    let is_grouped = enum_attrs.grouped.contains(&variant.name);

    if is_grouped {
        return analyze_grouped_variant(variant);
    }

    analyze_regular_variant(variant)
}

fn analyze_grouped_variant(variant: VariantInfo) -> AnalyzedVariant {
    // For grouped variants (Binary, Unary, Ternary):
    // - First field is the sub-enum filter (e.g., BinaryOp)
    // - Remaining fields are children
    let mut fields = variant.fields.into_iter();
    let filter_field = fields.next();
    let children: Vec<FieldInfo> = fields.collect();

    AnalyzedVariant {
        name: variant.name,
        kind: VariantKind::Grouped,
        filter_enum_type: filter_field.as_ref().map(|f| f.ty.clone()),
        children,
        variadic: None,
        filters: filter_field.into_iter().collect(),
        is_struct: variant.is_struct,
    }
}

fn analyze_regular_variant(variant: VariantInfo) -> AnalyzedVariant {
    let mut children = vec![];
    let mut variadic = None;
    let mut has_optional = false;
    let mut filters = vec![];

    for field in variant.fields {
        match field.classification {
            FieldClass::Child => children.push(field),
            FieldClass::VariadicChildren => variadic = Some(field),
            FieldClass::OptionalChild => has_optional = true,
            FieldClass::Filter => filters.push(field),
        }
    }

    let kind = if children.is_empty() && variadic.is_none() && !has_optional {
        VariantKind::Nullary
    } else {
        VariantKind::Regular
    };

    AnalyzedVariant {
        name: variant.name,
        kind,
        children,
        variadic,
        filters,
        is_struct: variant.is_struct,
        filter_enum_type: None,
    }
}

/// Group variants by kind for generation.
pub fn group_by_kind(variants: &[AnalyzedVariant]) -> VariantGroups<'_> {
    VariantGroups { grouped: variants.iter().filter(|v| v.kind == VariantKind::Grouped).collect() }
}

/// Variants grouped by kind for code generation.
pub struct VariantGroups<'a> {
    /// Grouped operations (Binary, Unary, Ternary) where first field is sub-enum filter.
    pub grouped: Vec<&'a AnalyzedVariant>,
}
