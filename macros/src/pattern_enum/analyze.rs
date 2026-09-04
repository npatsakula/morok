//! Analysis of parsed variants to determine pattern characteristics.

use super::parse::{EnumAttrs, FieldClass, FieldInfo, VariantInfo};
use syn::Ident;

/// Classification of a variant for pattern generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantKind {
    /// Grouped operation (Binary, Unary, Ternary) - first field is sub-enum filter.
    Grouped,
    /// Any other variant that takes part in pattern matching.
    Regular,
    /// Skipped variant (no pattern generation).
    Skipped,
}

/// Analyzed variant with computed pattern characteristics.
#[derive(Debug)]
pub struct AnalyzedVariant {
    pub name: Ident,
    pub kind: VariantKind,
    /// Fixed child fields (`Arc<UOp>`); for grouped variants, everything after the kind.
    pub children: Vec<FieldInfo>,
    /// Whether the variant carries no fields at all.
    pub is_unit: bool,
    /// For grouped variants, the sub-enum type carrying the operation kind.
    pub filter_enum_type: Option<syn::Type>,
}

/// Analyze all variants given the enum attributes.
pub fn analyze_variants(enum_attrs: &EnumAttrs, variants: Vec<VariantInfo>) -> Vec<AnalyzedVariant> {
    variants.into_iter().map(|v| analyze_variant(enum_attrs, v)).collect()
}

fn analyze_variant(enum_attrs: &EnumAttrs, variant: VariantInfo) -> AnalyzedVariant {
    let is_unit = variant.fields.is_empty();
    let (kind, children, filter_enum_type) = if variant.attrs.skip {
        (VariantKind::Skipped, vec![], None)
    } else if enum_attrs.grouped.contains(&variant.name) {
        let mut fields = variant.fields.into_iter();
        let filter = fields.next().map(|f| f.ty);
        (VariantKind::Grouped, fields.collect(), filter)
    } else {
        let children = variant.fields.into_iter().filter(|f| f.classification == FieldClass::Child).collect();
        (VariantKind::Regular, children, None)
    };
    AnalyzedVariant { name: variant.name, kind, children, is_unit, filter_enum_type }
}
