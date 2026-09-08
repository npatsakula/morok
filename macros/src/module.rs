//! `#[derive(Module)]`: state-dict serialisation derived from field types.
//!
//! Fields are classified by type syntax: a last path segment of `Tensor` is a
//! parameter, `Option<Tensor>` an optional parameter (opt in with
//! `#[module(optional)]`), primitives and containers of primitives are skipped,
//! and everything else is delegated to the field's own `Module` impl. Keys are
//! always built through `nn::prefixed`, so an empty prefix never grows a
//! leading dot.

use proc_macro2::{Ident, Span, TokenStream, TokenTree};
use quote::{ToTokens, quote};
use syn::{
    Attribute, Data, DeriveInput, Error, Expr, Fields, GenericArgument, LitStr, Path, PathArguments, Result, Token,
    Type, parse_quote, spanned::Spanned,
};

#[cfg(test)]
mod test;

/// Types whose fields carry no weights, plus the containers built from them.
const PRIMITIVES: &[&str] = &[
    "bool", "char", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128",
    "usize", "String", "str",
];

/// How a field takes part in the state dict.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Tensor,
    OptTensor,
    Skip,
    Child,
}

enum Optional {
    Always,
    If(Expr),
}

#[derive(Default)]
struct Attrs {
    key: Option<String>,
    skip: bool,
    optional: Option<Optional>,
    optional_span: Option<Span>,
}

pub fn generate(input: &DeriveInput) -> Result<TokenStream> {
    let krate = crate_path(&input.attrs)?;
    let name = &input.ident;
    let (writes, loads, fields) = match &input.data {
        Data::Struct(s) => {
            let (w, l, f) = struct_body(&krate, &s.fields, true)?;
            (quote!(#(#w)*), quote!(#(#l)*), f)
        }
        Data::Enum(e) => {
            let mut arms = (Vec::new(), Vec::new());
            let mut fields = Vec::new();
            for variant in &e.variants {
                let (w, l, f) = variant_arms(&krate, variant)?;
                arms.0.push(w);
                arms.1.push(l);
                fields.extend(f);
            }
            let (w, l) = arms;
            (quote!(match self { #(#w)* }), quote!(match self { #(#l)* }), fields)
        }
        Data::Union(u) => return Err(Error::new(u.union_token.span(), "`Module` cannot be derived for a union")),
    };

    let (idents, keys): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
    let (_, ty_generics, where_clause) = input.generics.split_for_impl();
    let mut bounded = input.generics.clone();
    add_bounds(&mut bounded, input, &krate);
    let (impl_generics, _, bounded_where) = bounded.split_for_impl();

    Ok(quote! {
        impl #impl_generics #krate::nn::Module for #name #ty_generics #bounded_where {
            fn write_state(&self, __prefix: &str, __out: &mut #krate::nn::StateDict) { #writes }

            fn load_state_dict(
                &mut self,
                __sd: &#krate::nn::StateDict,
                __prefix: &str,
            ) -> #krate::error::Result<()> {
                #loads
                ::core::result::Result::Ok(())
            }
        }

        impl #impl_generics #name #ty_generics #where_clause {
            /// Each weight-carrying field ident paired with its state-dict key segment.
            pub const MODULE_FIELDS: &'static [(&'static str, &'static str)] = &[#((#idents, #keys)),*];
        }
    })
}

/// Per-field code for a struct or a variant's field list. `owned` selects the
/// accessor shape: `self.x` for a struct, `(*x)` for a match binding.
fn struct_body(krate: &Path, fields: &Fields, owned: bool) -> Result<(Vec<TokenStream>, Vec<TokenStream>, Vec<Field>)> {
    let transparent = matches!(fields, Fields::Unnamed(f) if f.unnamed.len() == 1);
    let mut out = (Vec::new(), Vec::new(), Vec::new());
    for (i, field) in fields.iter().enumerate() {
        let attrs = parse_attrs(&field.attrs)?;
        let class = classify(&field.ty);
        check(class, &attrs, &field.ty)?;
        let (acc, default_key) = match &field.ident {
            Some(id) => (if owned { quote!(self.#id) } else { quote!((*#id)) }, id.to_string()),
            None => {
                let idx = syn::Index::from(i);
                let bind = binding(i);
                let key = if transparent { String::new() } else { i.to_string() };
                (if owned { quote!(self.#idx) } else { quote!((*#bind)) }, key)
            }
        };
        let key = attrs.key.clone().unwrap_or(default_key);
        let (w, l) = field_code(krate, &acc, &key, class, &attrs);
        if let (Some(id), false) = (&field.ident, attrs.skip || class == Class::Skip) {
            out.2.push((id.to_string(), key));
        }
        out.0.push(w);
        out.1.push(l);
    }
    Ok(out)
}

type Field = (String, String);

fn field_code(krate: &Path, acc: &TokenStream, key: &str, class: Class, attrs: &Attrs) -> (TokenStream, TokenStream) {
    if attrs.skip {
        return (quote!(), quote!());
    }
    let owned =
        if key.is_empty() { quote!(__prefix.to_string()) } else { quote!(#krate::nn::prefixed(__prefix, #key)) };
    let borrowed = if key.is_empty() { quote!(__prefix) } else { quote!(&#krate::nn::prefixed(__prefix, #key)) };
    match class {
        Class::Skip => (quote!(), quote!()),
        Class::Tensor => {
            (quote!(__out.insert(#owned, #acc.clone());), quote!(#acc = #krate::nn::get_tensor(__sd, #borrowed)?;))
        }
        Class::OptTensor => {
            let load = match &attrs.optional {
                Some(Optional::If(pred)) => quote! {{
                    let __want: bool = #pred;
                    #acc = if __want {
                        ::core::option::Option::Some(#krate::nn::get_tensor(__sd, #borrowed)?)
                    } else {
                        ::core::option::Option::None
                    };
                }},
                _ => quote!(#acc = __sd.get(#borrowed).cloned();),
            };
            (quote!(if let ::core::option::Option::Some(__t) = &#acc { __out.insert(#owned, __t.clone()); }), load)
        }
        Class::Child => (
            quote!(#krate::nn::Module::write_state(&#acc, #borrowed, __out);),
            quote!(#krate::nn::Module::load_state_dict(&mut #acc, __sd, #borrowed)?;),
        ),
    }
}

fn variant_arms(krate: &Path, variant: &syn::Variant) -> Result<(TokenStream, TokenStream, Vec<Field>)> {
    let attrs = parse_attrs(&variant.attrs)?;
    if let Some(span) = attrs.optional_span {
        return Err(Error::new(span, "`optional` belongs on a field, not on a variant"));
    }
    let name = &variant.ident;
    if matches!(variant.fields, Fields::Unit) {
        if attrs.key.is_some() {
            return Err(Error::new(variant.span(), "`key` has no effect on a unit variant: it carries no weights"));
        }
        return Ok((quote!(Self::#name => {}), quote!(Self::#name => {}), Vec::new()));
    }
    if attrs.skip {
        return Ok((quote!(Self::#name { .. } => {}), quote!(Self::#name { .. } => {}), Vec::new()));
    }

    let (writes, loads, fields) = struct_body(krate, &variant.fields, false)?;
    // A variant key nests the whole variant one segment deeper; the default is
    // pass-through, which makes a newtype variant transparent.
    let nest = match attrs.key.as_deref() {
        Some(k) if !k.is_empty() => quote!(let __prefix = &#krate::nn::prefixed(__prefix, #k);),
        _ => quote!(),
    };
    let pat = match &variant.fields {
        Fields::Named(f) => {
            let used = f.named.iter().zip(&writes).filter(|(_, w)| !w.is_empty()).filter_map(|(f, _)| f.ident.as_ref());
            quote!(Self::#name { #(#used,)* .. })
        }
        _ => {
            let binds = writes.iter().enumerate().map(|(i, w)| if w.is_empty() { quote!(_) } else { binding(i) });
            quote!(Self::#name(#(#binds),*))
        }
    };
    Ok((quote!(#pat => { #nest #(#writes)* }), quote!(#pat => { #nest #(#loads)* }), fields))
}

fn binding(i: usize) -> TokenStream {
    let id = Ident::new(&format!("__f{i}"), Span::call_site());
    quote!(#id)
}

fn check(class: Class, attrs: &Attrs, ty: &Type) -> Result<()> {
    match (class, &attrs.optional) {
        (Class::OptTensor, None) if !attrs.skip => Err(Error::new(
            ty.span(),
            "`Option<Tensor>` has no blanket `Module` impl: add `#[module(optional)]`, \
             `#[module(optional = \"<predicate over self>\")]` or `#[module(skip)]`",
        )),
        (c, Some(_)) if c != Class::OptTensor => Err(Error::new(
            attrs.optional_span.unwrap_or_else(Span::call_site),
            "`optional` applies only to an `Option<Tensor>` field",
        )),
        _ => Ok(()),
    }
}

fn classify(ty: &Type) -> Class {
    match ty {
        Type::Reference(r) => classify(&r.elem),
        Type::Paren(p) => classify(&p.elem),
        Type::Group(g) => classify(&g.elem),
        Type::Array(a) => nested(classify(&a.elem)),
        Type::Slice(s) => nested(classify(&s.elem)),
        Type::Tuple(t) => {
            if t.elems.iter().all(|e| classify(e) == Class::Skip) {
                Class::Skip
            } else {
                Class::Child
            }
        }
        Type::Path(p) => {
            let Some(seg) = p.path.segments.last() else { return Class::Child };
            let name = seg.ident.to_string();
            if name == "Tensor" {
                return Class::Tensor;
            }
            if PRIMITIVES.contains(&name.as_str()) {
                return Class::Skip;
            }
            let inner = || match &seg.arguments {
                PathArguments::AngleBracketed(a) => a.args.iter().find_map(|a| match a {
                    GenericArgument::Type(t) => Some(classify(t)),
                    _ => None,
                }),
                _ => None,
            };
            match name.as_str() {
                "Option" => match inner() {
                    Some(Class::Tensor) => Class::OptTensor,
                    Some(Class::Skip) => Class::Skip,
                    _ => Class::Child,
                },
                "Vec" | "VecDeque" | "SmallVec" | "HashSet" | "BTreeSet" => nested(inner().unwrap_or(Class::Child)),
                _ => Class::Child,
            }
        }
        _ => Class::Child,
    }
}

/// A container is inert exactly when its element type is.
fn nested(element: Class) -> Class {
    if element == Class::Skip { Class::Skip } else { Class::Child }
}

fn parse_attrs(attrs: &[Attribute]) -> Result<Attrs> {
    let mut out = Attrs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("module")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("key") {
                out.key = Some(meta.value()?.parse::<LitStr>()?.value());
            } else if meta.path.is_ident("skip") {
                out.skip = true;
            } else if meta.path.is_ident("optional") {
                out.optional_span = Some(meta.path.span());
                out.optional = Some(if meta.input.peek(Token![=]) {
                    Optional::If(meta.value()?.parse::<LitStr>()?.parse()?)
                } else {
                    Optional::Always
                });
            } else if meta.path.is_ident("crate") {
                meta.value()?.parse::<LitStr>()?;
            } else {
                return Err(meta.error("unknown `module` attribute: expected `key`, `skip`, `optional` or `crate`"));
            }
            Ok(())
        })?;
    }
    Ok(out)
}

fn crate_path(attrs: &[Attribute]) -> Result<Path> {
    let mut path: Path = parse_quote!(::svod_tensor);
    for attr in attrs.iter().filter(|a| a.path().is_ident("module")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("crate") {
                path = meta.value()?.parse::<LitStr>()?.parse()?;
                Ok(())
            } else {
                Err(meta.error("unknown `module` attribute on a type: expected `crate`"))
            }
        })?;
    }
    Ok(path)
}

/// Bound every type parameter that a delegated field mentions.
fn add_bounds(generics: &mut syn::Generics, input: &DeriveInput, krate: &Path) {
    let params: Vec<Ident> = generics.type_params().map(|p| p.ident.clone()).collect();
    if params.is_empty() {
        return;
    }
    let fields: Vec<&syn::Field> = match &input.data {
        Data::Struct(s) => s.fields.iter().collect(),
        Data::Enum(e) => e.variants.iter().flat_map(|v| v.fields.iter()).collect(),
        Data::Union(_) => Vec::new(),
    };
    let delegated: Vec<&Type> = fields
        .into_iter()
        .filter(|f| classify(&f.ty) == Class::Child && !parse_attrs(&f.attrs).is_ok_and(|a| a.skip))
        .map(|f| &f.ty)
        .collect();
    let used: Vec<Ident> =
        params.into_iter().filter(|p| delegated.iter().any(|ty| mentions(ty.to_token_stream(), p))).collect();
    let where_clause = generics.make_where_clause();
    for p in used {
        where_clause.predicates.push(parse_quote!(#p: #krate::nn::Module));
    }
}

fn mentions(tokens: TokenStream, param: &Ident) -> bool {
    tokens.into_iter().any(|t| match t {
        TokenTree::Ident(i) => i == *param,
        TokenTree::Group(g) => mentions(g.stream(), param),
        _ => false,
    })
}
