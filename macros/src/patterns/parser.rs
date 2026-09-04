//! Pattern DSL parser.
//!
//! The left-hand side of a rule follows Rust pattern syntax, extended with what a Rust
//! pattern cannot say across an `Arc<UOp>` edge: nested op patterns, `Op[a, b]`
//! commutative matching, `@zero`/`@one`, and `x @const(v)` value extraction. Fields
//! of a struct pattern that are not written in that extended form are kept verbatim
//! as Rust patterns.

use syn::{
    Expr, Ident, Pat, Result, Token, braced, bracketed,
    ext::IdentExt,
    parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token,
};

/// A list of pattern items (rules or for-blocks), optionally preceded by `@context Type;`.
#[derive(Debug)]
pub struct PatternList {
    pub context_type: Option<syn::Type>,
    pub items: Vec<PatternItem>,
}

#[derive(Debug)]
pub enum PatternItem {
    Rule(Box<PatternRule>),
    ForBlock(ForBlock),
}

/// `for op in binary [Add, Mul] { ... }` — one copy of every rule per listed op, or per
/// op of the kind with `[*]`.
#[derive(Debug)]
pub struct ForBlock {
    pub var: Ident,
    /// Grouped variant name (`Unary`, `Binary`, `Ternary`), from the lowercase keyword.
    pub kind: Ident,
    /// Listed ops, or `None` for `[*]`.
    pub ops: Option<Vec<Ident>>,
    pub body: Vec<PatternRule>,
}

/// `lhs if guard => rhs`. The `~>` arrow is accepted as an alias of `=>`.
#[derive(Debug)]
pub struct PatternRule {
    pub lhs: Pattern,
    pub guard: Option<Expr>,
    pub rhs: RewriteExpr,
}

/// How a rule names an ALU op: `Add(..)` or a for-block variable `op(..)`.
#[derive(Debug, Clone)]
pub enum OpRef {
    Named(Ident),
    Var(Ident),
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Var(Ident),
    /// `name @ pattern`
    Binding {
        name: Ident,
        pattern: Box<Pattern>,
    },
    /// A unit variant of `Op`: `Noop`.
    Unit(Ident),
    /// `Add(x, y)`, or `Add[x, y]` when commutative.
    Alu {
        op: OpRef,
        args: Vec<Pattern>,
        commutative: bool,
    },
    /// `Cast { src: x, dtype }`
    Struct {
        op: Ident,
        fields: Vec<FieldPattern>,
    },
    /// `Const(pat)` — a Rust pattern over the `ConstValue`.
    Const(Pat),
    /// `@zero`
    Zero,
    /// `@one`
    One,
    /// `uop @const(value)`
    ConstValue {
        uop: Ident,
        value: Ident,
    },
    /// `uop @vconst(values)`
    VConst {
        uop: Ident,
        values: Ident,
    },
    /// `uop @anyconst(values)` — CONST or VCONST, values as a `Vec`.
    AnyConst {
        uop: Ident,
        values: Ident,
    },
    /// `Some(pattern)` over an `Option<Arc<UOp>>` field.
    Some(Box<Pattern>),
}

#[derive(Debug, Clone)]
pub struct FieldPattern {
    pub name: Ident,
    pub pattern: FieldPat,
}

/// A struct field is either matched as a child `UOp` or kept as a verbatim Rust pattern.
#[derive(Debug, Clone)]
pub enum FieldPat {
    Child(Pattern),
    Verbatim(Pat),
}

/// The right-hand side of a rewrite rule.
#[derive(Debug)]
pub enum RewriteExpr {
    /// Bare binding: `x` — rewrites to a clone of it.
    Var(Ident),
    /// `|x, y| body` — the parameter list is documentation; bindings are already in scope.
    Closure(syn::ExprClosure),
    Expr(Expr),
}

impl Parse for PatternList {
    fn parse(input: ParseStream) -> Result<Self> {
        let context_type = if input.peek(Token![@]) && input.peek2(Ident::peek_any) && peek2_is(input, "context") {
            input.parse::<Token![@]>()?;
            Ident::parse_any(input)?;
            let ty: syn::Type = input.parse()?;
            input.parse::<Token![;]>()?;
            Some(ty)
        } else {
            None
        };

        let mut items = Vec::new();
        while !input.is_empty() {
            if input.peek(Token![for]) {
                items.push(PatternItem::ForBlock(input.parse()?));
            } else {
                items.push(PatternItem::Rule(Box::new(input.parse()?)));
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(PatternList { context_type, items })
    }
}

fn peek2_is(input: ParseStream, keyword: &str) -> bool {
    let fork = input.fork();
    fork.parse::<Token![@]>().is_ok() && Ident::parse_any(&fork).is_ok_and(|ident| ident == keyword)
}

impl Parse for ForBlock {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<Token![for]>()?;
        let var: Ident = input.parse()?;
        input.parse::<Token![in]>()?;

        let keyword: Ident = input.parse()?;
        let kind = match keyword.to_string().as_str() {
            "unary" => Ident::new("Unary", keyword.span()),
            "binary" => Ident::new("Binary", keyword.span()),
            "ternary" => Ident::new("Ternary", keyword.span()),
            _ => return Err(syn::Error::new_spanned(keyword, "expected `unary`, `binary` or `ternary`")),
        };

        let content;
        bracketed!(content in input);
        let ops = if content.peek(Token![*]) {
            content.parse::<Token![*]>()?;
            None
        } else {
            Some(Punctuated::<Ident, Token![,]>::parse_terminated(&content)?.into_iter().collect())
        };

        let content;
        braced!(content in input);
        let mut body = Vec::new();
        while !content.is_empty() {
            body.push(content.parse()?);
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }
        Ok(ForBlock { var, kind, ops, body })
    }
}

impl Parse for PatternRule {
    fn parse(input: ParseStream) -> Result<Self> {
        let lhs: Pattern = input.parse()?;
        let guard = if input.peek(Token![if]) {
            input.parse::<Token![if]>()?;
            Some(parse_guard_expr(input)?)
        } else {
            None
        };
        if input.peek(Token![~]) {
            input.parse::<Token![~]>()?;
            input.parse::<Token![>]>()?;
        } else {
            input.parse::<Token![=>]>()?;
        }
        let rhs: RewriteExpr = input.parse()?;
        Ok(PatternRule { lhs, guard, rhs })
    }
}

/// Parse a guard expression, which ends at the arrow.
fn parse_guard_expr(input: ParseStream) -> Result<Expr> {
    let mut tokens = proc_macro2::TokenStream::new();
    while !input.is_empty() && !input.peek(Token![~]) && !input.peek(Token![=>]) {
        let tt: proc_macro2::TokenTree = input.parse()?;
        tokens.extend(std::iter::once(tt));
    }
    if tokens.is_empty() {
        return Err(input.error("expected guard expression after `if`"));
    }
    syn::parse2(tokens)
}

impl Parse for Pattern {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Token![_]) {
            input.parse::<Token![_]>()?;
            return Ok(Pattern::Wildcard);
        }
        if input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            let ident: Ident = Ident::parse_any(input)?;
            return match ident.to_string().as_str() {
                "zero" => Ok(Pattern::Zero),
                "one" => Ok(Pattern::One),
                other => Err(syn::Error::new_spanned(ident, format!("unknown `@{other}`; expected `@zero` or `@one`"))),
            };
        }

        let ident: Ident = input.parse()?;
        if ident == "Some" && input.peek(token::Paren) {
            let content;
            parenthesized!(content in input);
            return Ok(Pattern::Some(Box::new(content.parse()?)));
        }
        if ident == "Const" && input.peek(token::Paren) {
            let content;
            parenthesized!(content in input);
            return Ok(Pattern::Const(Pat::parse_single(&content)?));
        }
        if input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            if let Some(extractor) = peek_value_extractor(input) {
                Ident::parse_any(input)?;
                let content;
                parenthesized!(content in input);
                let value: Ident = content.parse()?;
                return Ok(match extractor {
                    "const" => Pattern::ConstValue { uop: ident, value },
                    "vconst" => Pattern::VConst { uop: ident, values: value },
                    _ => Pattern::AnyConst { uop: ident, values: value },
                });
            }
            return Ok(Pattern::Binding { name: ident, pattern: Box::new(input.parse()?) });
        }
        if input.peek(token::Paren) || input.peek(token::Bracket) {
            let commutative = input.peek(token::Bracket);
            let content;
            if commutative {
                bracketed!(content in input);
            } else {
                parenthesized!(content in input);
            }
            let args = Punctuated::<Pattern, Token![,]>::parse_terminated(&content)?.into_iter().collect();
            let op = if is_lowercase(&ident) { OpRef::Var(ident) } else { OpRef::Named(ident) };
            return Ok(Pattern::Alu { op, args, commutative });
        }
        if input.peek(token::Brace) {
            let content;
            braced!(content in input);
            let mut fields = Vec::new();
            while !content.is_empty() {
                if content.peek(Token![..]) {
                    content.parse::<Token![..]>()?;
                    break;
                }
                let name: Ident = content.parse()?;
                let pattern = if content.peek(Token![:]) {
                    content.parse::<Token![:]>()?;
                    parse_field_pat(&content)?
                } else {
                    FieldPat::Child(Pattern::Var(name.clone()))
                };
                fields.push(FieldPattern { name, pattern });
                if content.peek(Token![,]) {
                    content.parse::<Token![,]>()?;
                }
            }
            return Ok(Pattern::Struct { op: ident, fields });
        }
        Ok(if is_lowercase(&ident) { Pattern::Var(ident) } else { Pattern::Unit(ident) })
    }
}

/// `const(` / `vconst(` / `anyconst(` right after an `@`.
fn peek_value_extractor(input: ParseStream) -> Option<&'static str> {
    let fork = input.fork();
    let ident = Ident::parse_any(&fork).ok()?;
    if !fork.peek(token::Paren) {
        return None;
    }
    ["const", "vconst", "anyconst"].into_iter().find(|name| ident == name)
}

fn is_lowercase(ident: &Ident) -> bool {
    ident.to_string().starts_with(|c: char| c.is_lowercase() || c == '_')
}

fn parse_field_pat(input: ParseStream) -> Result<FieldPat> {
    if starts_child_pattern(input) {
        Ok(FieldPat::Child(input.parse()?))
    } else {
        Ok(FieldPat::Verbatim(Pat::parse_multi(input)?))
    }
}

/// Whether the upcoming tokens are in the extended (child) form rather than a plain Rust
/// pattern: `_`, `@..`, a snake-case binding, or an ident applied to `(..)`, `[..]`,
/// `{..}` or `@`.
fn starts_child_pattern(input: ParseStream) -> bool {
    if input.peek(Token![_]) || input.peek(Token![@]) {
        return true;
    }
    let fork = input.fork();
    let Ok(ident) = Ident::parse_any(&fork) else { return false };
    if fork.peek(Token![@]) {
        fork.parse::<Token![@]>().ok();
        return starts_child_pattern(&fork);
    }
    if ident == "None" {
        return false;
    }
    fork.peek(token::Paren) || fork.peek(token::Bracket) || fork.peek(token::Brace) || is_lowercase(&ident)
}

impl Parse for RewriteExpr {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Token![|]) {
            // `|params| body`: the body extends to the next top-level comma after the
            // parameter list, so count pipes to know when the list has closed.
            let mut tokens = proc_macro2::TokenStream::new();
            let mut pipes = 0;
            while !input.is_empty() && !(pipes >= 2 && input.peek(Token![,])) {
                let tt: proc_macro2::TokenTree = input.parse()?;
                if matches!(&tt, proc_macro2::TokenTree::Punct(p) if p.as_char() == '|') {
                    pipes += 1;
                }
                tokens.extend(std::iter::once(tt));
            }
            return Ok(RewriteExpr::Closure(syn::parse2(tokens)?));
        }

        if input.peek(Ident) {
            let fork = input.fork();
            let _: Ident = fork.parse()?;
            if fork.peek(Token![,]) || fork.is_empty() {
                return Ok(RewriteExpr::Var(input.parse()?));
            }
        }

        let mut tokens = proc_macro2::TokenStream::new();
        while !input.is_empty() && !input.peek(Token![,]) {
            let tt: proc_macro2::TokenTree = input.parse()?;
            tokens.extend(std::iter::once(tt));
        }
        if tokens.is_empty() {
            return Err(input.error("expected expression"));
        }
        Ok(RewriteExpr::Expr(syn::parse2(tokens)?))
    }
}
