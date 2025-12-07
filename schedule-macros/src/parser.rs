//! Pattern DSL parser.
//!
//! Parses the declarative pattern syntax into an AST that can be used
//! for code generation.

use syn::{
    Expr, Ident, Result, Token, braced, bracketed,
    ext::IdentExt,
    parenthesized,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    token,
};

/// A list of pattern items (rules or for-blocks).
///
/// Optionally declares a context type with `@context Type;` at the start.
/// When a context type is declared:
/// - Generated `PatternMatcher<ContextType>` instead of `PatternMatcher<()>`
/// - Pattern closures receive `ctx: &mut ContextType`
/// - `ctx` is available in RHS expressions
#[derive(Debug)]
pub struct PatternList {
    /// Optional context type declaration (e.g., `@context KernelContext;`)
    pub context_type: Option<syn::Type>,
    /// Pattern items
    pub items: Vec<PatternItem>,
}

/// An item in a pattern list.
#[derive(Debug)]
pub enum PatternItem {
    /// A single pattern rule: `lhs => rhs`
    Rule(Box<PatternRule>),
    /// An iteration block: `for op in unary [Neg, Sqrt] { ... }`
    ForBlock(ForBlock),
}

/// An iteration block that generates multiple patterns.
#[derive(Debug)]
pub struct ForBlock {
    /// The iteration variable name (e.g., `op`)
    pub var: Ident,
    /// The iteration kind and values
    pub iter_kind: IterKind,
    /// The pattern rules inside the block
    pub body: Vec<PatternRule>,
}

/// The kind of iteration being performed.
#[derive(Debug, Clone)]
pub enum IterKind {
    /// Iterate over unary operations: `unary [Neg, Sqrt, ...]`
    Unary(Vec<Ident>),
    /// Iterate over binary operations: `binary [Add, Mul, ...]`
    Binary(Vec<Ident>),
    /// Iterate over ternary operations: `ternary [Where, MulAcc]`
    Ternary(Vec<Ident>),
}

/// The kind of arrow in a pattern rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowKind {
    /// `~>` - Infallible, RHS returns Arc<UOp>
    Infallible,
    /// `=>` - Fallible, RHS returns Option<Arc<UOp>>
    Fallible,
}

impl Parse for PatternList {
    fn parse(input: ParseStream) -> Result<Self> {
        // Check for optional @context declaration at the start
        let context_type = if input.peek(Token![@]) {
            let fork = input.fork();
            fork.parse::<Token![@]>()?;

            // Check if next is "context" keyword
            if fork.peek(Ident::peek_any) {
                let maybe_context: Ident = Ident::parse_any(&fork)?;
                if maybe_context == "context" {
                    // This is a context declaration - consume from real stream
                    input.parse::<Token![@]>()?;
                    let _: Ident = Ident::parse_any(input)?; // "context"

                    // Parse the type
                    let ty: syn::Type = input.parse()?;

                    // Expect semicolon
                    input.parse::<Token![;]>()?;

                    Some(ty)
                } else {
                    // Not a context declaration - probably @zero or similar
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let mut items = Vec::new();

        while !input.is_empty() {
            // Check if this is a for-block
            if input.peek(Token![for]) {
                items.push(PatternItem::ForBlock(input.parse()?));
            } else {
                items.push(PatternItem::Rule(Box::new(input.parse()?)));
            }

            // Allow trailing comma
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(PatternList { context_type, items })
    }
}

impl Parse for ForBlock {
    fn parse(input: ParseStream) -> Result<Self> {
        // Parse: for op in unary [Neg, Sqrt, ...]
        input.parse::<Token![for]>()?;
        let var: Ident = input.parse()?;
        input.parse::<Token![in]>()?;

        // Parse iteration kind
        let iter_kind: IterKind = input.parse()?;

        // Parse the body block
        let content;
        braced!(content in input);

        let mut body = Vec::new();
        while !content.is_empty() {
            body.push(content.parse()?);
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }

        Ok(ForBlock { var, iter_kind, body })
    }
}

impl Parse for IterKind {
    fn parse(input: ParseStream) -> Result<Self> {
        // Expect: unary [Neg, Sqrt, ...] or binary [Add, Mul, ...] or ternary [Where, MulAcc]
        let kind_ident: Ident = input.parse()?;

        // Parse the operation list in brackets
        let content;
        bracketed!(content in input);
        let ops: Punctuated<Ident, Token![,]> = Punctuated::parse_terminated(&content)?;
        let ops: Vec<Ident> = ops.into_iter().collect();

        match kind_ident.to_string().as_str() {
            "unary" => Ok(IterKind::Unary(ops)),
            "binary" => Ok(IterKind::Binary(ops)),
            "ternary" => Ok(IterKind::Ternary(ops)),
            _ => Err(syn::Error::new_spanned(kind_ident, "Expected 'unary', 'binary', or 'ternary'")),
        }
    }
}

/// A single pattern rule: `lhs if guard ~> rhs` or `lhs if guard => rhs`.
///
/// - `~>` (infallible): RHS returns `Arc<UOp>`, pattern always succeeds if matched
/// - `=>` (fallible): RHS returns `Option<Arc<UOp>>`, pattern may return None
#[derive(Debug)]
pub struct PatternRule {
    pub lhs: Pattern,
    pub guard: Option<Expr>,
    pub arrow: ArrowKind,
    pub rhs: RewriteExpr,
}

impl Parse for PatternRule {
    fn parse(input: ParseStream) -> Result<Self> {
        // 1. Parse LHS pattern
        let lhs: Pattern = input.parse()?;

        // 2. Parse optional guard BEFORE arrow (like Rust match arms)
        let guard = if input.peek(Token![if]) {
            input.parse::<Token![if]>()?;
            Some(parse_guard_expr(input)?)
        } else {
            None
        };

        // 3. Parse arrow kind (~> or =>)
        let arrow = if input.peek(Token![~]) {
            input.parse::<Token![~]>()?;
            input.parse::<Token![>]>()?;
            ArrowKind::Infallible
        } else {
            input.parse::<Token![=>]>()?;
            ArrowKind::Fallible
        };

        // 4. Parse RHS expression
        let rhs: RewriteExpr = input.parse()?;

        Ok(PatternRule { lhs, guard, arrow, rhs })
    }
}

/// Parse guard expression that ends at arrow (~> or =>).
fn parse_guard_expr(input: ParseStream) -> Result<syn::Expr> {
    let mut tokens = proc_macro2::TokenStream::new();

    while !input.is_empty() {
        // Stop at ~> or =>
        if input.peek(Token![~]) || input.peek(Token![=>]) {
            break;
        }
        let tt: proc_macro2::TokenTree = input.parse()?;
        tokens.extend(std::iter::once(tt));
    }

    if tokens.is_empty() {
        return Err(input.error("expected guard expression after `if`"));
    }

    syn::parse2(tokens)
}

/// A pattern in the LHS of a rule.
#[derive(Debug, Clone)]
pub enum Pattern {
    /// Wildcard: `_`
    Wildcard,
    /// Variable binding: `x`
    Var(Ident),
    /// Binding pattern: `name @ pattern`
    Binding { name: Ident, pattern: Box<Pattern> },
    /// Operation pattern with positional args: `Add(x, y)` or `End(comp, ..)`
    OpTuple { op: Ident, args: Vec<Pattern>, rest: bool },
    /// Operation pattern with named fields: `Bufferize { compute: x, .. }`
    OpStruct { op: Ident, fields: Vec<FieldPattern>, rest: bool },
    /// Constant pattern: `Const(value)` or `Const(_)`
    Const(ConstPattern),
    /// Operation variable reference: `op(x, y)` where `op` is an iteration variable
    OpVar { var_name: Ident, args: Vec<Pattern> },
    /// Constant with value extraction: `name@const(value_name)`
    /// Binds both the UOp (to uop_name) and the extracted ConstValue (to value_name)
    ConstWithValue { uop_name: Ident, value_name: Ident },
    /// Alternative patterns: `pat1 | pat2 | pat3` - matches if ANY alternative matches
    Any(Vec<Pattern>),
    /// Permutation pattern: `Add[x, y]` - tries all orderings of arguments
    OpPermute { op: Ident, args: Vec<Pattern> },
}

/// A named field in a struct-style pattern.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FieldPattern {
    pub name: Ident,
    pub pattern: Pattern,
}

/// Constant pattern variants.
#[derive(Debug, Clone)]
pub enum ConstPattern {
    /// Any constant: `Const(_)` or `@const`
    Any,
    /// Specific integer: `Const(0)`
    Int(i64),
    /// Specific float: `Const(1.0)`
    Float(f64),
    /// Zero constant: `@zero` - matches 0 or 0.0
    Zero,
    /// One constant: `@one` - matches 1 or 1.0
    One,
}

impl Parse for Pattern {
    fn parse(input: ParseStream) -> Result<Self> {
        // Parse single pattern first, then check for alternatives
        let first = parse_single_pattern(input)?;

        // Check for pipe operator to build alternatives
        if input.peek(Token![|]) {
            let mut alternatives = vec![first];
            while input.peek(Token![|]) {
                input.parse::<Token![|]>()?;
                alternatives.push(parse_single_pattern(input)?);
            }
            return Ok(Pattern::Any(alternatives));
        }

        Ok(first)
    }
}

/// Parse a single pattern (without checking for alternatives).
fn parse_single_pattern(input: ParseStream) -> Result<Pattern> {
    // Check for parenthesized group: `(pattern)` or `(pat1 | pat2)`
    if input.peek(token::Paren) {
        // Fork to check if this is an operation-name-alternatives pattern: (Add | Mul)(args)
        let fork = input.fork();
        let content;
        parenthesized!(content in fork);

        // Check if this is just identifiers separated by pipes followed by tuple args
        if is_op_alternatives(&content) && fork.peek(token::Paren) {
            // Parse: (Add | Mul)(args)
            let content;
            parenthesized!(content in input);
            let ops = parse_op_alternatives(&content)?;

            // Now parse the args
            let args_content;
            parenthesized!(args_content in input);
            let (args, rest) = parse_pattern_args(&args_content)?;

            // Build alternatives for each op
            let alternatives: Vec<Pattern> =
                ops.into_iter().map(|op| Pattern::OpTuple { op, args: args.clone(), rest }).collect();

            return Ok(Pattern::Any(alternatives));
        }

        // Regular grouped pattern: (pattern) or (pat1 | pat2)
        let content;
        parenthesized!(content in input);
        return content.parse();
    }

    // Check for wildcard
    if input.peek(Token![_]) {
        input.parse::<Token![_]>()?;
        return Ok(Pattern::Wildcard);
    }

    // Check for special constant syntax: @zero, @one, @const
    if input.peek(Token![@]) {
        input.parse::<Token![@]>()?;
        // Use parse_any to allow keywords like `const`
        let ident: Ident = Ident::parse_any(input)?;
        return match ident.to_string().as_str() {
            "zero" => Ok(Pattern::Const(ConstPattern::Zero)),
            "one" => Ok(Pattern::Const(ConstPattern::One)),
            "const" => Ok(Pattern::Const(ConstPattern::Any)),
            other => Err(syn::Error::new_spanned(
                ident,
                format!("Unknown special constant '@{}'. Use @zero, @one, or @const", other),
            )),
        };
    }

    // Parse identifier
    let ident: Ident = input.parse()?;

    // Check for name@const(value) or name @ pattern
    if input.peek(Token![@]) {
        // Use lookahead to check for @const(value) syntax
        let lookahead = input.fork();
        lookahead.parse::<Token![@]>()?;

        // Check if next is "const" keyword followed by parentheses
        if lookahead.peek(Ident::peek_any) {
            let maybe_const: Ident = Ident::parse_any(&lookahead)?;
            if maybe_const == "const" && lookahead.peek(token::Paren) {
                // Consume @ and const from real stream
                input.parse::<Token![@]>()?;
                let _: Ident = Ident::parse_any(input)?; // "const"

                // Parse (value_name)
                let content;
                parenthesized!(content in input);
                let value_name: Ident = content.parse()?;

                return Ok(Pattern::ConstWithValue { uop_name: ident, value_name });
            }
        }

        // Fall through to regular binding pattern: `name @ pattern`
        input.parse::<Token![@]>()?;
        let pattern = parse_single_pattern(input)?;
        return Ok(Pattern::Binding { name: ident, pattern: Box::new(pattern) });
    }

    // Check for permutation pattern: `Op[args]`
    if input.peek(token::Bracket) {
        let content;
        bracketed!(content in input);

        let (args, _rest) = parse_pattern_args(&content)?;
        return Ok(Pattern::OpPermute { op: ident, args });
    }

    // Check for operation pattern with tuple args: `Op(args)` or `op(args)` (iteration variable)
    if input.peek(token::Paren) {
        let content;
        parenthesized!(content in input);

        // Special case for Const
        if ident == "Const" {
            return parse_const_pattern(&content);
        }

        let (args, rest) = parse_pattern_args(&content)?;

        // If the identifier starts with lowercase, treat as operation variable
        // (Real op names like Add, Mul start with uppercase)
        let is_lowercase = ident.to_string().chars().next().is_some_and(|c| c.is_lowercase());

        if is_lowercase {
            return Ok(Pattern::OpVar { var_name: ident, args });
        }

        return Ok(Pattern::OpTuple { op: ident, args, rest });
    }

    // Check for operation pattern with struct fields: `Op { fields }`
    if input.peek(token::Brace) {
        let content;
        braced!(content in input);

        let mut fields = Vec::new();
        let mut rest = false;

        while !content.is_empty() {
            // Check for `..`
            if content.peek(Token![..]) {
                content.parse::<Token![..]>()?;
                rest = true;
                break;
            }

            // Parse field: `name: pattern` or just `name`
            let field_name: Ident = content.parse()?;

            let pattern = if content.peek(Token![:]) {
                content.parse::<Token![:]>()?;
                content.parse()?
            } else {
                // Shorthand: `name` means `name: name`
                Pattern::Var(field_name.clone())
            };

            fields.push(FieldPattern { name: field_name, pattern });

            // Allow trailing comma
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }

        return Ok(Pattern::OpStruct { op: ident, fields, rest });
    }

    // Just a variable
    Ok(Pattern::Var(ident))
}

/// Parse pattern arguments from a parenthesized or bracketed group.
fn parse_pattern_args(content: ParseStream) -> Result<(Vec<Pattern>, bool)> {
    let mut args = Vec::new();
    let mut rest = false;

    while !content.is_empty() {
        // Check for `..` (rest pattern)
        if content.peek(Token![..]) {
            content.parse::<Token![..]>()?;
            rest = true;
            break;
        }

        // Parse pattern
        args.push(content.parse()?);

        // Handle comma
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }

    Ok((args, rest))
}

/// Check if content contains only identifiers separated by pipes (for op alternatives).
fn is_op_alternatives(content: ParseStream) -> bool {
    let fork = content.fork();

    // Must start with an identifier
    if fork.parse::<Ident>().is_err() {
        return false;
    }

    // Check for pattern: (Ident (| Ident)*)
    while fork.peek(Token![|]) {
        if fork.parse::<Token![|]>().is_err() {
            return false;
        }
        if fork.parse::<Ident>().is_err() {
            return false;
        }
    }

    // Must be fully consumed
    fork.is_empty()
}

/// Parse operation alternatives: `Add | Mul | Sub`
fn parse_op_alternatives(content: ParseStream) -> Result<Vec<Ident>> {
    let mut ops = vec![content.parse::<Ident>()?];
    while content.peek(Token![|]) {
        content.parse::<Token![|]>()?;
        ops.push(content.parse::<Ident>()?);
    }
    Ok(ops)
}

fn parse_const_pattern(input: ParseStream) -> Result<Pattern> {
    // Check for wildcard
    if input.peek(Token![_]) {
        input.parse::<Token![_]>()?;
        return Ok(Pattern::Const(ConstPattern::Any));
    }

    // Try to parse a literal
    let lit: syn::Lit = input.parse()?;
    match lit {
        syn::Lit::Int(i) => {
            let value: i64 = i.base10_parse()?;
            Ok(Pattern::Const(ConstPattern::Int(value)))
        }
        syn::Lit::Float(f) => {
            let value: f64 = f.base10_parse()?;
            Ok(Pattern::Const(ConstPattern::Float(value)))
        }
        _ => Err(syn::Error::new_spanned(lit, "Expected integer or float literal in Const pattern")),
    }
}

/// The right-hand side of a rewrite rule.
#[derive(Debug)]
pub enum RewriteExpr {
    /// Simple variable reference: `x`
    Var(Ident),
    /// Block expression: `{ ... }`
    Block(syn::ExprBlock),
    /// General expression: `Rc::clone(x)`, `foo.bar()`, etc.
    Expr(syn::Expr),
}

impl Parse for RewriteExpr {
    fn parse(input: ParseStream) -> Result<Self> {
        // Check for block
        if input.peek(token::Brace) {
            let block: syn::ExprBlock = input.parse()?;
            return Ok(RewriteExpr::Block(block));
        }

        // Check for simple variable: ident followed by comma/end
        // (guards are now parsed before the arrow, so no need to check for `if`)
        if input.peek(Ident) {
            let fork = input.fork();
            let _: Ident = fork.parse()?;
            if fork.peek(Token![,]) || fork.is_empty() {
                let ident: Ident = input.parse()?;
                return Ok(RewriteExpr::Var(ident));
            }
        }

        // General expression: parse token trees until separator
        // (no guard detection needed - guards are parsed before arrow)
        let mut tokens = proc_macro2::TokenStream::new();

        while !input.is_empty() {
            // Stop at comma (pattern separator)
            if input.peek(Token![,]) {
                break;
            }

            // Consume next token tree (handles grouping automatically)
            let tt: proc_macro2::TokenTree = input.parse()?;
            tokens.extend(std::iter::once(tt));
        }

        if tokens.is_empty() {
            return Err(input.error("expected expression"));
        }

        let expr: syn::Expr = syn::parse2(tokens)?;
        Ok(RewriteExpr::Expr(expr))
    }
}
