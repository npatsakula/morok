use super::generate;
use syn::{DeriveInput, parse_quote};

/// The expansion with every space removed, so assertions can be written as
/// compact needles instead of `quote!`'s token spacing.
fn expand(input: DeriveInput) -> String {
    let tokens = generate(&input).expect("expansion failed");
    tokens.to_string().replace([' ', '\n'], "")
}

fn error(input: DeriveInput) -> String {
    generate(&input).expect_err("expected a compile error").to_string()
}

#[test]
fn tensor_and_child_fields_use_prefixed_keys() {
    let out = expand(parse_quote! {
        struct Attention {
            query: Tensor,
            #[module(key = "output.dense.weight")]
            out_weight: Tensor,
            proj: Linear,
            layers: Vec<Layer>,
        }
    });
    assert!(out.contains(r#"__out.insert(::svod_tensor::nn::prefixed(__prefix,"query"),self.query.clone());"#));
    assert!(out.contains(
        r#"self.query=::svod_tensor::nn::get_tensor(__sd,&::svod_tensor::nn::prefixed(__prefix,"query"))?;"#
    ));
    assert!(out.contains(r#"prefixed(__prefix,"output.dense.weight")"#));
    assert!(out.contains(
        r#"::svod_tensor::nn::Module::write_state(&self.proj,&::svod_tensor::nn::prefixed(__prefix,"proj"),__out);"#
    ));
    assert!(out.contains(
        r#"::svod_tensor::nn::Module::load_state_dict(&mutself.layers,__sd,&::svod_tensor::nn::prefixed(__prefix,"layers"))?;"#
    ));
    // No `format!` anywhere: every key is built through `prefixed`.
    assert!(!out.contains("format!"));
}

#[test]
fn primitives_and_skipped_fields_vanish() {
    let out = expand(parse_quote! {
        struct Block {
            weight: Tensor,
            eps: f64,
            name: String,
            dims: Vec<usize>,
            shape: [usize; 4],
            pair: (usize, bool),
            maybe: Option<usize>,
            #[module(skip)]
            mode: SubsamplingMode,
            #[module(skip)]
            kinds: Vec<BlockKind>,
        }
    });
    for absent in ["eps", "name", "dims", "shape", "pair", "maybe", "mode", "kinds"] {
        assert!(!out.contains(&format!("\"{absent}\"")), "{absent} should not be a key");
        assert!(!out.contains(&format!("self.{absent}")), "{absent} should not be read");
    }
    assert!(out.contains(r#""weight""#));
}

#[test]
fn optional_tensor_needs_the_attribute_and_is_absent_tolerant() {
    let out = expand(parse_quote! {
        struct Linear {
            weight: Tensor,
            #[module(optional)]
            bias: Option<Tensor>,
        }
    });
    assert!(out.contains(r#"iflet::core::option::Option::Some(__t)=&self.bias{__out.insert(::svod_tensor::nn::prefixed(__prefix,"bias"),__t.clone());}"#));
    assert!(out.contains(r#"self.bias=__sd.get(&::svod_tensor::nn::prefixed(__prefix,"bias")).cloned();"#));

    let err = error(parse_quote! {
        struct Linear {
            bias: Option<Tensor>,
        }
    });
    assert!(err.contains("no blanket `Module` impl"), "{err}");
}

#[test]
fn optional_predicate_makes_the_key_required() {
    let out = expand(parse_quote! {
        struct ConvLayerBlock {
            conv_weight: Tensor,
            #[module(key = "conv.bias", optional = "self.conv_bias.is_some()")]
            conv_bias: Option<Tensor>,
        }
    });
    assert!(out.contains("let__want:bool=self.conv_bias.is_some();"));
    assert!(out.contains(
        r#"Option::Some(::svod_tensor::nn::get_tensor(__sd,&::svod_tensor::nn::prefixed(__prefix,"conv.bias"))?)"#
    ));
    assert!(out.contains("}else{::core::option::Option::None};"));
}

#[test]
fn empty_key_flattens_onto_the_parent_prefix() {
    let out = expand(parse_quote! {
        struct Encoder {
            #[module(skip)]
            config: ModernBertConfig,
            #[module(key = "")]
            layers: Vec<EncoderLayer>,
        }
    });
    assert!(out.contains("::svod_tensor::nn::Module::write_state(&self.layers,__prefix,__out);"));
    assert!(out.contains("::svod_tensor::nn::Module::load_state_dict(&mutself.layers,__sd,__prefix)?;"));
    assert!(!out.contains(r#"prefixed(__prefix,"layers")"#));
    assert!(!out.contains("config"));
}

#[test]
fn tuple_struct_fields_are_indexed() {
    let out = expand(parse_quote! {
        struct Downsample(Conv2dWeights, BatchNormWeights);
    });
    assert!(out.contains(r#"write_state(&self.0,&::svod_tensor::nn::prefixed(__prefix,"0"),__out);"#));
    assert!(out.contains(r#"write_state(&self.1,&::svod_tensor::nn::prefixed(__prefix,"1"),__out);"#));
    // A tuple struct has no named fields, so the table is empty.
    assert!(out.contains("MODULE_FIELDS:&'static[(&'staticstr,&'staticstr)]=&[];"));
}

#[test]
fn module_fields_table_records_key_segments() {
    let out = expand(parse_quote! {
        struct Attention {
            #[module(key = "self.query.weight")]
            query_weight: Tensor,
            #[module(key = "")]
            inner: Encoder,
            #[module(skip)]
            heads: Config,
            n_heads: usize,
        }
    });
    assert!(out.contains(r#"=&[("query_weight","self.query.weight"),("inner","")];"#));
}

#[test]
fn enum_newtype_variants_are_transparent() {
    let out = expand(parse_quote! {
        enum Block {
            Basic(BasicBlock),
            Bottleneck(Bottleneck),
            Empty,
        }
    });
    assert!(out.contains("Self::Basic(__f0)=>{::svod_tensor::nn::Module::write_state(&(*__f0),__prefix,__out);}"));
    assert!(
        out.contains(
            "Self::Bottleneck(__f0)=>{::svod_tensor::nn::Module::load_state_dict(&mut(*__f0),__sd,__prefix)?;}"
        )
    );
    assert!(out.contains("Self::Empty=>{}"));
    assert!(out.contains("matchself{"));
}

#[test]
fn enum_tuple_and_struct_variants_key_by_index_and_name() {
    let out = expand(parse_quote! {
        enum C3k2Inner {
            Attn(AttnBottleneck, Conv),
            Norm { weight: Tensor, #[module(skip)] kind: NormKind },
            #[module(key = "wavlm_model")]
            Nested(Backbone),
        }
    });
    assert!(out.contains(r#"Self::Attn(__f0,__f1)=>{"#));
    assert!(out.contains(r#"write_state(&(*__f0),&::svod_tensor::nn::prefixed(__prefix,"0"),__out);"#));
    assert!(out.contains(r#"write_state(&(*__f1),&::svod_tensor::nn::prefixed(__prefix,"1"),__out);"#));
    // The skipped field is dropped from the pattern instead of binding unused.
    assert!(out.contains("Self::Norm{weight,..}=>{"));
    assert!(out.contains(r#"__out.insert(::svod_tensor::nn::prefixed(__prefix,"weight"),(*weight).clone());"#));
    assert!(!out.contains("kind"));
    // A variant key nests the whole variant one segment deeper.
    assert!(out.contains(r#"let__prefix=&::svod_tensor::nn::prefixed(__prefix,"wavlm_model");"#));
}

#[test]
fn generic_parameters_gain_a_module_bound() {
    let out = expand(parse_quote! {
        struct Wrapper<T, C: Clone> {
            inner: T,
            head: Vec<T>,
            #[module(skip)]
            config: C,
        }
    });
    assert!(out.contains("impl<T,C:Clone>::svod_tensor::nn::ModuleforWrapper<T,C>whereT:::svod_tensor::nn::Module{"));
    // The inherent impl keeps the declared bounds only.
    assert!(out.contains("impl<T,C:Clone>Wrapper<T,C>{"));
}

#[test]
fn crate_path_is_overridable() {
    let out = expand(parse_quote! {
        #[module(crate = "::my_tensor")]
        struct Linear {
            weight: Tensor,
        }
    });
    assert!(out.contains("impl::my_tensor::nn::ModuleforLinear"));
    assert!(out.contains(r#"::my_tensor::nn::prefixed(__prefix,"weight")"#));
    assert!(!out.contains("svod_tensor"));
}

#[test]
fn bad_attributes_report_at_their_span() {
    let unknown = error(parse_quote! {
        struct S {
            #[module(rename = "x")]
            weight: Tensor,
        }
    });
    assert!(unknown.contains("unknown `module` attribute"), "{unknown}");

    let misplaced = error(parse_quote! {
        struct S {
            #[module(optional)]
            weight: Tensor,
        }
    });
    assert!(misplaced.contains("applies only to an `Option<Tensor>` field"), "{misplaced}");

    let unit = error(parse_quote! {
        enum E {
            #[module(key = "x")]
            Empty,
        }
    });
    assert!(unit.contains("unit variant"), "{unit}");

    let on_variant = error(parse_quote! {
        enum E {
            #[module(optional)]
            V(Inner),
        }
    });
    assert!(on_variant.contains("belongs on a field"), "{on_variant}");

    let union = error(parse_quote! {
        union U { a: u32 }
    });
    assert!(union.contains("cannot be derived for a union"), "{union}");
}
