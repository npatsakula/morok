use std::path::Path;

fn main() {
    prost_build::compile_protos(&["proto/onnx.proto"], &["proto/"]).unwrap();
    generate_node_tests();
}

fn generate_node_tests() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let node_dir = Path::new(&manifest_dir).join("../submodules/onnx/onnx/backend/test/data/node");

    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("onnx_node_tests.rs");

    if !node_dir.exists() {
        std::fs::write(&out_path, "// ONNX submodule not found\n").unwrap();
        return;
    }

    let mut test_names: Vec<String> = std::fs::read_dir(&node_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    test_names.sort();

    let mut code = String::new();
    for name in &test_names {
        let ignored = should_skip(name);
        code.push_str(&format!(
            "#[test]\n{attr}fn {name}() {{\n    \
               run_onnx_node_test(concat!(env!(\"CARGO_MANIFEST_DIR\"), \
               \"/../submodules/onnx/onnx/backend/test/data/node/{name}\"));\n}}\n\n",
            attr = if ignored { "#[ignore]\n" } else { "" },
        ));
    }

    std::fs::write(&out_path, code).unwrap();
}

fn should_skip(name: &str) -> bool {
    const SKIP_PREFIXES: &[&str] = &[
        // ML domain ops
        "test_ai_onnx_ml_",
        // String ops
        "test_string_",
        "test_strnormalizer_",
        "test_tfidfvectorizer_",
        // Sequence ops
        "test_sequence_",
        "test_split_to_sequence_",
        // Control flow iteration (we support If, not Loop/Scan)
        "test_loop",
        "test_scan",
        // Recurrent ops (LSTM/GRU unimplemented, batchwise layout unsupported)
        "test_lstm_",
        "test_gru_",
        "test_simple_rnn_batchwise",
        // Quantization
        "test_quantize",
        "test_dequantize",
        "test_qlinear",
        "test_dynamicquantize",
        "test_convinteger",
        // Training ops
        "test_training_",
        "test_adam_",
        // Image decoding
        "test_image_decoder_",
        // Deformable convolution
        "test_basic_deform_conv",
        "test_deform_conv",
        // Column-to-image
        "test_col2im",
        // NMS
        "test_nonmaxsuppression_",
        // ROI align
        "test_roialign_",
        // Unique values
        "test_unique_",
        // Signal processing
        "test_dft",
        "test_stft",
        "test_melweight",
        // Max unpooling
        "test_maxunpool_",
        // Lp pooling
        "test_lppool_",
        // Matrix determinant
        "test_det_",
        // Bitcast reinterpretation
        "test_bitcast_",
        // Window functions
        "test_hannwindow",
        "test_hammingwindow",
        "test_blackmanwindow",
        // Random (non-deterministic)
        "test_bernoulli",
        // Cumulative product
        "test_cumprod_",
        // Reverse sequence
        "test_reversesequence_",
        // New opset 24 scatter
        "test_tensorscatter",
        // Optional type ops
        "test_optional_",
        // Affine grid generation
        "test_affine_grid",
        // Center crop pad
        "test_center_crop_pad",
        // Exotic dtype casts
        "test_cast_e8m0_",
        "test_cast_no_saturate_",
    ];

    const SKIP_EXACT: &[&str] = &[
        "test_batchnorm_example_training_mode",
        "test_batchnorm_epsilon_training_mode",
        "test_dropout_random_old",
        "test_constantofshape_int_shape_zero",
        // If variants using sequence/optional types in subgraphs
        "test_if_seq",
        "test_if_opt",
        // Identity variants using optional/sequence container types
        "test_identity_opt",
        "test_identity_sequence",
    ];

    const SKIP_CONTAINS: &[&str] =
        &["_expanded", "FLOAT8", "INT4", "UINT4", "INT2", "UINT2", "FLOAT4E2M1", "FLOAT8E8M0", "COMPLEX"];

    SKIP_PREFIXES.iter().any(|p| name.starts_with(p))
        || SKIP_EXACT.contains(&name)
        || SKIP_CONTAINS.iter().any(|c| name.contains(c))
}
