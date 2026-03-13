#![allow(non_snake_case)]

fn setup_tracing() {
    morok_schedule::testing::setup_test_tracing();
}

use super::helpers::run_onnx_node_test;
include!(concat!(env!("OUT_DIR"), "/onnx_node_tests.rs"));
