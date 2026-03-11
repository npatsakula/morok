#![allow(non_snake_case)]

use std::sync::Once;
use tracing_subscriber::EnvFilter;

static TRACING_INIT: Once = Once::new();

pub fn setup_tracing() {
    TRACING_INIT.call_once(|| {
        tracing_subscriber::fmt()
            .json()
            .with_current_span(false)
            .with_span_list(false)
            .with_env_filter(EnvFilter::from_default_env())
            .with_test_writer()
            .init();
    });
}

use super::helpers::run_onnx_light_test;
include!(concat!(env!("OUT_DIR"), "/onnx_light_tests.rs"));
