fn main() {
    prost_build::compile_protos(&["proto/onnx.proto"], &["proto/"]).unwrap();
}
