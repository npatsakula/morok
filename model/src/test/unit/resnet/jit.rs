use crate::jit::InputSpec;
use crate::resnet::{OutputMode, ResNet, ResNetConfig, ResNetDepth, ResNetJit};

fn build_classifier_jit(max_batch: usize, num_classes: usize) -> ResNetJit {
    let config =
        ResNetConfig::new(ResNetDepth::R18, OutputMode::Classification { num_classes }).with_max_batch_size(max_batch);
    let model = ResNet::with_zero_weights(config);
    ResNetJit::new(model)
}

/// Realize-based JIT smoke gated behind `--ignored`. The default suite covers
/// the JIT mechanics through the toy `jit_recurrent` tests and the GigaAM
/// batch suite (which actually exercises variable rebinding); this is here
/// for hands-on ResNet work.
#[test]
#[ignore = "heavy: full ResNet-18 graph compile through the CPU backend"]
fn prepare_and_execute_at_max_batch() {
    let max_batch = 2;
    let mut jit = build_classifier_jit(max_batch, 5);
    jit.prepare(InputSpec::f32(&[max_batch, 3, 32, 32])).unwrap();
    jit.execute_bound(max_batch as i64).unwrap();
    assert_eq!(jit.logits_shape().unwrap(), vec![max_batch, 5]);
    assert_eq!(jit.logits_to_vec::<f32>().unwrap().len(), max_batch * 5);
}

#[test]
#[ignore = "heavy: three executes of the full ResNet-18 graph"]
fn rebind_batch_without_reprepare() {
    let max_batch = 4;
    let mut jit = build_classifier_jit(max_batch, 3);
    jit.prepare(InputSpec::f32(&[max_batch, 3, 32, 32])).unwrap();

    // The live output shape tracks `b` without a re-prepare.
    for b in [1, 2, 4] {
        jit.execute_bound(b as i64).unwrap();
        assert_eq!(jit.logits_shape().unwrap(), vec![b, 3], "logits shape for b={b}");
    }
}

#[test]
#[ignore = "heavy: full ResNet-18 graph compile through the CPU backend"]
fn features_mode_returns_spatial_map() {
    let max_batch = 1;
    let config = ResNetConfig::new(ResNetDepth::R18, OutputMode::Features).with_max_batch_size(max_batch);
    let model = ResNet::with_zero_weights(config);
    let mut jit = ResNetJit::new(model);
    jit.prepare(InputSpec::f32(&[max_batch, 3, 32, 32])).unwrap();
    jit.execute_bound(1).unwrap();
    assert_eq!(jit.logits_shape().unwrap(), vec![1, 512, 1, 1]);
}
