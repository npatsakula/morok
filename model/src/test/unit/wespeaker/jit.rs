use crate::jit::InputSpec;
use crate::wespeaker::{WeSpeakerConfig, WeSpeakerResNet34, WeSpeakerResNet34Jit};

fn build_jit(max_batch: usize) -> WeSpeakerResNet34Jit {
    let cfg = WeSpeakerConfig::new().with_max_batch_size(max_batch);
    let model = WeSpeakerResNet34::with_zero_weights(cfg);
    WeSpeakerResNet34Jit::new(model)
}

/// Smoke: prepare and execute the JIT graph at `max_batch`. Confirms both
/// input buffers (`feats` + `weights`) get wired up and the output is the
/// expected `[B, 256]` size.
#[test]
#[ignore = "heavy: full WeSpeaker ResNet34 graph compile through the CPU backend"]
fn prepare_and_execute_at_max_batch() {
    let max_batch = 1;
    let mut jit = build_jit(max_batch);
    jit.prepare(InputSpec::f32(&[max_batch, 1598, 80]), InputSpec::f32(&[max_batch, 799])).unwrap();
    jit.execute_bound(max_batch as i64).unwrap();
    assert_eq!(jit.embeddings_shape().unwrap(), vec![max_batch, 256]);
    assert_eq!(jit.embeddings_to_vec::<f32>().unwrap().len(), max_batch * 256);
}
