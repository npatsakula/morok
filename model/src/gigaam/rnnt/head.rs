//! RNN-T head = predictor + joint. Composed similarly to Python `RNNTHead`.

use svod_tensor::nn::Module;

use super::joint::RnntJoint;
use super::predictor::RnntPredictor;

#[derive(Clone, Module)]
pub struct RnntHead {
    pub predictor: RnntPredictor,
    pub joint: RnntJoint,
    pub pred_rnn_layers: usize,
    pub pred_hidden: usize,
    pub joint_hidden: usize,
    pub num_classes: usize,
}

impl RnntHead {
    pub fn empty(
        enc_hidden: usize,
        pred_hidden: usize,
        pred_rnn_layers: usize,
        joint_hidden: usize,
        num_classes: usize,
    ) -> Self {
        Self {
            predictor: RnntPredictor::empty(pred_hidden, pred_rnn_layers, num_classes),
            joint: RnntJoint::empty(enc_hidden, pred_hidden, joint_hidden, num_classes),
            pred_rnn_layers,
            pred_hidden,
            joint_hidden,
            num_classes,
        }
    }
}
