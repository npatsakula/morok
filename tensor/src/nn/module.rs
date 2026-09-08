//! The Module trait: state-dict hooks for layer structs.
//!
//! A [`StateDict`] is a flat `name -> Tensor` map using PyTorch's dotted key
//! convention. [`Module`] both writes and reads it; implement it by hand or,
//! far more usually, derive it with [`#[derive(Module)]`](svod_macros::Module),
//! which classifies fields by type and routes every key through [`prefixed`].
//!
//! Forward passes stay out of this trait: they live in [`Layer`](super::Layer)
//! when the signature is `&Tensor -> Tensor`, and in inherent methods when it
//! is not.

use std::collections::HashMap;

use snafu::OptionExt;

use crate::Tensor;
use crate::error::{MissingKeySnafu, Result};

/// A flat map of parameter name to tensor, keyed as PyTorch names them.
pub type StateDict = HashMap<String, Tensor>;

/// Save and load a module's parameters through a [`StateDict`].
///
/// `prefix` is the dotted path of the module itself: `""` at the root, so
/// keys come out bare, and e.g. `"encoder.0"` for a nested child.
pub trait Module {
    /// Insert every parameter of this module into `out` under `prefix`.
    fn write_state(&self, prefix: &str, out: &mut StateDict);

    /// Replace every parameter of this module with the one `sd` holds under
    /// `prefix`, failing with [`ErrorKind::MissingKey`](crate::error::ErrorKind::MissingKey)
    /// on the first required key that is absent.
    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> Result<()>;

    /// This module's parameters as a fresh dict.
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut out = StateDict::new();
        self.write_state(prefix, &mut out);
        out
    }

    /// Call `f` on every `(key, parameter)` pair, in unspecified order.
    fn visit_params(&self, prefix: &str, f: &mut dyn FnMut(&str, &Tensor)) {
        for (key, tensor) in self.state_dict(prefix) {
            f(&key, &tensor);
        }
    }
}

/// Join a module prefix and a field name, dropping the dot at the root.
pub fn prefixed(prefix: &str, name: &str) -> String {
    if prefix.is_empty() { name.to_string() } else { format!("{prefix}.{name}") }
}

/// Look up a required parameter.
pub fn get_tensor(sd: &StateDict, key: &str) -> Result<Tensor> {
    Ok(sd.get(key).cloned().context(MissingKeySnafu { key })?)
}

impl Module for Tensor {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        out.insert(prefix.to_string(), self.clone());
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> Result<()> {
        *self = get_tensor(sd, prefix)?;
        Ok(())
    }
}

fn write_seq<'a, M: Module + 'a>(it: impl Iterator<Item = &'a M>, prefix: &str, out: &mut StateDict) {
    for (i, m) in it.enumerate() {
        m.write_state(&prefixed(prefix, &i.to_string()), out);
    }
}

fn load_seq<'a, M: Module + 'a>(it: impl Iterator<Item = &'a mut M>, sd: &StateDict, prefix: &str) -> Result<()> {
    for (i, m) in it.enumerate() {
        m.load_state_dict(sd, &prefixed(prefix, &i.to_string()))?;
    }
    Ok(())
}

impl<M: Module> Module for Vec<M> {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        write_seq(self.iter(), prefix, out)
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> Result<()> {
        load_seq(self.iter_mut(), sd, prefix)
    }
}

impl<M: Module, const N: usize> Module for [M; N] {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        write_seq(self.iter(), prefix, out)
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> Result<()> {
        load_seq(self.iter_mut(), sd, prefix)
    }
}

/// A `None` child neither writes nor loads: the shape of the module is fixed
/// by its owner, not by the dict. `Option<Tensor>` deliberately has no impl —
/// a missing optional parameter must clear the field, which this cannot do, so
/// the derive handles it through `#[module(optional)]`.
impl<M: Module> Module for Option<M> {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        if let Some(m) = self {
            m.write_state(prefix, out)
        }
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> Result<()> {
        match self {
            Some(m) => m.load_state_dict(sd, prefix),
            None => Ok(()),
        }
    }
}

impl<M: Module> Module for Box<M> {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        (**self).write_state(prefix, out)
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> Result<()> {
        (**self).load_state_dict(sd, prefix)
    }
}

impl<A: Module, B: Module> Module for (A, B) {
    fn write_state(&self, prefix: &str, out: &mut StateDict) {
        self.0.write_state(&prefixed(prefix, "0"), out);
        self.1.write_state(&prefixed(prefix, "1"), out);
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> Result<()> {
        self.0.load_state_dict(sd, &prefixed(prefix, "0"))?;
        self.1.load_state_dict(sd, &prefixed(prefix, "1"))
    }
}
