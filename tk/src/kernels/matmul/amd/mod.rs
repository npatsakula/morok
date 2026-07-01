//! Per-arch AMD specializations of the tile matmul: the gfx942 (CDNA3, wave64)
//! inline-`asm` HK-pipeline microkernel + size-adaptive configs ([`gfx942`]) and the
//! gfx1151 (RDNA3.5, wave32 WMMA) config home ([`gfx1151`]).

pub mod gfx1151;
pub mod gfx942;
