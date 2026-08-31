//! Shared input-buffer packing for the ModernBERT heads: pad each chunk's
//! `input_ids` / `attention_mask` into the `[max_batch, max_seq]` i64 JIT
//! buffers the compiled plans read. Generic JIT-buffer packing, independent of
//! any particular head.

use svod_arch::pipelines::text::Encoding;

/// Pad each chunk's `input_ids` into the `[max_batch, max_seq]` i64 JIT buffer,
/// zero-filling past each chunk's length and over unused rows.
pub(crate) fn pack_ids_buffer(
    buf: &mut svod_device::Buffer,
    batch: &[&Encoding],
    max_seq: usize,
) -> Result<(), svod_device::error::Error> {
    let mut view = buf.as_array_mut::<i64>()?;
    let slice = view.as_slice_mut().expect("contiguous ids buffer");
    slice.fill(0);
    for (i, enc) in batch.iter().enumerate() {
        let take = enc.input_ids.len().min(max_seq);
        for (j, &id) in enc.input_ids[..take].iter().enumerate() {
            slice[i * max_seq + j] = id as i64;
        }
    }
    Ok(())
}

/// Pad each chunk's `attention_mask` (1 = real token, 0 = pad) into the
/// `[max_batch, max_seq]` i64 JIT buffer. The mask follows the chunk's real
/// token count; trailing pad positions and unused rows stay 0.
pub(crate) fn pack_mask_buffer(
    buf: &mut svod_device::Buffer,
    batch: &[&Encoding],
    max_seq: usize,
) -> Result<(), svod_device::error::Error> {
    let mut view = buf.as_array_mut::<i64>()?;
    let slice = view.as_slice_mut().expect("contiguous mask buffer");
    slice.fill(0);
    for (i, enc) in batch.iter().enumerate() {
        let take = enc.attention_mask.len().min(max_seq);
        for (j, &m) in enc.attention_mask[..take].iter().enumerate() {
            slice[i * max_seq + j] = m as i64;
        }
    }
    Ok(())
}
