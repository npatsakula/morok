use crate::amd::iface::{AllocKind, compose_flags};
use crate::amd::sys::kfd;

/// On an APU, device buffers have no VRAM to live in, so `DeviceVram` must be
/// allocated from GTT (system memory via the GART) — same modifier bits, just a
/// different heap. Pure function, no global state: safe under parallel tests.
#[test]
fn device_vram_uses_gtt_on_apu_vram_on_discrete() {
    let discrete = compose_flags(AllocKind::DeviceVram, /*cpu_access=*/ false, /*is_apu=*/ false);
    let apu = compose_flags(AllocKind::DeviceVram, /*cpu_access=*/ false, /*is_apu=*/ true);

    // Heap bit flips VRAM<->GTT and the two are mutually exclusive.
    assert_ne!(discrete & kfd::KFD_IOC_ALLOC_MEM_FLAGS_VRAM, 0);
    assert_eq!(discrete & kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT, 0);
    assert_ne!(apu & kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT, 0);
    assert_eq!(apu & kfd::KFD_IOC_ALLOC_MEM_FLAGS_VRAM, 0);

    // Everything except the heap bit is identical (modifier bits unchanged, so
    // the copy paths' coherence contract is preserved).
    let mask = !(kfd::KFD_IOC_ALLOC_MEM_FLAGS_VRAM | kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT);
    assert_eq!(discrete & mask, apu & mask);
}

/// `DeviceVram` carries the tinygrad/ROCr base flags (WRITABLE | EXECUTABLE |
/// NO_SUBSTITUTE) unconditionally; `cpu_access` adds PUBLIC on both heaps.
#[test]
fn device_vram_modifier_bits() {
    for is_apu in [false, true] {
        let plain = compose_flags(AllocKind::DeviceVram, false, is_apu);
        let public = compose_flags(AllocKind::DeviceVram, true, is_apu);
        assert_eq!(plain & kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC, 0);
        assert_ne!(public & kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC, 0);
        assert_ne!(plain & kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE, 0);
        assert_ne!(plain & kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE, 0);
        assert_ne!(plain & kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE, 0);
    }
}

/// The control-structure heap is GTT regardless of device type (arch-independent)
/// and carries the full tinygrad/ROCr control-page flag set — including
/// fine-grained COHERENT, which the completion-signal handshake and the CP's
/// rptr/wptr reads rely on across every gfx generation.
#[test]
fn uncached_gtt_is_heap_independent() {
    let d = compose_flags(AllocKind::UncachedGtt, true, false);
    let a = compose_flags(AllocKind::UncachedGtt, true, true);
    assert_eq!(d, a);
    assert_ne!(d & kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT, 0);
    assert_ne!(d & kfd::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT, 0);
    assert_ne!(d & kfd::KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED, 0);
    assert_ne!(d & kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE, 0);
    assert_ne!(d & kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC, 0);
    assert_eq!(d & kfd::KFD_IOC_ALLOC_MEM_FLAGS_VRAM, 0);
}
