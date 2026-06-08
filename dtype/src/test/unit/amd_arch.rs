use super::*;

#[test]
fn version_round_trip() {
    for arch in
        [AmdArch::Gfx942, AmdArch::Gfx1030, AmdArch::Gfx1034, AmdArch::Gfx1036, AmdArch::Gfx1100, AmdArch::Gfx1201]
    {
        let s = arch.mcpu();
        assert_eq!(AmdArch::parse(s), Some(arch));
    }
}

#[test]
fn family_predicates() {
    assert!(AmdArch::Gfx942.is_cdna() && !AmdArch::Gfx942.is_rdna3());
    assert!(AmdArch::Gfx1100.is_rdna3() && !AmdArch::Gfx1100.is_cdna());
    assert!(AmdArch::Gfx1201.is_rdna4() && !AmdArch::Gfx1201.is_rdna3());
    assert!(AmdArch::Gfx1100.has_matrix_cores());
    assert!(AmdArch::Gfx942.has_matrix_cores());
    // RDNA2 (gfx10): its own family, no matrix cores, gfx_major 10, wave32.
    assert!(AmdArch::Gfx1030.is_rdna2() && !AmdArch::Gfx1030.is_cdna() && !AmdArch::Gfx1030.is_rdna3());
    assert!(!AmdArch::Gfx1030.has_matrix_cores());
    assert!(!AmdArch::Gfx1100.is_rdna2() && !AmdArch::Gfx942.is_rdna2());
    assert_eq!(AmdArch::Gfx1030.gfx_major(), 10);
    // RDNA2 APUs (gfx10.3 integrated): still RDNA2, no matrix cores, but flagged
    // as APU dies. gfx1036 = Raphael (Ryzen 7000 iGPU); discrete gfx1030 is not.
    assert!(AmdArch::Gfx1036.is_rdna2() && AmdArch::Gfx1036.is_rdna2_apu());
    assert!(!AmdArch::Gfx1036.has_matrix_cores());
    assert_eq!(AmdArch::Gfx1036.gfx_major(), 10);
    assert!(!AmdArch::Gfx1030.is_rdna2_apu());
    assert!(AmdArch::Gfx1033.is_rdna2_apu() && AmdArch::Gfx1035.is_rdna2_apu());
}

#[test]
fn wave_size_by_family() {
    assert_eq!(AmdArch::Gfx942.wave_size(), 64);
    assert_eq!(AmdArch::Gfx1030.wave_size(), 32);
    assert_eq!(AmdArch::Gfx1100.wave_size(), 32);
    assert_eq!(AmdArch::Gfx1200.wave_size(), 32);
}

#[test]
fn from_kfd_version() {
    assert_eq!(AmdArch::from_gfx_target_version(110_000), Some(AmdArch::Gfx1100));
    assert_eq!(AmdArch::from_gfx_target_version(90_402), Some(AmdArch::Gfx942));
    // RDNA2: gfx1030 (RX 6900 XT) = 100300, gfx1034 = 100304.
    assert_eq!(AmdArch::from_gfx_target_version(100_300), Some(AmdArch::Gfx1030));
    assert_eq!(AmdArch::from_gfx_target_version(100_304), Some(AmdArch::Gfx1034));
    // RDNA2 APU: gfx1036 = Raphael (7950X3D iGPU) = 100306.
    assert_eq!(AmdArch::from_gfx_target_version(100_306), Some(AmdArch::Gfx1036));
    assert_eq!(AmdArch::from_gfx_target_version(100_303), Some(AmdArch::Gfx1033));
    assert_eq!(AmdArch::from_gfx_target_version(0), None);
}
