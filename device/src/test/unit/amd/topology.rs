use crate::amd::topology::*;
use std::io::Write;
use std::path::PathBuf;

#[test]
fn parse_properties_handles_real_kfd_format() {
    let raw = "cpu_cores_count 0\nsimd_count 4\ngfx_target_version 110000\ndrm_render_minor 128\n";
    let map = parse_properties(raw);
    assert_eq!(map["cpu_cores_count"], 0);
    assert_eq!(map["simd_count"], 4);
    assert_eq!(map["gfx_target_version"], 110000);
    assert_eq!(map["drm_render_minor"], 128);
}

#[test]
fn enumerate_returns_empty_when_topology_missing() {
    let temp = tempfile_dir();
    // Point at a non-existent directory.
    unsafe {
        std::env::set_var("SVOD_KFD_TOPOLOGY", temp.join("does_not_exist"));
    }
    let nodes = enumerate();
    unsafe {
        std::env::remove_var("SVOD_KFD_TOPOLOGY");
    }
    assert!(nodes.is_empty());
}

#[test]
fn enumerate_skips_cpu_nodes_and_parses_gpu() {
    let root = tempfile_dir();
    let n0 = root.join("0");
    let n1 = root.join("1");
    std::fs::create_dir_all(&n0).unwrap();
    std::fs::create_dir_all(&n1).unwrap();
    // Node 0: CPU (gpu_id 0).
    let mut f = std::fs::File::create(n0.join("properties")).unwrap();
    write!(f, "gpu_id 0\ncpu_cores_count 32\nsimd_count 0\n").unwrap();
    // Node 1: GPU.
    let mut f = std::fs::File::create(n1.join("properties")).unwrap();
    write!(
            f,
            "gpu_id 5710\nsimd_count 4\narray_count 4\nsimd_arrays_per_engine 2\ngfx_target_version 110000\ndrm_render_minor 128\nwave_front_size 32\nnum_cp_queues 8\n"
        )
        .unwrap();

    unsafe {
        std::env::set_var("SVOD_KFD_TOPOLOGY", &root);
    }
    let nodes = enumerate();
    unsafe {
        std::env::remove_var("SVOD_KFD_TOPOLOGY");
    }
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].node_id, 1);
    assert_eq!(nodes[0].gpu_id, 5710);
    assert_eq!(nodes[0].gfx_target_version, 110000);
    assert_eq!(nodes[0].drm_render_minor, 128);
    assert_eq!(nodes[0].wave_front_size, 32);
    assert_eq!(nodes[0].num_cp_queues, 8);
    assert_eq!(nodes[0].simd_arrays_per_engine, 2);
}

#[test]
fn is_apu_keys_off_cpu_cores_count() {
    // libhsakmt's discrete-GPU test is `!NumCPUCores && NumFComputeCores`, so a
    // GPU node with cpu_cores_count > 0 is an APU. Pure check on constructed
    // nodes — no SVOD_KFD_TOPOLOGY env mutation, so it's race-free under the
    // parallel test runner (the enumerate()-based tests own that global).
    let discrete = gpu_node(/*cpu_cores_count=*/ 0); // e.g. gfx1030 (RX 6900 XT)
    let apu = gpu_node(/*cpu_cores_count=*/ 32); // e.g. gfx1036 (7950X3D iGPU)
    assert!(!discrete.is_apu());
    assert!(apu.is_apu());
}

/// Minimal GPU `AmdNode` for pure (non-enumerate) logic tests.
fn gpu_node(cpu_cores_count: u32) -> AmdNode {
    AmdNode {
        node_id: 1,
        gpu_id: 5711,
        drm_render_minor: 128,
        gfx_target_version: 100_306,
        simd_count: 4,
        array_count: 2,
        simd_arrays_per_engine: 1,
        simd_per_cu: 2,
        max_waves_per_simd: 16,
        lds_size_in_kb: 64,
        wave_front_size: 32,
        num_xcc: 1,
        num_cp_queues: 8,
        max_slots_scratch_cu: 32,
        cpu_cores_count,
    }
}

fn tempfile_dir() -> PathBuf {
    // Build a fresh per-test tempdir so concurrent tests don't collide on
    // `SVOD_KFD_TOPOLOGY`. We don't pull `tempfile` for one test path.
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("svod-kfd-topo-{pid}-{nonce}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Reads the real `/sys/devices/virtual/kfd/kfd/topology/nodes/` if it
/// exists on this host. Asserts only that the parser doesn't choke on
/// real-world data; the field values depend on the local hardware.
#[test]
fn enumerate_real_host_topology_does_not_panic() {
    // Ensure no test-suite override leaks in.
    unsafe {
        std::env::remove_var("SVOD_KFD_TOPOLOGY");
    }
    let nodes = enumerate();
    for n in &nodes {
        assert!(n.gpu_id != 0, "enumerate must skip CPU nodes (got gpu_id 0 in {n:?})");
    }
    eprintln!("host has {} KFD GPU node(s): {nodes:?}", nodes.len());
}
