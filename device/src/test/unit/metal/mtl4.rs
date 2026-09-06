use std::time::Instant;

use test_case::test_case;

use super::metal_device_or_skip;
use crate::metal::compile::macos_product_version;
use crate::metal::mtl4::ticks_to_ns;

#[test_case(24, 24_000_000, 1_000; "24 MHz timebase, one microsecond")]
#[test_case(0, 24_000_000, 0; "zero")]
#[test_case(123_456_789, 1_000_000_000, 123_456_789; "nanosecond timebase is identity")]
#[test_case(1 << 40, 24_000_000, 45_812_984_490_666; "large tick counts do not overflow the intermediate")]
fn ticks_convert_exactly(ticks: u64, frequency: u64, expected: u64) {
    assert_eq!(ticks_to_ns(ticks, frequency), expected);
}

/// macOS 26 offers Metal 4 on every Metal device; older systems get `None`
/// and profiling stays on command-buffer stamps.
#[test]
fn profiler_availability_follows_the_os() {
    let Some(dev) = metal_device_or_skip() else { return };
    let major: u32 = macos_product_version()
        .and_then(|version| version.split('.').next()?.parse().ok())
        .expect("Darwin reports a product version");
    match dev.mtl4() {
        Some(profiler) => {
            assert!(major >= 26, "Metal 4 profiler on macOS {major}");
            assert!(profiler.timestamp_frequency() > 0);
        }
        None => assert!(major < 26, "macOS {major} should offer Metal 4"),
    }
}

/// The Metal 4 stamps of a captured chain add up to the chain's real GPU time:
/// each kernel's span is positive, they follow capture order, and their sum
/// agrees with the legacy command-buffer stamps of the same chain.
#[test]
fn metal4_stamps_match_command_buffer_timing() {
    let Some(chain) = super::graph::Chain::new() else { return };
    let Some(_) = chain.alloc.dev.mtl4() else {
        eprintln!("skipping: no Metal 4 on this OS");
        return;
    };
    let graph = chain.capture();
    let wall = Instant::now();
    let stamps: Vec<(u64, u64)> = graph
        .replay_profiled(&[], &[])
        .unwrap()
        .expect("stamped")
        .iter()
        .map(|handle| handle.timestamps_ns().expect("resolved"))
        .collect();
    let wall = wall.elapsed().as_nanos() as u64;
    assert_eq!(stamps.len(), 3);
    let mut previous_end = 0;
    for &(start, end) in &stamps {
        assert!(end > start, "{stamps:?}");
        assert!(start >= previous_end, "{stamps:?}");
        previous_end = end;
    }
    let total: u64 = stamps.iter().map(|(start, end)| end - start).sum();
    assert!(total < wall, "GPU spans {total} ns exceed the wall clock {wall} ns");
    let mut legacy = 0u64;
    for kernel in chain.kernels() {
        legacy += unsafe { kernel.program.execute_timed(&kernel.buffers, &[], kernel.global_size, kernel.local_size) }
            .unwrap()
            .expect("stamped")
            .as_nanos() as u64;
    }
    let ratio = total as f64 / legacy as f64;
    assert!((0.25..=4.0).contains(&ratio), "metal4 {total} ns vs command buffers {legacy} ns");
    assert_eq!(super::program::download(&chain.alloc, &chain.out, super::graph::N), chain.expected());
}
