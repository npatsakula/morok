//! Host-only CUPTI binding tests: the symbol table and the params ABI. The
//! struct layouts are also pinned by `const` asserts in the binding, so a
//! mismatch there is a compile error; these cover what the asserts cannot say
//! out loud, and run on hosts with no CUDA at all.

use crate::cuda::cupti::{self, CUptiResult};

/// Every bound entry point is a CUPTI export, named once.
#[test]
fn symbol_table_is_well_formed() {
    assert!(!cupti::SYMBOLS.is_empty());
    for (field, symbol) in cupti::SYMBOLS {
        assert!(symbol.starts_with("cupti"), "{field} binds non-CUPTI symbol {symbol}");
        assert!(!field.is_empty());
    }
    let mut symbols: Vec<&str> = cupti::SYMBOLS.iter().map(|(_, s)| *s).collect();
    symbols.sort_unstable();
    let bound = symbols.len();
    symbols.dedup();
    assert_eq!(symbols.len(), bound, "a CUPTI symbol is bound twice");
}

/// The whole range-profiling sequence lives in one library, so every call it
/// needs must be bound: CUDA 13 folded the PerfWorks host API into CUPTI and we
/// deliberately never bind `libnvperf_host.so`.
#[test]
fn the_full_capture_sequence_is_bound() {
    let bound: Vec<&str> = cupti::SYMBOLS.iter().map(|(_, s)| *s).collect();
    for required in [
        "cuptiProfilerInitialize",
        "cuptiDeviceGetChipName",
        "cuptiProfilerGetCounterAvailability",
        "cuptiProfilerHostInitialize",
        "cuptiProfilerHostConfigAddMetrics",
        "cuptiProfilerHostGetConfigImageSize",
        "cuptiProfilerHostGetConfigImage",
        "cuptiProfilerHostEvaluateToGpuValues",
        "cuptiRangeProfilerEnable",
        "cuptiRangeProfilerSetConfig",
        "cuptiRangeProfilerStart",
        "cuptiRangeProfilerStop",
        "cuptiRangeProfilerDecodeData",
        "cuptiRangeProfilerGetCounterDataSize",
        "cuptiRangeProfilerCounterDataImageInitialize",
        "cuptiRangeProfilerGetCounterDataInfo",
    ] {
        assert!(bound.contains(&required), "{required} is not bound");
    }
    for never in ["NVPW_InitializeHost", "NVPW_CUDA_MetricsContext_Create"] {
        assert!(!bound.contains(&never), "{never} must not be bound: CUPTI loads PerfWorks itself");
    }
}

/// A result code describes itself even with no CUPTI present, so a failure
/// during capability probing is always reportable.
#[test]
fn result_codes_describe_themselves() {
    let described = CUptiResult::INSUFFICIENT_PRIVILEGES.describe();
    assert!(!described.is_empty());
    // Without CUPTI loaded the fallback still names the numeric code.
    if cupti::api().is_none() {
        assert_eq!(described, "CUptiResult(35)");
    }
    assert_eq!(CUptiResult::SUCCESS, CUptiResult(0));
}

/// CUPTI accepts exactly the sizes a params struct has ever had, so a binding
/// built against newer headers steps back to the older size on an older
/// library, remembers the accepted one, and stops at any other answer.
#[test]
fn abi_ladder_steps_back_to_the_size_an_older_cupti_knows() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let chosen = AtomicUsize::new(0);
    let mut tried = Vec::new();
    let outcome = cupti::abi_ladder(&[56, 48], &chosen, "probe", |size| {
        tried.push(size);
        if size == 48 { CUptiResult::SUCCESS } else { CUptiResult::INVALID_PARAMETER }
    });
    assert!(outcome.is_ok());
    assert_eq!(tried, [56, 48]);
    assert_eq!(chosen.load(Ordering::Relaxed), 48);

    // The accepted size is tried alone from then on.
    tried.clear();
    cupti::abi_ladder(&[56, 48], &chosen, "probe", |size| {
        tried.push(size);
        CUptiResult::SUCCESS
    })
    .expect("remembered size");
    assert_eq!(tried, [48]);

    // A real answer at the newest size ends the ladder, even a failure.
    let chosen = AtomicUsize::new(0);
    tried.clear();
    let outcome = cupti::abi_ladder(&[56, 48], &chosen, "probe", |size| {
        tried.push(size);
        CUptiResult::INSUFFICIENT_PRIVILEGES
    });
    assert!(outcome.is_err());
    assert_eq!(tried, [56]);
    assert_eq!(chosen.load(Ordering::Relaxed), 56);

    // Rejected everywhere: the error names the call.
    let outcome = cupti::abi_ladder(&[56, 48], &AtomicUsize::new(0), "probe", |_| CUptiResult::INVALID_PARAMETER);
    assert!(outcome.is_err_and(|error| error.starts_with("probe")));
}
