use snafu::ResultExt;

use crate::error::{Error, ErrorKind, NoBufferSnafu, Result, UOpSnafu};

/// `Result<T>` is threaded through every tensor builder, so the `Err` payload
/// must stay pointer-sized regardless of how wide `ErrorKind` grows.
#[test]
fn error_is_pointer_sized() {
    assert!(size_of::<Error>() <= 32, "Error grew to {} bytes", size_of::<Error>());
    assert_eq!(size_of::<Error>(), size_of::<Box<ErrorKind>>());
    assert_eq!(size_of::<Result<()>>(), size_of::<Error>(), "Result<()> should be niche-packed into Error");
}

#[test]
fn error_forwards_display_debug_and_source() {
    let inner = svod_ir::Error::VoidTypeInOp;
    let expected_source = inner.to_string();
    let error: Error = Err::<(), _>(inner).context(UOpSnafu).unwrap_err().into();

    assert!(error.to_string().contains(&expected_source));
    assert!(format!("{error:?}").starts_with("UOp"), "Debug forwards to the kind: {error:?}");
    assert_eq!(std::error::Error::source(&error).map(ToString::to_string), Some(expected_source));
}

#[test]
fn error_exposes_its_kind() {
    let error: Error = NoBufferSnafu.build().into();
    assert!(matches!(error.kind(), ErrorKind::NoBuffer));
    assert!(matches!(error.into_kind(), ErrorKind::NoBuffer));
}
