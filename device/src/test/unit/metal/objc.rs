use std::mem::{offset_of, size_of};

use crate::metal::objc::{BlockDescriptor, BlockLiteral, MTLSize, NSRange, ObjcBool, objc};

/// clang's Block-ABI-Apple layout, which `MTLCodeGenServiceBuildRequest`
/// dereferences to reach `invoke` (tinygrad's `&fn - 0x10` hack pins offset 16).
#[test]
fn block_literal_matches_clang_abi() {
    type Block = BlockLiteral<*const ()>;
    assert_eq!(size_of::<Block>(), 40);
    assert_eq!(offset_of!(Block, isa), 0);
    assert_eq!(offset_of!(Block, flags), 8);
    assert_eq!(offset_of!(Block, reserved), 12);
    assert_eq!(offset_of!(Block, invoke), 16);
    assert_eq!(offset_of!(Block, descriptor), 24);
    assert_eq!(offset_of!(Block, context), 32);
    assert_eq!(size_of::<BlockDescriptor>(), 16);
    let descriptor = Block::descriptor();
    assert_eq!((descriptor.reserved, descriptor.size), (0, 40));
}

#[test]
fn mtl_size_is_three_nsuintegers() {
    assert_eq!(size_of::<MTLSize>(), 24);
    assert_eq!(offset_of!(MTLSize, width), 0);
    assert_eq!(offset_of!(MTLSize, height), 8);
    assert_eq!(offset_of!(MTLSize, depth), 16);
    assert_eq!(MTLSize::from([2, 3, 4]), MTLSize { width: 2, height: 3, depth: 4 });
}

#[test]
fn ns_range_is_two_nsuintegers() {
    assert_eq!(size_of::<NSRange>(), 16);
    assert_eq!(offset_of!(NSRange, location), 0);
    assert_eq!(offset_of!(NSRange, length), 8);
}

#[test]
fn objc_bool_is_signed_char() {
    assert_eq!(size_of::<ObjcBool>(), 1);
}

/// Off Darwin the runtime is unavailable with a typed error naming the missing
/// library; on Darwin loading succeeds and `has_devices` may still be false
/// (headless VM), but never the other way around.
#[test]
fn runtime_availability_is_consistent() {
    match objc() {
        Ok(_) => {}
        Err(error) => {
            assert!(matches!(error, crate::Error::DeviceUnavailable { .. }), "{error:?}");
            assert!(format!("{error}").contains("cannot load"), "{error}");
            assert!(!crate::metal::has_devices());
        }
    }
}
