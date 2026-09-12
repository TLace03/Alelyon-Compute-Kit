//! Host-only optional-extension ABI refusals. These do not open a device;
//! real shader and buffer-lifetime execution is a separate required gate.
use alelyon_compute_kit::{ffi, Buffer};
use std::ffi::{c_char, CStr};

fn error() -> String {
    let mut buf = [0 as c_char; 512];
    assert!(unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) } >= 0);
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn block(words: [u32; 8]) -> [u8; 32] {
    let mut out = [0; 32];
    for (i, word) in words.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

unsafe fn call(plan: *const u8, len: usize) -> i32 {
    // Deliberately invalid non-null ids. Validation must never dereference
    // them, and malformed plans must refuse before registry lookup.
    let dev = usize::MAX as *const ffi::AckDevice;
    let b = usize::MAX as *const Buffer;
    unsafe { ffi::ack_vq(dev, b, b, b, b, b, b, plan, len) }
}

#[test]
fn optional_extension_does_not_relabel_existing_abi_layout() {
    assert_eq!(ffi::ack_vq_schema(), 1);
    assert_eq!(ffi::ACK_ABI_VERSION, 10);
    assert_eq!(ffi::VQ_PUSH_BYTES, 32);
}

#[test]
fn null_and_wrong_sized_blocks_refuse_before_any_handle_lookup() {
    assert_eq!(unsafe { call(std::ptr::null(), 32) }, ffi::ACK_ERR_NULL);
    assert!(error().contains("null plan"));
    let bytes = block([0, 1, 1, 1, 8, 16, 0, 0]);
    for len in [0, 1, 31, 33, usize::MAX] {
        assert_eq!(unsafe { call(bytes.as_ptr(), len) }, ffi::ACK_ERR_SIZE);
        assert!(error().contains("plan has"));
    }
}

#[test]
fn malformed_schema_refuses_by_name_before_any_device_lookup() {
    for (words, code, name) in [
        (
            [4, 1, 1, 1, 8, 16, 0, 0],
            ffi::ACK_ERR_SHAPE,
            "vq-operation-out-of-range",
        ),
        (
            [0, 1, 1, 1, 4, 16, 0, 0],
            ffi::ACK_ERR_SHAPE,
            "vq-unsupported-group",
        ),
        (
            [0, 1, 1, 1, 8, 15, 0, 0],
            ffi::ACK_ERR_SHAPE,
            "vq-codebook-entries-not-16",
        ),
        (
            [2, 1, 1, 1, 8, 16, 4, 0],
            ffi::ACK_ERR_SHAPE,
            "vq-unsupported-flags",
        ),
        (
            [0, 1, 1, 1, 8, 16, 0, 1],
            ffi::ACK_ERR_SHAPE,
            "vq-nonzero-reserved-word",
        ),
        (
            [0, 0, 1, 1, 8, 16, 0, 0],
            ffi::ACK_ERR_SHAPE,
            "vq-invalid-geometry",
        ),
        (
            [0, u32::MAX, 1, 1, 8, 16, 0, 0],
            ffi::ACK_ERR_SIZE,
            "vq-address-limit-exceeded",
        ),
        (
            [3, 1 << 28, 1, 1, 8, 16, 0, 0],
            ffi::ACK_ERR_SIZE,
            "vq-address-limit-exceeded",
        ),
    ] {
        let bytes = block(words);
        assert_eq!(unsafe { call(bytes.as_ptr(), bytes.len()) }, code);
        assert!(error().ends_with(name), "{}", error());
    }
}

#[test]
fn all_valid_operations_reach_closed_device_refusal_without_dereference() {
    for op in 0..4 {
        let bytes = block([op, 1, 1, 1, 8, 16, 0, 0]);
        assert_eq!(
            unsafe { call(bytes.as_ptr(), bytes.len()) },
            ffi::ACK_ERR_CLOSED
        );
        assert!(error().contains("is not open"));
    }
}

#[test]
fn excessive_scalar_matmul_work_refuses_before_device_lookup() {
    let bytes = block([2, 1024, 1024, 257, 8, 16, 0, 0]);
    assert_eq!(
        unsafe { call(bytes.as_ptr(), bytes.len()) },
        ffi::ACK_ERR_SIZE
    );
    assert!(error().ends_with("vq-matmul-work-limit"));
}
