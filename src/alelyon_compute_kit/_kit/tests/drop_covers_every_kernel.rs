//! Every kernel family the device state holds is destroyed when the device is
//! dropped (2026-09-30).
//!
//! `Kernel` holds raw Vulkan handles (a pipeline, its layout, a descriptor set
//! layout and a pool) and has no `Drop` of its own: `Inner`'s `Drop` destroys
//! each family by name. A family added to `Inner` and left out of that list
//! survives every `ack_close`, and nothing reports it: `ack_live_objects` can
//! only be read while the device is open. Five families were missed that way
//! (intwise, intwise_f64, radix, swiglu_scaled, nf4_decode) until this test.
//!
//! So this test reads `src/ffi.rs` and holds `Drop` to `Inner`'s fields: every
//! field whose type holds a `Kernel` must be reached in `Drop` as `.field`,
//! whitespace ignored (a method chain may break across lines), the name ending
//! at a non-identifier character (`.intwise` is not satisfied by
//! `.intwise_f64`). It is a structural check: that each destroy runs is the
//! leak tests' part (tests/leaks.rs counts the objects a context creates).
use std::fs;

/// The text from `start` to the first line that is exactly `}` after it.
fn item_body<'a>(src: &'a str, start: &str) -> &'a str {
    let at = src
        .find(start)
        .unwrap_or_else(|| panic!("`{start}` not found in src/ffi.rs"));
    let rest = &src[at..];
    let end = rest
        .find("\n}\n")
        .unwrap_or_else(|| panic!("`{start}` has no closing line"));
    &rest[..end]
}

#[test]
fn every_kernel_field_of_the_device_state_is_destroyed_by_its_drop() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/ffi.rs");
    let src = fs::read_to_string(path)
        .expect("src/ffi.rs")
        .replace("\r\n", "\n");
    let fields: Vec<&str> = item_body(&src, "struct Inner {")
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.trim();
            if line.starts_with("//") {
                return None;
            }
            let (name, ty) = line.split_once(':')?;
            let name = name.trim();
            let is_ident =
                !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            (is_ident && ty.contains("Kernel")).then_some(name)
        })
        .collect();
    assert!(
        fields.len() >= 30,
        "the parse found {} kernel fields: it no longer reads `struct Inner` ({fields:?})",
        fields.len()
    );
    let body: String = item_body(&src, "impl Drop for Inner {")
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let reached = |name: &str| {
        let needle = format!(".{name}");
        body.match_indices(&needle).any(|(i, _)| {
            body[i + needle.len()..]
                .chars()
                .next()
                .is_some_and(|c| !(c.is_ascii_alphanumeric() || c == '_'))
        })
    };
    let missing: Vec<&str> = fields.iter().copied().filter(|f| !reached(f)).collect();
    assert!(
        missing.is_empty(),
        "kernel families held by the device and never destroyed at close: {missing:?}"
    );
}
