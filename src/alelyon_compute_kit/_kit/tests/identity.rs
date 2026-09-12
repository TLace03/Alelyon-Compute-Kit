//! The device identity the handshake reports, on a real device: stable across
//! contexts on the same device, non-empty, and carrying the limits the safe
//! API checks against. Without a device every test prints UNMEASURED and
//! returns (`ACK_REQUIRE_DEVICE=1` turns that into a failure).

use alelyon_compute_kit::Context;

fn open_or_unmeasured(what: &str) -> Option<Context> {
    match Context::open() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
                panic!("{what}: ACK_REQUIRE_DEVICE is set and no device opened: {e}");
            }
            eprintln!("UNMEASURED here ({what}): {e}");
            None
        }
    }
}

#[test]
fn two_contexts_on_one_device_report_the_same_identity() {
    let Some(a) = open_or_unmeasured("identity") else {
        return;
    };
    let Some(b) = open_or_unmeasured("identity, second context") else {
        return;
    };
    assert_ne!(a.id(), b.id(), "context ids are per context");
    assert_eq!(a.report.device_uuid, b.report.device_uuid);
    assert_eq!(a.report.driver_uuid, b.report.driver_uuid);
    assert_eq!(a.report.identity_bytes(), b.report.identity_bytes());
    assert_ne!(
        a.report.device_uuid, [0u8; 16],
        "a device UUID is never all zero"
    );
    // vendor and device ids are not guaranteed nonzero for every implementation
    // (software devices); the API version and a name are
    assert_ne!(a.report.api_version_raw, 0);
    assert!(!a.report.device_name.is_empty());
}

#[test]
fn the_report_carries_the_limits_the_safe_api_checks_against() {
    let Some(ctx) = open_or_unmeasured("limits") else {
        return;
    };
    assert!(ctx.max_storage_buffer_bytes > 0);
    assert!(
        ctx.max_push_constant_bytes >= 128,
        "Vulkan guarantees at least 128 bytes of push constants"
    );
    assert!(ctx.max_workgroup_count.iter().all(|&c| c > 0));
    assert!(ctx.report.max_workgroup_invocations >= 128);
    assert!(ctx.report.max_workgroup_size.iter().all(|&c| c > 0));
    assert!(ctx.timestamp_valid_bits >= 36 && ctx.timestamp_valid_bits <= 64);
    assert!(ctx.report.subgroup_size.is_power_of_two());
}
