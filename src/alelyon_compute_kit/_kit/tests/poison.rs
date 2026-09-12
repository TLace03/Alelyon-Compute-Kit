//! ACK-L0-03 (review finding 520d51eb): after a submission has succeeded, a
//! failed fence wait may leave the GPU still reading the fence and the command
//! buffer. Only a successful `device_wait_idle` or an established device loss
//! makes destroying them safe; on any other idle error the context must leave
//! them allocated, mark itself poisoned, and refuse every later call by name.
//!
//! The faults are injected AFTER the real wait and the real idle have run, so
//! the GPU work has genuinely completed while the code path under test sees
//! the injected result. That makes the tests safe on a healthy card; it does
//! not make them an observation of a real fault (the device here is healthy
//! on every path, so what a really failed idle leaves behind is UNMEASURED).
//! Needs a device and the `fault-injection` feature; without a device every
//! test prints UNMEASURED and returns, unless ACK_REQUIRE_DEVICE is set.
#![cfg(feature = "fault-injection")]

use alelyon_compute_kit::ffi;
use alelyon_compute_kit::{AckError, Context, Fault};
use ash::vk;
use std::ffi::{c_char, CStr};

const CAST_F32_BF16: &[u8] = include_bytes!("../kernels/cast_f32_bf16.spv");

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

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

#[test]
fn a_failed_wait_and_a_failed_idle_poison_the_context_and_destroy_nothing() {
    let Some(ctx) = open_or_unmeasured("poison") else {
        return;
    };
    let buf = ctx.buffer(4096).expect("buffer");
    ctx.upload(&buf, &vec![1u8; 4096])
        .expect("a healthy upload");
    let kernel = ctx.kernel(CAST_F32_BF16, 2, 4).expect("a healthy kernel");
    let torn_down_before = ctx.submissions_torn_down();
    assert_eq!(ctx.leaked_on_poison(), 0, "a healthy context leaks nothing");
    ctx.inject_fault(Fault::WaitFails(vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    ctx.inject_fault(Fault::IdleFails(vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    let err = ctx.download(&buf).expect_err("the download must fail");
    assert!(matches!(err, AckError::Poisoned(_)), "{err}");
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_down_before,
        "with the device state unknown, the fence and command buffer stay allocated"
    );
    assert!(ctx.is_poisoned());
    assert_eq!(
        ctx.leaked_on_poison(),
        4,
        "the fence, the command buffer, and the download's staging buffer and memory stay allocated"
    );
    assert_eq!(
        ctx.live_objects(),
        10,
        "the two instruments agree: the buffer (2), the kernel (4) and the four leaked objects are all live"
    );
    // every later call refuses by name, before any Vulkan call
    assert!(matches!(ctx.buffer(64), Err(AckError::Poisoned(_))));
    assert!(matches!(
        ctx.upload(&buf, &vec![0u8; 4096]),
        Err(AckError::Poisoned(_))
    ));
    assert!(matches!(ctx.download(&buf), Err(AckError::Poisoned(_))));
    assert!(matches!(
        ctx.kernel(CAST_F32_BF16, 2, 4),
        Err(AckError::Poisoned(_))
    ));
    assert!(matches!(
        ctx.bind(&kernel, &[&buf, &buf]),
        Err(AckError::Poisoned(_))
    ));
    assert!(matches!(
        ctx.dispatch_timed(&kernel, &[0u8; 4], [1, 1, 1], 1),
        Err(AckError::Poisoned(_))
    ));
    // destroying leaks on purpose: the buffer and its memory, then the kernel's four objects
    ctx.destroy_buffer(buf);
    assert_eq!(ctx.leaked_on_poison(), 6);
    ctx.destroy_kernel(kernel);
    assert_eq!(ctx.leaked_on_poison(), 10);
    assert_eq!(
        ctx.live_objects(),
        ctx.leaked_on_poison() as i64,
        "nothing else is held: every live object is a leaked one"
    );
    // dropping the poisoned context leaves its pool, device, instance and loader
    // handle allocated; that is unobservable here without validation layers
    // (UNMEASURED). A reopen beside the leaked device works on this healthy
    // card, which is all this measures about "close and reopen".
    drop(ctx);
    let again = Context::open().expect("a reopen beside a leaked device");
    let b = again.buffer(64).expect("buffer on the reopened context");
    again
        .upload(&b, &[9u8; 64])
        .expect("upload on the reopened context");
    assert_eq!(again.download(&b).expect("download"), vec![9u8; 64]);
    again.destroy_buffer(b);
}

#[test]
fn an_established_device_loss_permits_teardown_but_poisons_everything_after() {
    let Some(ctx) = open_or_unmeasured("device loss") else {
        return;
    };
    let buf = ctx.buffer(1024).expect("buffer");
    let torn_down_before = ctx.submissions_torn_down();
    ctx.inject_fault(Fault::WaitFails(vk::Result::ERROR_DEVICE_LOST));
    ctx.inject_fault(Fault::IdleFails(vk::Result::ERROR_DEVICE_LOST));
    let err = ctx
        .upload(&buf, &vec![7u8; 1024])
        .expect_err("the upload must fail");
    assert!(
        matches!(err, AckError::Vulkan(vk::Result::ERROR_DEVICE_LOST)),
        "{err}"
    );
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_down_before + 1,
        "a lost device drops its work: teardown is permitted"
    );
    assert!(ctx.is_poisoned());
    assert_eq!(
        ctx.leaked_on_poison(),
        0,
        "a lost device permits releasing the staging pair too"
    );
    assert!(matches!(ctx.download(&buf), Err(AckError::Poisoned(_))));
    ctx.destroy_buffer(buf);
    assert_eq!(
        ctx.leaked_on_poison(),
        0,
        "a lost device permits destroying the buffer"
    );
}

#[test]
fn a_wait_that_itself_reports_the_device_lost_is_an_established_loss() {
    let Some(ctx) = open_or_unmeasured("wait reports loss") else {
        return;
    };
    let buf = ctx.buffer(1024).expect("buffer");
    let torn_down_before = ctx.submissions_torn_down();
    // only the wait reports the loss; the idle (really run, healthy card) says Ok
    ctx.inject_fault(Fault::WaitFails(vk::Result::ERROR_DEVICE_LOST));
    let err = ctx
        .upload(&buf, &vec![3u8; 1024])
        .expect_err("the upload must fail");
    assert!(
        matches!(err, AckError::Vulkan(vk::Result::ERROR_DEVICE_LOST)),
        "{err}"
    );
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_down_before + 1,
        "an established loss permits teardown"
    );
    assert!(
        ctx.is_poisoned(),
        "a wait that reports the device lost is an established loss whatever the idle says afterwards"
    );
    assert!(matches!(ctx.download(&buf), Err(AckError::Poisoned(_))));
    ctx.destroy_buffer(buf);
}

#[test]
fn a_failed_wait_with_a_successful_idle_is_an_error_but_not_a_poison() {
    let Some(ctx) = open_or_unmeasured("recoverable wait") else {
        return;
    };
    let buf = ctx.buffer(1024).expect("buffer");
    let torn_down_before = ctx.submissions_torn_down();
    ctx.inject_fault(Fault::WaitFails(vk::Result::ERROR_OUT_OF_HOST_MEMORY));
    let err = ctx
        .upload(&buf, &vec![3u8; 1024])
        .expect_err("the upload must fail");
    assert!(
        matches!(err, AckError::Vulkan(vk::Result::ERROR_OUT_OF_HOST_MEMORY)),
        "{err}"
    );
    assert_eq!(
        ctx.submissions_torn_down(),
        torn_down_before + 1,
        "quiesced: the fence and command buffer are released"
    );
    assert!(!ctx.is_poisoned());
    assert_eq!(
        ctx.leaked_on_poison(),
        0,
        "quiesced: the upload's staging pair was released too"
    );
    // the context works again, and the second upload must be visible as such:
    // the first one really landed (the real wait succeeded before the injected
    // failure), so a different byte is what separates the two
    ctx.upload(&buf, &vec![5u8; 1024])
        .expect("a later upload succeeds");
    let back = ctx.download(&buf).expect("download");
    assert!(
        back.iter().all(|&x| x == 5),
        "the later upload's bytes, not the first one's"
    );
    ctx.destroy_buffer(buf);
}

#[test]
fn the_c_abi_reports_a_poison_as_minus_ten_and_a_free_on_it_as_a_leak() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "C ABI poison: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (C ABI poison): {}", last_error());
        return;
    }
    let buf = ffi::ack_buffer_alloc(dev, 1024);
    assert!(!buf.is_null(), "{}", last_error());
    let data = vec![1u8; 1024];
    let rc = unsafe { ffi::ack_upload(dev, buf, data.as_ptr(), data.len()) };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    let oom = vk::Result::ERROR_OUT_OF_HOST_MEMORY.as_raw();
    assert_eq!(
        ffi::ack_inject_fault(dev, 0, oom),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    assert_eq!(
        ffi::ack_inject_fault(dev, 1, oom),
        ffi::ACK_OK,
        "{}",
        last_error()
    );
    let rc = unsafe { ffi::ack_upload(dev, buf, data.as_ptr(), data.len()) };
    assert_eq!(rc, ffi::ACK_ERR_POISONED, "{}", last_error());
    assert!(
        last_error().contains("poisoned context"),
        "{}",
        last_error()
    );
    let mut out = vec![0u8; 1024];
    let rc = unsafe { ffi::ack_download(dev, buf, out.as_mut_ptr(), out.len()) };
    assert_eq!(rc, ffi::ACK_ERR_POISONED, "{}", last_error());
    // a free consumes the handle and says the buffer was leaked, not freed
    assert_eq!(ffi::ack_buffer_free(dev, buf), ffi::ACK_ERR_POISONED);
    assert!(last_error().contains("leaked"), "{}", last_error());
    assert_eq!(
        ffi::ack_buffer_free(dev, buf),
        ffi::ACK_ERR_FREED,
        "the handle is gone either way"
    );
    // close succeeds on a poisoned device: the registry entry goes, the Vulkan objects stay
    assert_eq!(ffi::ack_close(dev), ffi::ACK_OK, "{}", last_error());
}
