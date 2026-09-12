//! The safe runtime's ownership contract on a real device (review finding
//! ACK-L0-02 and the push residual of ACK-L0-01): a resource from another
//! context is refused before any Vulkan call, a kernel refuses to dispatch
//! after a bound buffer was destroyed or with nothing bound, and push
//! constants must fit the declared range.
//!
//! Needs a Vulkan device. Without one every test here prints UNMEASURED and
//! returns, which is a skip with its reason, never a pass; set
//! `ACK_REQUIRE_DEVICE=1` to turn that into a failure on a machine that is
//! supposed to have the device.

use alelyon_compute_kit::{AckError, Context};

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

#[test]
fn a_buffer_of_another_context_is_refused_before_any_vulkan_call() {
    let Some(a) = open_or_unmeasured("foreign buffer") else {
        return;
    };
    let Some(b) = open_or_unmeasured("foreign buffer, second context") else {
        return;
    };
    assert_ne!(a.id(), b.id());
    let buf_a = a.buffer(1024).expect("buffer on a");
    assert_eq!(buf_a.context_id(), a.id());
    let data = vec![0u8; 1024];
    assert!(matches!(b.upload(&buf_a, &data), Err(AckError::Foreign(_))));
    assert!(matches!(b.download(&buf_a), Err(AckError::Foreign(_))));
    let kernel_b = b.kernel(CAST_F32_BF16, 2, 4).expect("cast kernel on b");
    let buf_b = b.buffer(512).expect("buffer on b");
    assert!(matches!(
        b.bind(&kernel_b, &[&buf_a, &buf_b]),
        Err(AckError::Foreign(_))
    ));
    // a kernel of b handed to a is refused the same way
    assert!(matches!(
        a.dispatch_timed(&kernel_b, &[0, 0, 0, 0], [1, 1, 1], 1),
        Err(AckError::Foreign(_))
    ));
    // the buffers still work where they belong
    a.upload(&buf_a, &data).expect("a uploads its own buffer");
    assert_eq!(
        a.download(&buf_a)
            .expect("a downloads its own buffer")
            .len(),
        1024
    );
    b.destroy_buffer(buf_b);
    b.destroy_kernel(kernel_b);
    a.destroy_buffer(buf_a);
}

#[test]
fn a_kernel_refuses_to_dispatch_after_a_bound_buffer_was_destroyed() {
    let Some(ctx) = open_or_unmeasured("bound after free") else {
        return;
    };
    let kernel = ctx.kernel(CAST_F32_BF16, 2, 4).expect("cast kernel");
    let src = ctx.buffer(4 * 256).expect("src");
    let dst = ctx.buffer(2 * 256).expect("dst");
    ctx.upload(&src, &vec![0u8; 4 * 256]).expect("upload");
    ctx.bind(&kernel, &[&src, &dst]).expect("bind");
    let n = 256u32.to_le_bytes();
    ctx.dispatch_timed(&kernel, &n, [1, 1, 1], 1)
        .expect("dispatch with both buffers live");
    ctx.destroy_buffer(src);
    assert!(
        matches!(
            ctx.dispatch_timed(&kernel, &n, [1, 1, 1], 1),
            Err(AckError::Freed(_))
        ),
        "a descriptor pointing at freed memory must never reach the queue"
    );
    ctx.destroy_buffer(dst);
    ctx.destroy_kernel(kernel);
}

#[test]
fn an_unbound_kernel_and_a_bad_push_range_are_refused_by_name() {
    let Some(ctx) = open_or_unmeasured("unbound and push") else {
        return;
    };
    let kernel = ctx.kernel(CAST_F32_BF16, 2, 4).expect("cast kernel");
    assert!(matches!(
        ctx.dispatch_timed(&kernel, &[0, 0, 0, 0], [1, 1, 1], 1),
        Err(AckError::Unsupported(_))
    ));
    let src = ctx.buffer(4 * 256).expect("src");
    let dst = ctx.buffer(2 * 256).expect("dst");
    ctx.bind(&kernel, &[&src, &dst]).expect("bind");
    // 8 bytes into a 4-byte range, and 3 bytes (not a multiple of 4)
    assert!(matches!(
        ctx.dispatch_timed(&kernel, &[0; 8], [1, 1, 1], 1),
        Err(AckError::Unsupported(_))
    ));
    assert!(matches!(
        ctx.dispatch_timed(&kernel, &[0; 3], [1, 1, 1], 1),
        Err(AckError::Unsupported(_))
    ));
    // a layout with a push range that is not a multiple of 4 is refused before any object exists
    assert!(matches!(
        ctx.kernel(CAST_F32_BF16, 2, 6),
        Err(AckError::Unsupported(_))
    ));
    ctx.destroy_buffer(src);
    ctx.destroy_buffer(dst);
    ctx.destroy_kernel(kernel);
}

#[test]
fn destroying_another_contexts_buffer_is_a_contract_violation_that_panics_before_vulkan() {
    let Some(a) = open_or_unmeasured("destroy foreign") else {
        return;
    };
    let Some(b) = open_or_unmeasured("destroy foreign, second context") else {
        return;
    };
    let buf_a = a.buffer(64).expect("buffer on a");
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.destroy_buffer(buf_a)));
    let payload = outcome
        .expect_err("destroying a foreign buffer must panic, not free memory on the wrong device");
    let text = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(text.contains("destroy_buffer"), "{text}");
    // `a` still works: nothing of its state was touched by the refused destroy
    let again = a
        .buffer(64)
        .expect("a allocates after the refused foreign destroy");
    a.destroy_buffer(again);
    // the buffer the panic unwound past is abandoned, not destroyed: `a` will
    // be dropped with that buffer and its memory alive (the validation layer
    // reports both at vkDestroyDevice), which is the caller's contract
    // violation, not a runtime leak; the count says exactly that
    // SUBTRACT WHAT THE POOL IS HOLDING, do not turn the pool off. The claim
    // is about the ABANDONED buffer -- the one the panic unwound past, which
    // nothing will ever destroy -- and `again` above was allocated and freed
    // normally, so with the pool on its pair is held rather than destroyed and
    // is live for a reason that is not a contract violation. `live_objects()`
    // counts both; `pooled_objects()` names the half that is the pool's, so
    // the difference is exactly the quantity this assertion has always been
    // about. It is 2 with the pool on and 2 with it off.
    assert_eq!(
        a.live_objects() - a.pooled_objects(),
        2,
        "the abandoned buffer and its memory"
    );
    assert_eq!(b.live_objects(), 0, "the refusing context created nothing");
}
