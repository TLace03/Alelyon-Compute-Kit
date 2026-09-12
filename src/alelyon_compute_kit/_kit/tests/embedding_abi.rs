//! embedding-f32/v2 at the C ABI (`ack_embedding`, ABI 9): the entry validates
//! the caller's host ids through `embedding_ops::EmbeddingPlan`, uploads the
//! plan's OWN index and offset images -- the gather's canonical id vector, or
//! the backward's CSR -- and dispatches. The host-only checks need no device;
//! the operations need one and print UNMEASURED without it unless
//! ACK_REQUIRE_DEVICE is set. Fixed-fixture evidence only.
use alelyon_compute_kit::embedding_ops::{EmbeddingOp, EmbeddingPlan};
use alelyon_compute_kit::ffi;
use std::ffi::{c_char, CStr};

/// A byte pattern no index or CSR image contains at these sizes, so a word
/// still holding it was not written by the entry's upload.
const POISON: i32 = 0x5A5A_5A5Au32 as i32;

fn last_error() -> String {
    let mut buf = vec![0 as c_char; 1024];
    let rc = unsafe { ffi::ack_last_error(buf.as_mut_ptr(), buf.len()) };
    assert!(rc >= 0, "ack_last_error returned {rc}");
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// The gather, on the host: row `ids[t]` of the table becomes row `t` of the
/// output. Exact -- it moves f32 words and computes nothing.
fn host_gather(table: &[f32], ids: &[i64], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(ids.len() * dim);
    for &id in ids {
        let start = id as usize * dim;
        out.extend_from_slice(&table[start..start + dim]);
    }
    out
}

/// The dense backward, on the host in f64: every token row is added into the
/// vocabulary row its id names.
fn host_backward(upstream: &[f32], ids: &[i64], vocab: usize, dim: usize) -> Vec<f32> {
    let mut acc = vec![0.0f64; vocab * dim];
    for (token, &id) in ids.iter().enumerate() {
        for column in 0..dim {
            acc[id as usize * dim + column] += f64::from(upstream[token * dim + column]);
        }
    }
    acc.into_iter().map(|v| v as f32).collect()
}

#[test]
fn a_malformed_request_is_refused_before_any_device_lookup() {
    let dev = std::ptr::null();
    let none = std::ptr::null();
    let ids = [0i64, 1, 2, 1];
    let call = |n: u32, vocab: u32, dim: u32, op: u32, v: &[i64]| unsafe {
        ffi::ack_embedding(
            dev,
            none,
            none,
            none,
            none,
            n,
            vocab,
            dim,
            op,
            v.as_ptr(),
            v.len(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    // a null id pointer with a nonzero length is refused first of all
    assert_eq!(
        unsafe {
            ffi::ack_embedding(
                dev,
                none,
                none,
                none,
                none,
                4,
                8,
                3,
                0,
                std::ptr::null(),
                4,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        },
        ffi::ACK_ERR_NULL
    );
    // Each case is called INSIDE the loop: an array literal would evaluate every
    // call first and leave ack_last_error holding only the last message.
    for (n, vocab, dim, op, v, name) in [
        (
            4u32,
            8u32,
            3u32,
            2u32,
            &ids[..],
            "embedding-operation-out-of-range",
        ),
        (0, 8, 3, 0, &ids[..], "embedding-dimension-out-of-range"),
        (4, 0, 3, 0, &ids[..], "embedding-dimension-out-of-range"),
        (4, 8, 0, 0, &ids[..], "embedding-dimension-out-of-range"),
        (5, 8, 3, 0, &ids[..], "embedding-index-shape-mismatch"),
        (
            4,
            8,
            3,
            0,
            &[0i64, 1, 8, 1][..],
            "embedding-index-out-of-range",
        ),
        (
            4,
            8,
            3,
            0,
            &[0i64, 1, -1, 1][..],
            "embedding-index-out-of-range",
        ),
    ] {
        assert_eq!(call(n, vocab, dim, op, v), ffi::ACK_ERR_SHAPE, "{name}");
        assert!(last_error().contains(name), "{name}: {}", last_error());
    }
    // with a well-formed request the null device handle is what refuses next
    assert_eq!(call(4, 8, 3, 0, &ids), ffi::ACK_ERR_NULL);
}

#[test]
fn the_embedding_operations_match_a_host_reference_and_the_upload_is_reported_exactly() {
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "embedding ABI: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (embedding ABI): {}", last_error());
        return;
    }
    let vocab = 7usize;
    let dim = 5usize;
    let n = 9usize;
    // id 2 appears four times and id 0 twice: the duplicate case is what the
    // backward's CSR exists for, so the fixture must contain it. Ids 1 and 4
    // appear NOWHERE, which is the unselected-row case below.
    let ids: Vec<i64> = vec![2, 0, 2, 5, 2, 0, 6, 2, 3];
    let table: Vec<f32> = (0..vocab * dim)
        .map(|i| ((i * 7919) % 53) as f32 / 8.0 - 3.0)
        .collect();
    let upstream: Vec<f32> = (0..n * dim)
        .map(|i| ((i * 104_729) % 37) as f32 / 16.0 - 1.0)
        .collect();

    let alloc = |elements: usize| {
        let b = ffi::ack_buffer_alloc(dev, (elements * 4) as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    };
    let upload_f32 = |buf, data: &[f32]| {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            unsafe { ffi::ack_upload(dev, buf, bytes.as_ptr(), bytes.len()) },
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    };
    let upload_i32 = |buf, data: &[i32]| {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            unsafe { ffi::ack_upload(dev, buf, bytes.as_ptr(), bytes.len()) },
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    };
    let download_f32 = |buf, elements: usize| {
        let mut raw = vec![0u8; elements * 4];
        assert_eq!(
            unsafe { ffi::ack_download(dev, buf, raw.as_mut_ptr(), raw.len()) },
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>()
    };
    let download_i32 = |buf, elements: usize| {
        let mut raw = vec![0u8; elements * 4];
        assert_eq!(
            unsafe { ffi::ack_download(dev, buf, raw.as_mut_ptr(), raw.len()) },
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        raw.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<i32>>()
    };

    // ---- the gather -------------------------------------------------------
    let gather_plan = EmbeddingPlan::new(
        n as u32,
        vocab as u32,
        dim as u32,
        EmbeddingOp::Gather,
        &ids,
    )
    .expect("gather plan");
    let index_words = gather_plan.index_bytes().len() / 4;
    let offset_words = gather_plan.offset_bytes().len() / 4;
    let b_table = alloc(vocab * dim);
    let b_index = alloc(index_words);
    let b_offset = alloc(offset_words);
    let b_out = alloc(n * dim);
    upload_f32(b_table, &table);
    // poison BOTH index images: the entry is what must write them, and a word
    // still holding the pattern afterwards was never written
    upload_i32(b_index, &vec![POISON; index_words]);
    upload_i32(b_offset, &vec![POISON; offset_words]);

    let mut uploaded = 0u64;
    let mut ms = 0.0f64;
    let rc = unsafe {
        ffi::ack_embedding(
            dev,
            b_table,
            b_index,
            b_offset,
            b_out,
            n as u32,
            vocab as u32,
            dim as u32,
            EmbeddingOp::Gather as u32,
            ids.as_ptr(),
            ids.len(),
            &mut uploaded,
            &mut ms,
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    assert_eq!(
        uploaded as usize,
        (index_words + offset_words) * 4,
        "the gather must report the bytes of both images it wrote"
    );
    // No poison survives inside what the entry REPORTED it wrote. Had it written
    // fewer bytes than it claimed, a poisoned word would remain inside the
    // reported extent, which is the failure a byte count alone cannot show.
    let words_written = uploaded as usize / 4;
    let seen: Vec<i32> = download_i32(b_index, index_words)
        .into_iter()
        .chain(download_i32(b_offset, offset_words))
        .take(words_written)
        .collect();
    assert!(
        !seen.contains(&POISON),
        "a word inside the reported upload was never written"
    );
    let got = download_f32(b_out, n * dim);
    let want = host_gather(&table, &ids, dim);
    assert_eq!(got, want, "a gather moves f32 words; it is exact");

    // ---- the dense backward ----------------------------------------------
    let back_plan = EmbeddingPlan::new(
        n as u32,
        vocab as u32,
        dim as u32,
        EmbeddingOp::Backward,
        &ids,
    )
    .expect("backward plan");
    let b_index2 = alloc(back_plan.index_bytes().len() / 4);
    let b_offset2 = alloc(back_plan.offset_bytes().len() / 4);
    let b_up = alloc(n * dim);
    let b_grad = alloc(vocab * dim);
    upload_f32(b_up, &upstream);
    // Fill the gradient buffer with a value no correct result contains BEFORE
    // dispatching. Without this the "unselected rows are zero" assertion below
    // is vacuous: a freshly allocated buffer may already read zero, so it would
    // pass whether the kernel wrote those rows or skipped them entirely.
    upload_f32(b_grad, &vec![-77.5f32; vocab * dim]);
    let mut uploaded2 = 0u64;
    let rc = unsafe {
        ffi::ack_embedding(
            dev,
            b_up,
            b_index2,
            b_offset2,
            b_grad,
            n as u32,
            vocab as u32,
            dim as u32,
            EmbeddingOp::Backward as u32,
            ids.as_ptr(),
            ids.len(),
            &mut uploaded2,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_OK, "{}", last_error());
    assert_eq!(
        uploaded2 as usize,
        back_plan.index_bytes().len() + back_plan.offset_bytes().len()
    );
    let grad = download_f32(b_grad, vocab * dim);
    let want_grad = host_backward(&upstream, &ids, vocab, dim);
    for (index, (&got, &want)) in grad.iter().zip(want_grad.iter()).enumerate() {
        assert!(
            (f64::from(got) - f64::from(want)).abs() <= 2.0e-5 + 2.0e-5 * f64::from(want).abs(),
            "row {} column {}: {got} vs {want}",
            index / dim,
            index % dim
        );
    }
    // A vocabulary row nothing selected must be written as ZERO rather than left
    // as it was. The CSR gives such a row an empty bucket, which is exactly the
    // case a kernel visiting only selected rows would skip -- leaving the
    // caller's allocation, and a gradient that is whatever was there before.
    // The buffer was filled with -77.5 above, so this fails if the kernel skips
    // an empty bucket rather than writing the zero it reports.
    for id in 0..vocab {
        if !ids.contains(&(id as i64)) {
            assert!(
                grad[id * dim..(id + 1) * dim].iter().all(|&v| v == 0.0),
                "unselected vocabulary row {id} must be written as zero"
            );
        }
    }

    // ---- what the entry refuses at the ABI --------------------------------
    // The index slots are ones the entry WRITES, so unlike a read-only operand
    // they are not "at least": Context::upload moves a whole buffer and refuses
    // a length mismatch, which would otherwise surface as the kit's opaque
    // "upload of N bytes into an M-byte buffer" rather than as a name a refusal
    // histogram can carry.
    let oversized = alloc(index_words + 3);
    let rc = unsafe {
        ffi::ack_embedding(
            dev,
            b_table,
            oversized,
            b_offset,
            b_out,
            n as u32,
            vocab as u32,
            dim as u32,
            EmbeddingOp::Gather as u32,
            ids.as_ptr(),
            ids.len(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(rc, ffi::ACK_ERR_SIZE);
    assert!(
        last_error().contains("embedding-indices-size-mismatch"),
        "{}",
        last_error()
    );
    // ---- output aliasing, and the check that speaks before it -------------
    //
    // The output may be none of the buffers the kernel reads, and the two index
    // images are read by it as surely as the table is. But the CAPACITY check
    // runs first, so an aliased buffer that is also too small is refused as
    // `-too-small` and never reaches the aliasing name. That was measured here,
    // not reasoned about: the first version of this test aliased the output onto
    // the table at vocab 7 and n 9 and got `embedding-output-too-small`, because
    // a gather's output is n*dim and the table is only vocab*dim.
    //
    // So each of the three operands gets a shape where the aliased buffer is
    // large enough for the output, which is what makes the aliasing check the
    // one that fires. Reading these as one loop would let a case that silently
    // took the capacity path look like a case the aliasing guard caught.
    let alias_case = |primary, indices, offsets, out, n: u32, vocab: u32, dim: u32, op: u32| unsafe {
        ffi::ack_embedding(
            dev,
            primary,
            indices,
            offsets,
            out,
            n,
            vocab,
            dim,
            op,
            ids.as_ptr(),
            ids.len(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    // the capacity check first, as the fact it is: an aliased output too small
    // for the result is refused by SIZE, and the aliasing is never reported
    let rc = alias_case(
        b_table,
        b_index,
        b_offset,
        b_table,
        n as u32,
        vocab as u32,
        dim as u32,
        EmbeddingOp::Gather as u32,
    );
    assert_eq!(rc, ffi::ACK_ERR_SIZE, "a too-small alias is a size refusal");
    assert!(
        last_error().contains("embedding-output-too-small"),
        "{}",
        last_error()
    );

    // table aliasing, at a vocabulary large enough to hold the gather's output
    let wide_vocab = 12usize;
    let b_wide = alloc(wide_vocab * dim);
    upload_f32(
        b_wide,
        &(0..wide_vocab * dim)
            .map(|i| i as f32)
            .collect::<Vec<f32>>(),
    );
    let rc = alias_case(
        b_wide,
        b_index,
        b_offset,
        b_wide,
        n as u32,
        wide_vocab as u32,
        dim as u32,
        EmbeddingOp::Gather as u32,
    );
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "output aliasing the table");
    assert!(
        last_error().contains("embedding-output-aliases-input"),
        "table: {}",
        last_error()
    );

    // index aliasing: at dim 1 a gather's output is n words, which is exactly
    // the size of its index image, so the index buffer can hold the output
    let b_narrow_out = alloc(n);
    let b_narrow_index = alloc(n);
    let b_narrow_offset = alloc(1);
    let rc = alias_case(
        b_table,
        b_narrow_index,
        b_narrow_offset,
        b_narrow_index,
        n as u32,
        vocab as u32,
        1,
        EmbeddingOp::Gather as u32,
    );
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "output aliasing the indices");
    assert!(
        last_error().contains("embedding-output-aliases-input"),
        "indices: {}",
        last_error()
    );

    // offset aliasing: at dim 1 the backward's output is vocab words and its CSR
    // is vocab+1, so the offset buffer is the one that is big enough
    let b_narrow_up = alloc(n);
    let rc = alias_case(
        b_narrow_up,
        b_narrow_index,
        b_narrow_offset,
        b_narrow_offset,
        n as u32,
        vocab as u32,
        1,
        EmbeddingOp::Backward as u32,
    );
    // the CSR image is vocab+1 words, so this needs the offset buffer at that
    // size rather than the gather's single dummy word
    assert_eq!(
        rc,
        ffi::ACK_ERR_SIZE,
        "the gather's one-word offset buffer is not the backward's CSR"
    );
    assert!(
        last_error().contains("embedding-offsets-size-mismatch"),
        "{}",
        last_error()
    );
    let b_csr = alloc(vocab + 1);
    let rc = alias_case(
        b_narrow_up,
        b_narrow_index,
        b_csr,
        b_csr,
        n as u32,
        vocab as u32,
        1,
        EmbeddingOp::Backward as u32,
    );
    assert_eq!(rc, ffi::ACK_ERR_SHAPE, "output aliasing the offsets");
    assert!(
        last_error().contains("embedding-output-aliases-input"),
        "offsets: {}",
        last_error()
    );
    let _ = b_narrow_out;

    for b in [
        b_table,
        b_index,
        b_offset,
        b_out,
        b_index2,
        b_offset2,
        b_up,
        b_grad,
        oversized,
        b_wide,
        b_narrow_out,
        b_narrow_index,
        b_narrow_offset,
        b_narrow_up,
        b_csr,
    ] {
        ffi::ack_buffer_free(dev, b);
    }
    ffi::ack_close(dev);
}

#[test]
fn the_xl_width_that_v2_refused_outright_now_runs_on_the_device() {
    // THE RAISE IS ONLY REAL IF THE KERNEL CARRIES IT. Bounds v2 capped MAX_DIM
    // at 1,024 while the 1B `xl` model's d_model is 2,048, so the embedding
    // family refused that model outright -- and a raised constant tested at the
    // OLD width would prove nothing at all. This runs the shape v2 refused.
    //
    // The device limits were measured before the bounds moved: max_workgroup_count
    // [4294967295, 65535, 65535] and max_storage_buffer_bytes 4,294,967,295 on the
    // RX 9070 XT, against 1,048,576 groups and a 268 MB table here.
    let dev = ffi::ack_open();
    if dev.is_null() {
        if std::env::var_os("ACK_REQUIRE_DEVICE").is_some() {
            panic!(
                "embedding xl width: ACK_REQUIRE_DEVICE is set and ack_open failed: {}",
                last_error()
            );
        }
        eprintln!("UNMEASURED here (embedding xl width): {}", last_error());
        return;
    }
    let vocab = 32_768usize; // the registered AKV vocabulary
    let dim = 2_048usize; // xl's d_model -- TWICE the v2 bound
    let n = 64usize; // few tokens: the width is what is under test
    assert!(
        dim > 1_024,
        "this row is pointless unless dim exceeds the v2 bound"
    );

    let ids: Vec<i64> = (0..n).map(|t| ((t * 7919) % vocab) as i64).collect();
    // a table this size is 268 MB, so it is built with a closed form rather than
    // held twice: row r column c is r + c/4096, exactly representable in f32
    let table: Vec<f32> = (0..vocab * dim)
        .map(|i| (i / dim) as f32 + ((i % dim) as f32) / 4096.0)
        .collect();

    let alloc = |elements: usize| {
        let b = ffi::ack_buffer_alloc(dev, (elements * 4) as u64);
        assert!(!b.is_null(), "{}", last_error());
        b
    };
    let upload_f32 = |buf, data: &[f32]| {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            unsafe { ffi::ack_upload(dev, buf, bytes.as_ptr(), bytes.len()) },
            ffi::ACK_OK,
            "{}",
            last_error()
        );
    };
    let download_f32 = |buf, elements: usize| {
        let mut raw = vec![0u8; elements * 4];
        assert_eq!(
            unsafe { ffi::ack_download(dev, buf, raw.as_mut_ptr(), raw.len()) },
            ffi::ACK_OK,
            "{}",
            last_error()
        );
        raw.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>()
    };

    let plan = EmbeddingPlan::new(
        n as u32,
        vocab as u32,
        dim as u32,
        EmbeddingOp::Gather,
        &ids,
    )
    .expect("the xl width must plan under v3 bounds");
    let b_table = alloc(vocab * dim);
    let b_index = alloc(plan.index_bytes().len() / 4);
    let b_offset = alloc(plan.offset_bytes().len() / 4);
    let b_out = alloc(n * dim);
    upload_f32(b_table, &table);

    let rc = unsafe {
        ffi::ack_embedding(
            dev,
            b_table,
            b_index,
            b_offset,
            b_out,
            n as u32,
            vocab as u32,
            dim as u32,
            EmbeddingOp::Gather as u32,
            ids.as_ptr(),
            ids.len(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    assert_eq!(
        rc,
        ffi::ACK_OK,
        "the xl width must dispatch: {}",
        last_error()
    );

    // exact, as everywhere in this family: a gather moves f32 words
    let got = download_f32(b_out, n * dim);
    for (t, &id) in ids.iter().enumerate() {
        let row = id as usize;
        for c in [0usize, 1, dim / 2, dim - 2, dim - 1] {
            assert_eq!(
                got[t * dim + c],
                table[row * dim + c],
                "token {t} (row {row}) column {c} at dim {dim}"
            );
        }
    }
    for b in [b_table, b_index, b_offset, b_out] {
        ffi::ack_buffer_free(dev, b);
    }
    ffi::ack_close(dev);
}
