//! Host-only vq4 admission; no device/model allocation or timing claims.
use alelyon_compute_kit::vq_ops::{
    VqOp, VqPlan, VqPlanError as E, MAX_DIM, MAX_VALUES, PUSH_BYTES,
};

fn vector(op: u32, n: u32, group: u32) -> VqPlan {
    VqPlan::from_words([op, n, 1, 1, group, 16, 0, 0]).unwrap()
}

#[test]
fn packed_cost_counts_padding_and_complete_codebooks() {
    for group in [8, 16, 32] {
        for (len, words) in [(1, 1), (group, 1), (group * 8, 1), (group * 8 + 1, 2)] {
            let encode = vector(0, len, group);
            let decode = vector(1, len, group);
            assert_eq!(
                encode.required_bytes(),
                [
                    u64::from(len) * 4,
                    u64::from(group) * 64,
                    0,
                    0,
                    words * 4,
                    4
                ]
            );
            assert_eq!(
                decode.required_bytes(),
                [
                    words * 4,
                    u64::from(group) * 64,
                    0,
                    0,
                    u64::from(len) * 4,
                    4
                ]
            );
            assert_eq!(encode.work_items(), words as u32);
            assert_eq!(decode.work_items(), len);
        }
    }
}

#[test]
fn adamw_is_four_packed_states_and_three_dense_scratch_arrays() {
    let p = vector(3, 129, 16);
    assert_eq!(p.op(), VqOp::AdamW);
    assert_eq!(p.required_bytes(), [32, 4096, 28, 0, 1548, 4]);
    assert_eq!(p.work_items(), 129);
    assert_eq!(p.dispatch_groups(), [3, 1, 1]);
    assert!(VqPlan::from_words([3, MAX_VALUES / 3, 1, 1, 8, 16, 0, 0]).is_ok());
    assert_eq!(
        VqPlan::from_words([3, MAX_VALUES / 3 + 1, 1, 1, 8, 16, 0, 0]),
        Err(E::AddressLimit)
    );
}

#[test]
fn matmul_transposes_preserve_storage_counts_and_only_change_wire_flags() {
    for flags in 0..4 {
        let p = VqPlan::from_words([2, 3, 5, 7, 8, 16, flags, 0]).unwrap();
        assert_eq!(p.required_bytes(), [4, 512, 4, 512, 60, 4]);
        assert_eq!(p.work_items(), 15);
        assert_eq!(p.dispatch_groups(), [1, 1, 1]);
        let block = p.push_constants();
        assert_eq!(block.len(), PUSH_BYTES);
        let expected = [2u32, 3, 5, 7, 8, 16, flags, 0];
        for (i, &word) in expected.iter().enumerate() {
            assert_eq!(&block[4 * i..4 * i + 4], &word.to_le_bytes());
        }
    }
}

#[test]
fn all_schema_words_are_closed() {
    let good = [0, 64, 1, 1, 8, 16, 0, 0];
    for (slot, value, error) in [
        (0, 4, E::Operation),
        (4, 0, E::Group),
        (4, 4, E::Group),
        (4, 64, E::Group),
        (5, 0, E::Entries),
        (5, 15, E::Entries),
        (5, 17, E::Entries),
        (6, 1, E::Flags),
        (7, 1, E::Reserved),
    ] {
        let mut words = good;
        words[slot] = value;
        assert_eq!(VqPlan::from_words(words), Err(error));
    }
    assert_eq!(VqPlan::from_words([2, 1, 1, 1, 8, 16, 4, 0]), Err(E::Flags));
    for op in [1, 3] {
        assert_eq!(
            VqPlan::from_words([op, 1, 1, 1, 8, 16, 1, 0]),
            Err(E::Flags)
        );
    }
}

#[test]
fn zero_mismatched_or_overflowing_geometry_refuses_without_allocating() {
    for op in 0..4 {
        for axis in 1..4 {
            let mut words = [op, 1, 1, 1, 8, 16, 0, 0];
            words[axis] = 0;
            assert_eq!(VqPlan::from_words(words), Err(E::Geometry));
        }
    }
    for op in [0, 1, 3] {
        for axis in [2, 3] {
            let mut words = [op, 1, 1, 1, 8, 16, 0, 0];
            words[axis] = 2;
            assert_eq!(VqPlan::from_words(words), Err(E::Geometry));
        }
    }
    assert_eq!(
        VqPlan::from_words([0, u32::MAX, 1, 1, 8, 16, 0, 0]),
        Err(E::AddressLimit)
    );
    assert_eq!(
        VqPlan::from_words([1, MAX_VALUES + 1, 1, 1, 8, 16, 0, 0]),
        Err(E::AddressLimit)
    );
    assert_eq!(
        VqPlan::from_words([2, MAX_DIM + 1, 1, 1, 8, 16, 0, 0]),
        Err(E::Geometry)
    );
    for words in [
        [2, MAX_DIM, MAX_DIM, 1, 8, 16, 0, 0],
        [2, MAX_DIM, 1, MAX_DIM, 8, 16, 0, 0],
        [2, 1, MAX_DIM, MAX_DIM, 8, 16, 0, 0],
    ] {
        assert_eq!(VqPlan::from_words(words), Err(E::AddressLimit));
    }
}

#[test]
fn each_read_and_write_capacity_is_checked_in_bytes() {
    for op in 0..4 {
        let p = if op == 2 {
            VqPlan::from_words([2, 11, 9, 7, 16, 16, 0, 0]).unwrap()
        } else {
            vector(op, 129, 16)
        };
        let capacities = p.required_bytes();
        let ids = [1, 2, 3, 4, 5, 6];
        assert_eq!(p.validate_buffers(capacities, ids), Ok(()));
        for slot in 0..6 {
            if capacities[slot] == 0 {
                continue;
            }
            let mut short = capacities;
            short[slot] -= 1;
            assert_eq!(p.validate_buffers(short, ids), Err(E::Capacity(slot)));
        }
    }
}

#[test]
fn every_output_alias_including_unread_inputs_refuses() {
    let p = vector(0, 64, 8);
    let capacities = [4096; 6];
    for input in 0..4 {
        let mut ids = [1, 2, 3, 4, 5, 6];
        ids[4] = ids[input];
        assert_eq!(p.validate_buffers(capacities, ids), Err(E::OutputAlias));
    }
    for operand in 0..5 {
        let mut ids = [1, 2, 3, 4, 5, 6];
        ids[5] = ids[operand];
        assert_eq!(p.validate_buffers(capacities, ids), Err(E::StatusAlias));
    }
    assert_eq!(p.validate_buffers(capacities, [1, 1, 1, 1, 2, 3]), Ok(()));
    for slot in 0..6 {
        let mut ids = [1, 2, 3, 4, 5, 6];
        ids[slot] = 0;
        assert_eq!(p.validate_buffers(capacities, ids), Err(E::NullIdentity));
    }
}

#[test]
fn maximum_tensor_uses_bounded_two_dimensional_grid() {
    let p = vector(1, MAX_VALUES, 32);
    let [x, y, z] = p.dispatch_groups();
    assert_eq!(x, 65535);
    assert_eq!(y, 65);
    assert_eq!(z, 1);
    assert!(u64::from(x) * u64::from(y) * 64 >= u64::from(MAX_VALUES));
    assert!(u64::from(x) * u64::from(y) * 64 < u64::from(u32::MAX));
    assert_eq!(p.validate_dispatch([65535; 3]), Ok(()));
    for axis in 0..3 {
        let mut limits = [x, y, z];
        limits[axis] -= 1;
        assert_eq!(p.validate_dispatch(limits), Err(E::Dispatch));
    }
}

#[test]
fn every_refusal_is_named_and_slot_specific() {
    for error in [
        E::Operation,
        E::Group,
        E::Entries,
        E::Flags,
        E::Reserved,
        E::Geometry,
        E::AddressLimit,
        E::WorkLimit,
        E::NullIdentity,
        E::OutputAlias,
        E::StatusAlias,
        E::Dispatch,
    ] {
        assert!(error.to_string().starts_with("vq-"));
    }
    for slot in 0..6 {
        assert_eq!(
            E::Capacity(slot).to_string(),
            format!("vq-binding-{slot}-too-small")
        );
    }
}

#[test]
fn matmul_work_is_bounded_even_when_every_buffer_extent_fits() {
    assert!(VqPlan::from_words([2, 1024, 1024, 256, 8, 16, 0, 0]).is_ok());
    assert_eq!(
        VqPlan::from_words([2, 1024, 1024, 257, 8, 16, 0, 0]),
        Err(E::WorkLimit)
    );
    assert_eq!(
        VqPlan::from_words([2, 4096, 4096, 4096, 8, 16, 0, 0]),
        Err(E::WorkLimit)
    );
    assert_eq!(E::WorkLimit.to_string(), "vq-matmul-work-limit");
}
