"""A fitted codebook must return K entries, all distinct, and all reachable by its own encoder.

WHY THIS EXISTS. An entry no input can select is paid for and never used: the format bills
log2(ENTRIES) bits an index while the effective codebook is smaller. In a Lloyd fit that happens
when a cluster goes empty and the update is guarded by `if len(selected)`, which leaves that entry
holding a stale duplicate for ever — and `encode`'s lowest-index tie-break then guarantees the
duplicate is never selected again, so the cluster can never refill.

Dispersed data hides it: jitter keeps every cluster populated, and a run on gaussian or
row-normalised blocks comes back clean. The cases below therefore use EXACT DUPLICATES WITH UNEVEN
OCCUPANCY, which is what makes the `linspace` seeding pick the same atom twice.

MEASURED against this module before the fix, as distinct/reachable out of ENTRIES:

    occupancy                 published   random reseed   farthest point
    even (64 each)                16/16           16/16            16/16
    8 common + 8 rare             12/12           10/10            16/16
    skewed geometric                9/9           10/10            16/16
    one dominant, 15 rare           4/4             4/4            16/16

Two things follow, and the second is the reason the fix is what it is. The defect costs up to
twelve of sixteen entries. And reseeding an empty cluster from a RANDOM data point does not fix it
— a random draw can land on an atom another entry already covers, and above it was no better than
leaving the defect alone. Farthest-point reseeding recovers every entry in every non-degenerate
case and needs no generator, so it also keeps this function's "bounded deterministic calibration"
promise literally rather than approximately.

The occupancy patterns and the three-rule comparison were contributed by another session working in
`research/lowbit_weight_coding`, which found the same weakness in its own codec when it ran them.

THE DEGENERATE EXCEPTION is real and is asserted rather than hidden: if every input point is
identical there is nothing for a second entry to represent, and any rule collapses to one distinct
centre. A fix is not expected to invent variety the data does not contain.
"""
import numpy as np
import pytest

from alelyon_compute_kit._kit.python.vq import (ENTRIES, VQAdamW, VQArray, encode,
                                              fit_codebook)

DIMS, SEED = 16, 1

OCCUPANCY = {
    "even": [64] * ENTRIES,
    "eight common, eight rare": [120] * 8 + [8] * 8,
    "skewed geometric": [max(1, int(512 * 0.55 ** i)) for i in range(ENTRIES)],
    "one dominant, fifteen rare": [900] + [4] * (ENTRIES - 1),
}


def fit(sample):
    """`fit_codebook` takes a FLATTENED array and reshapes internally."""
    return fit_codebook(sample.reshape(-1), group=DIMS)


def indices(sample, book):
    """The entry each vector actually selects.

    `encode` returns a VQArray whose `codes` pack eight four-bit indices per
    uint32 word, so the index stream has to be unpacked — reading `codes`
    directly would compare packed words and silently answer a different
    question.
    """
    codes = encode(sample, book, group=DIMS).codes
    pos = np.arange(len(sample))
    return (codes[pos // 8] >> ((pos % 8) * 4).astype(np.uint32)) & 15


def sample_of(counts):
    """Exactly ENTRIES distinct vectors, repeated with the given occupancy.

    A perfect fit is reachable by construction: one entry per atom, zero
    distortion. Anything less is the fit losing entries, not the data.
    """
    atoms = np.random.default_rng(SEED).standard_normal((ENTRIES, DIMS))
    return np.repeat(atoms, counts, axis=0).astype(np.float32)


@pytest.mark.parametrize("name", list(OCCUPANCY))
def test_every_entry_is_distinct_and_reachable(name):
    sample = sample_of(OCCUPANCY[name])
    book = fit(sample)
    assert len(book) == ENTRIES, "an entry was dropped"
    assert np.isfinite(book).all()
    assert len(np.unique(book.round(6), axis=0)) == ENTRIES, (
        f"{name}: duplicate entries are paid for and can never be selected")
    reachable = len(np.unique(indices(sample, book)))
    assert reachable == ENTRIES, (
        f"{name}: only {reachable} of {ENTRIES} entries are reachable, while the "
        f"index still costs {np.log2(ENTRIES):.0f} bits")


def test_a_perfect_fit_is_actually_available_in_these_cases():
    """Positive control. If the data could not be fitted exactly, the test above
    would be asserting something impossible and its failure would mean nothing."""
    sample = sample_of(OCCUPANCY["one dominant, fifteen rare"])
    atoms = np.unique(sample, axis=0).astype(np.float32)
    assert len(atoms) == ENTRIES
    residual = sample - atoms[indices(sample, atoms)]
    assert float((residual ** 2).sum()) == pytest.approx(0.0, abs=1e-8)


def test_identical_input_collapses_and_that_is_correct():
    """The documented exception, asserted so the guarantee above is not read as
    stronger than it is."""
    book = fit(np.ones((512, DIMS), dtype=np.float32))
    assert len(book) == ENTRIES
    assert len(np.unique(book.round(6), axis=0)) == 1


def test_the_suite_can_see_the_defect_it_is_guarding_against():
    """The other positive control: the pre-fix loop must FAIL the property, or
    these cases do not exercise the defect and the suite above is vacuous."""
    def unfixed(sample, iterations=4):
        book = sample[np.linspace(0, len(sample) - 1, ENTRIES, dtype=np.int64)].copy()
        for _ in range(iterations):
            idx = indices(sample, book)
            for entry in range(ENTRIES):
                selected = sample[idx == entry]
                if len(selected):
                    book[entry] = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
        return book

    sample = sample_of(OCCUPANCY["one dominant, fifteen rare"])
    assert len(np.unique(unfixed(sample).round(6), axis=0)) < ENTRIES, (
        "the unfixed loop must lose entries here, otherwise these cases do not "
        "exercise the defect this file guards against")


# ── the other half of the same finding ───────────────────────────────────────
#
# `docs/audits/2026-09-14-ack-lowbit-commit-review.md` finding 1 (private repo)
# named TWO consequences of the empty-cluster defect: a codebook that loses
# capacity, which everything above covers, and weights that "move without a
# gradient", which none of it does -- those cases fit once, and this one is
# about what `VQAdamW.step` does on every step.
#
# The mechanism: `step` refits the codebook from scratch each time. If that
# refit collapses relative to the book the weights are currently encoded with,
# re-encoding lands them on different centres and the values move even though
# the gradient was exactly zero. MEASURED on this construction, changed values
# out of 4,112 with max|dw|: pre-fix refit 912 and 2.68; the shipped
# farthest-point refit 0 and 0. So the movement was a CONSEQUENCE of the
# collapse rather than a second defect, and fixing the collapse closed it.
#
# Kept as its own test because that is a different claim from the ones above
# and could regress on its own -- `step` still re-seeds from `linspace` rather
# than warm-starting from the state's codebook, which is the other half of the
# fix that audit prescribed and which is NOT done.


class _CpuBackend:
    """The arithmetic only, so this runs with no device. Mirrors the reference
    AdamW in `tests/test_harness.py`."""

    def encode(self, values, codebook, *, group):
        return encode(values, codebook, group=group)

    def decode(self, value):
        return value.decode()

    def adamw(self, weight, momentum, variance, gradient, *, lr, beta1, beta2,
              eps, weight_decay, step):
        w, m, v, g = (value.decode() for value in (weight, momentum, variance, gradient))
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * np.square(g)
        w = w - lr * ((m / (1.0 - beta1 ** step)) / (np.sqrt(v / (1.0 - beta2 ** step)) + eps)
                      + weight_decay * w)
        return w.astype(np.float32), m.astype(np.float32), v.astype(np.float32)


def test_a_zero_gradient_moves_no_weight():
    """A step that learns nothing must change nothing.

    The weights are encoded with a good book, so any movement comes from the
    in-step refit disagreeing with it -- which is exactly what an entry-losing
    fit does.
    """
    sample = sample_of([200] + [4] * (ENTRIES - 2) + [1])
    book = fit(sample)
    assert len(np.unique(book.round(6), axis=0)) == ENTRIES, "precondition: a sound encoding book"

    weight = encode(sample, book, group=DIMS)
    zero = VQArray(weight.shape, weight.group, np.zeros_like(weight.codes),
                   np.zeros((ENTRIES, DIMS), dtype=np.float32))
    optimizer = VQAdamW(_CpuBackend(), weight, lr=1e-3, weight_decay=0.0)

    before = optimizer.weight.decode().copy()
    optimizer.step(zero)
    after = optimizer.weight.decode()

    moved = int((before != after).sum())
    assert moved == 0, (
        f"{moved} of {before.size} weight values moved on a zero gradient; "
        f"max|dw| {float(np.abs(after - before).max()):.6g}. The codebook came "
        f"back with {len(np.unique(optimizer.weight.codebook, axis=0))} distinct "
        f"rows of {ENTRIES}")
