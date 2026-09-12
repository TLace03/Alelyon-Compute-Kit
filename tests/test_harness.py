import numpy as np
import pytest

from alelyon_compute_kit import expert_bank, harness, vq


class _FakeBackend:
    def encode(self, values, codebook, *, group):
        return vq.encode(values, codebook, group=group)

    def decode(self, value):
        return value.decode()

    def matmul(self, left, right, *, transpose_a=False, transpose_b=False):
        a = left.decode()
        b = right.decode()
        return (a.T if transpose_a else a) @ (b.T if transpose_b else b)

    def adamw(self, weight, momentum, variance, gradient, *, lr, beta1, beta2,
              eps, weight_decay, step):
        w, m, v, g = (value.decode() for value in
                       (weight, momentum, variance, gradient))
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * np.square(g)
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        w = w - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * w)
        return w.astype(np.float32), m.astype(np.float32), v.astype(np.float32)


def _bank(tmp_path):
    bank = expert_bank.ExpertBank.create(
        tmp_path / "bank",
        expert_bank.BankConfig(1, 1, ((4, 3), (4, 3), (4, 3)), 8),
    )
    bank.initialize_expert(0, 0, seed=4, zero_optimizer_state=True)
    return bank


def test_forward_uses_packed_expert_and_rejects_wrong_width(tmp_path):
    bank = _bank(tmp_path)
    runner = harness.VulkanExpertHarness(bank, backend=_FakeBackend())
    inputs = np.arange(8, dtype=np.float32).reshape(2, 4)
    state = bank.read_expert_state(0, 0)
    expected = inputs @ state.tensors[0].decode()
    np.testing.assert_allclose(runner.forward(inputs), expected, rtol=0, atol=0)
    with pytest.raises(harness.HarnessError, match="width"):
        runner.forward(np.zeros((2, 5), dtype=np.float32))


def test_train_step_commits_optimizer_state_and_rejects_nonfinite(tmp_path):
    bank = _bank(tmp_path)
    runner = harness.VulkanExpertHarness(bank, backend=_FakeBackend())
    inputs = np.ones((2, 4), dtype=np.float32)
    targets = np.zeros((2, 3), dtype=np.float32)
    before = bank.read_expert_state(0, 0)
    step = runner.train_step(inputs, targets, max_grad_norm=0.01)
    after = bank.read_expert_state(0, 0)
    assert step.optimizer_step == 1
    assert step.revision == 2
    assert after.optimizer_step == 1
    assert after.parameter_update_applications == 12
    assert step.persistent_state_bytes > 0
    assert before.tensors[0].to_bytes() != after.tensors[0].to_bytes()
    with pytest.raises(harness.HarnessError, match="finite"):
        runner.forward(np.full((2, 4), np.nan, dtype=np.float32))
