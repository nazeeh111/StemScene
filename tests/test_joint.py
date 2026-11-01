"""Structural tests only. Synthetic signals do not establish audio quality."""
import numpy as np
import pytest
import torch

from stemscene.joint import JointRefiner, RefinerAnchor, RefinerConfig, consistency_projection


@pytest.fixture
def example():
    torch.manual_seed(10)
    return torch.randn(4096) * 0.1, torch.randn(3, 4096) * 0.03, torch.randn(3, 512)


def test_query_permutation_equivariance_after_nontrivial_parameter_change(example):
    x, p, e = example
    model = JointRefiner()
    with torch.no_grad():
        model.network[-1].weight.normal_(0, 0.03)
        model.network[-1].bias.normal_(0, 0.03)
        expected = model(x, p, e)
        permutation = torch.tensor([2, 0, 1])
        actual = model(x, p[permutation], e[permutation])
    torch.testing.assert_close(actual, expected[permutation], atol=2e-7, rtol=2e-5)


def test_initial_model_is_proposal_identity_and_differentiable(example):
    x, p, e = example
    model = JointRefiner()
    result = model(x, p, e)
    torch.testing.assert_close(result, p, atol=1e-7, rtol=2e-5)
    result.square().mean().backward()
    assert model.network[-1].weight.grad.abs().sum() > 0


def test_residual_reconstructs_original_and_ambiguity_is_exposed(example):
    x, p, e = example
    e[1] = e[0]
    result = JointRefiner().refine(x.numpy()[:, None].astype(np.float64), 32000,
                                    p.numpy()[:, :, None], e.numpy())
    recombined = result.samples.sum(axis=0) + result.residual
    np.testing.assert_allclose(recombined[:, 0], x.numpy(), atol=1e-16, rtol=0)
    assert [0, 1] in result.diagnostics["ambiguous_query_pairs"]


def test_absent_anchor_uses_explicit_span_and_present_anchor_is_mixed_context(example):
    x, p, e = example
    model = JointRefiner()
    anchors = [RefinerAnchor(0, 0.02, 0.08, False), RefinerAnchor(1, 0.01, 0.1, True)]
    actual = model(x, p, e, 32000, anchors)
    assert torch.count_nonzero(actual[0, 1000:2000]) == 0
    torch.testing.assert_close(actual[0, :600], p[0, :600], atol=1e-7, rtol=2e-5)
    torch.testing.assert_close(actual[1], p[1], atol=1e-7, rtol=2e-5)
    with pytest.raises(ValueError, match="conflict"):
        model(x, p, e, 32000, anchors + [RefinerAnchor(0, 0.03, 0.06, True)])


def test_silence_is_finite_and_output_aligned():
    model = JointRefiner()
    y = model(torch.zeros(4096), torch.zeros(2, 4096), torch.ones(2, 512))
    assert y.shape == (2, 4096)
    assert torch.isfinite(y).all() and torch.count_nonzero(y) == 0


def test_failure_bounds_and_safe_checkpoint_round_trip(example, tmp_path):
    model = JointRefiner()
    x, p, e = example
    with pytest.raises(ValueError):
        model(x, p[:, :-1], e)
    with pytest.raises(ValueError):
        model(x, p, e[:, :511])
    with pytest.raises(ValueError):
        model(torch.full_like(x, float("nan")), p, e)
    with pytest.raises(ValueError):
        JointRefiner(RefinerConfig(hidden=100_000))
    target = tmp_path / "weights.pt"
    model.save(target, provenance={"test_only": True})
    with pytest.raises(FileExistsError):
        model.save(target, provenance={})
    loaded = JointRefiner.load(target)
    torch.testing.assert_close(loaded(x, p, e), model(x, p, e))


def test_known_consistency_projection_is_order_independent(example):
    x, p, _ = (value.numpy() for value in example)
    y = consistency_projection(x[:, None], p[:, :, None])
    np.testing.assert_allclose(y.sum(axis=0)[:, 0], x, atol=1e-16, rtol=0)
    permutation = [2, 0, 1]
    np.testing.assert_allclose(consistency_projection(x[:, None], p[permutation, :, None]), y[permutation], atol=1e-16)


def test_bounded_global_context_matches_full_centered_stft(example):
    x, p, _ = example
    model = JointRefiner(RefinerConfig(n_fft=256, hop=64, max_samples=4096, hidden=4))
    profile, _, mean = model._global_context(x.numpy()[:, None], p.numpy()[:, :, None], 32000, (), frame_batch=7)
    xz, pz = model._spectrum(x), model._spectrum(p)
    expected = (pz.abs().mean(dim=-1) / xz.abs().clamp_min(1e-7).mean(dim=-1)).clamp(max=8)
    torch.testing.assert_close(profile, expected, atol=3e-7, rtol=3e-6)
    torch.testing.assert_close(mean, xz.abs().clamp_min(1e-7).mean(), atol=3e-7, rtol=3e-6)


def test_long_refinement_identity_anchor_and_query_association():
    torch.manual_seed(21)
    n, sr = 12_289, 8000
    p = np.random.default_rng(21).normal(0, 0.02, size=(3, n, 1))
    x = p.sum(axis=0)
    e = np.random.default_rng(22).normal(size=(3, 512))
    model = JointRefiner(RefinerConfig(n_fft=256, hop=64, max_samples=4096, hidden=4))
    # This anchor crosses several refinement windows. One global envelope is
    # required so no internal boundary restarts the fade or leaks the span.
    anchor = RefinerAnchor(0, 0.25, 1.25, False)
    result = model.refine_long(x, sr, p, e, [anchor], window_seconds=0.4, overlap_seconds=0.05)
    np.testing.assert_allclose(result.samples[1:], p[1:], atol=3e-8, rtol=2e-5)
    assert np.count_nonzero(result.samples[0, 2200:9800]) == 0
    np.testing.assert_allclose(result.samples.sum(axis=0) + result.residual, x, atol=1e-16, rtol=0)
    assert result.diagnostics["window_count"] > 2
    assert len(result.diagnostics["chunk_association"]) > 1
    for association in result.diagnostics["chunk_association"]:
        assert min(association["same_query_zero_lag_correlation"]) > 0.999


def test_long_refinement_permutation_and_resource_failures():
    rng = np.random.default_rng(31)
    x, p, e = rng.normal(size=(9000, 1)) * 0.1, rng.normal(size=(2, 9000, 1)) * 0.03, rng.normal(size=(2, 512))
    model = JointRefiner(RefinerConfig(n_fft=256, hop=64, max_samples=4096, hidden=4))
    with torch.no_grad():
        model.network[-1].weight.normal_(0, 0.02)
    expected = model.refine_long(x, 8000, p, e, window_seconds=0.4, overlap_seconds=0.05)
    actual = model.refine_long(x, 8000, p[::-1].copy(), e[::-1].copy(), window_seconds=0.4, overlap_seconds=0.05)
    np.testing.assert_allclose(actual.samples, expected.samples[::-1], atol=5e-8, rtol=1e-5)
    with pytest.raises(ValueError, match="memory bound"):
        model.refine_long(x, 8000, p, e, max_array_bytes=10)
    with pytest.raises(ValueError, match="duration bound"):
        model.refine_long(x, 8000, p, e, max_seconds=0.5)
    with pytest.raises(ValueError, match="overlap"):
        model.refine_long(x, 8000, p, e, window_seconds=0.4, overlap_seconds=0.3)


def test_pre_cancelled_long_refinement_stops_before_tensor_work():
    import numpy as np
    import pytest
    model = JointRefiner()
    with pytest.raises(InterruptedError,match='before computation'):
        model.refine_long(np.empty((0,1)),32000,np.empty((0,0,1)),np.empty((0,512)),cancelled=lambda:True)
