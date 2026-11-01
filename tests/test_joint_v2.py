"""Structural checks only; synthetic signals are not quality evidence."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from stemscene.joint import RefinerAnchor, RefinerConfig
from stemscene.joint_v2 import PeerRepairRefiner


def data(k=3, n=2048):
    generator = torch.Generator().manual_seed(41)
    p = torch.randn(k, n, generator=generator)*0.04
    x = p.sum(dim=0)
    e = torch.randn(k, 512, generator=generator)
    return x, p, e


def model(max_samples=960000):
    return PeerRepairRefiner(RefinerConfig(n_fft=64, hop=16, pool_frequency=2, pool_time=2,
                                          max_samples=max_samples, version=PeerRepairRefiner.architecture))


def test_identity_and_real_gradient():
    m = model(); x, p, e = data()
    y = m(x, p, e)
    torch.testing.assert_close(y, p, atol=2e-7, rtol=2e-5)
    (y-p*0.8).square().mean().backward()
    assert m.network[-1].weight.grad.abs().sum() > 0


def test_no_peer_and_indistinguishable_queries_protect_identity():
    m = model()
    with torch.no_grad(): m.network[-1].bias.fill_(0.8)
    x, p, e = data(1)
    assert torch.equal(m(x, p, e), p)
    x, p, e = data()
    e[:] = e[0].clone()
    assert torch.equal(m(x, p, e), p)


def test_permutation_and_nontrivial_peer_effect():
    m = model(); x, p, e = data()
    # Shared contamination creates a controlled structural collision.
    p[1] += p[0]*0.8
    with torch.no_grad(): m.network[-1].bias.copy_(torch.tensor([-0.2, 0.4, 0.3, -0.1]))
    y = m(x, p, e)
    order = torch.tensor([2, 0, 1])
    restored = m(x, p[order], e[order])[torch.argsort(order)]
    torch.testing.assert_close(y, restored, atol=3e-7, rtol=2e-5)
    assert (y-p).abs().max() > 1e-4
    assert torch.equal(m(x, p[:1], e[:1]), p[:1])


def test_anchor_and_residual_contract():
    m = model(); x, p, e = data(n=2400)
    result = m.refine(x[:, None].numpy(), 8000, p[:, :, None].numpy(), e.numpy(),
                      [RefinerAnchor(0, 0.05, 0.20, False)])
    assert np.max(np.abs(result.samples[0, 500:1500])) == 0
    np.testing.assert_allclose(result.residual+result.samples.sum(axis=0), x[:, None], atol=1e-15)
    with pytest.raises(ValueError):
        m(x, p, e, 8000, [RefinerAnchor(0, .05, .2, False), RefinerAnchor(0, .1, .25, True)])


def test_long_path_and_finite_failure():
    m = model(max_samples=1024); x, p, e = data(n=3000)
    result = m.refine_long(x[:, None].numpy(), 8000, p[:, :, None].numpy(), e.numpy(),
                           window_seconds=.128, overlap_seconds=.016)
    assert result.samples.shape == (3, 3000, 1)
    assert np.isfinite(result.samples).all()
    assert result.diagnostics["window_count"] > 1
    p[0, 0] = float("nan")
    with pytest.raises(ValueError): m(x[:1024], p[:, :1024], e, 8000)


def test_weight_roundtrip_architecture(tmp_path):
    m = model(); x, p, e = data()
    path = tmp_path/"v2.pt"
    m.save(path, provenance={"trained": False, "fixture": "structural only"})
    restored = PeerRepairRefiner.load(path)
    assert len(restored.artifact_sha256) == 64
    torch.testing.assert_close(m(x, p, e), restored(x, p, e))
    with pytest.raises(FileExistsError): m.save(path, provenance={})


def test_silent_inputs_do_not_create_energy_and_query_count_is_bounded():
    m = model()
    with torch.no_grad(): m.network[-1].bias.fill_(0.7)
    x, p, e = data(n=1024)
    assert torch.count_nonzero(m(torch.zeros_like(x), torch.zeros_like(p), e)) == 0
    x, p, e = data(k=9, n=1024)
    with pytest.raises(ValueError): m(x, p, e)


def test_loss_uses_complete_retained_audio_and_all_query_score():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from train_refiner_v2 import objective, all_request_score
    x, p, _ = data(n=40000)
    reference = p.clone(); reference[2] = 0
    output = reference.clone(); output[2] = x*0.1
    loss, terms = objective(output, reference, x)
    assert torch.isfinite(loss) and terms["retained_projection"] > 0
    assert all_request_score(output, reference, x, 8000) > 0
    assert all_request_score(reference, reference, x, 8000) == 0
