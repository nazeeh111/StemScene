"""Experimental coherent-peer repair, kept separate from the frozen v1 model.

Inference accepts mixed audio, real separator proposals, prompt embeddings and
mixed-recording anchors. Clean sources are training/evaluator inputs only.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .joint import JointRefiner, RefinerAnchor, RefinerConfig


class PeerRepairRefiner(JointRefiner):
    """Shared, nonexclusive corrections at coherent cross-proposal collisions.

    A peer is a requested class, not a guaranteed distinct physical source.
    Indistinguishable prompts are not forced into competing identities.
    """

    architecture = "stemscene-coherent-peer-repair-v2"
    feature_count = 19

    def __init__(self, config: RefinerConfig | None = None) -> None:
        super().__init__(config or RefinerConfig(version=self.architecture))
        self.network = nn.Sequential(
            nn.Linear(self.feature_count, self.config.hidden), nn.SiLU(),
            nn.Linear(self.config.hidden, self.config.hidden), nn.SiLU(),
            nn.Linear(self.config.hidden, 4),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _pool(self, values: Tensor) -> Tensor:
        return F.avg_pool2d(values, (self.config.pool_frequency, self.config.pool_time), ceil_mode=True)

    def forward(self, mixture: Tensor, proposals: Tensor, embeddings: Tensor,
                sample_rate: int = 32000,
                anchors: Sequence[RefinerAnchor] = (), *,
                global_profile: Tensor | None = None,
                global_anchor_affinity: Tensor | None = None,
                global_mixture_magnitude_mean: Tensor | None = None,
                apply_anchor_envelopes: bool = True) -> Tensor:
        if mixture.ndim != 1 or proposals.ndim != 2 or proposals.shape[1] != len(mixture):
            raise ValueError("expected mixture[N] and proposals[K,N]")
        k, n = proposals.shape
        if not 1 <= k <= self.config.max_queries or not self.config.n_fft <= n <= self.config.max_samples:
            raise ValueError("query count or sample count exceeds refiner bounds")
        if embeddings.shape != (k, 512) or not 8000 <= sample_rate <= 96000:
            raise ValueError("expected embeddings[K,512] and supported sample rate")
        if not all(torch.isfinite(a).all() for a in (mixture, proposals, embeddings)):
            raise ValueError("inference inputs must be finite")
        unit = F.normalize(embeddings, dim=-1)
        similarity = (unit @ unit.T).clamp(-1, 1)
        distinct = ((similarity < self.config.ambiguity_cosine)
                    & ~torch.eye(k, dtype=torch.bool, device=mixture.device))
        xz = self._spectrum(mixture)
        signed, affinity, envelope = self._anchors(xz, k, n, sample_rate, anchors)
        # A lone class or a set of indistinguishable descriptions is preserved.
        # No hidden winner is selected from an otherwise identical query set.
        if not distinct.any():
            return proposals * envelope if apply_anchor_envelopes else proposals
        pz = self._spectrum(proposals)
        xa, pa = xz.abs().clamp_min(1e-7), pz.abs()
        ratio = (pa / xa).clamp(max=8)
        phase = ((pz * xz.conj()).real / (pa * xa).clamp_min(1e-7)).clamp(-1, 1)
        common = torch.zeros_like(pz)
        collision = torch.zeros_like(pa)
        peer_phase = torch.zeros_like(pa)
        counts = distinct.sum(dim=1).clamp_min(1).to(pa.dtype)
        power = self._pool(pa.square()[:, None])[:, 0]
        for j in range(k):
            cross = (pz * pz[j].conj()).real
            local = self._pool(cross[:, None])[:, 0]
            coherence = (local / (power * power[j]).sqrt().clamp_min(1e-7)).clamp(0, 1)
            coherence = F.interpolate(coherence[:, None], size=xa.shape,
                                      mode="bilinear", align_corners=False)[:, 0]
            strength = (torch.minimum(pa, pa[j]) / pa.clamp_min(1e-7)).clamp(0, 1)
            weight = coherence * strength * distinct[:, j, None, None]
            common += pz[j] * (pa / pa[j].clamp_min(1e-7)).clamp(max=1) * weight
            collision = torch.maximum(collision, weight)
            peer_phase += coherence * distinct[:, j, None, None]
        common /= counts[:, None, None]
        peer_phase /= counts[:, None, None]
        unresolved = xz - pz.sum(dim=0)
        residual_ratio = (unresolved.abs() / xa).clamp(max=8).expand_as(ratio)
        residual_phase = ((unresolved * pz.conj()).real /
                          (unresolved.abs() * pa).clamp_min(1e-7)).clamp(-1, 1)
        common_ratio = (common.abs() / pa.clamp_min(1e-7)).clamp(max=2)
        if global_profile is None:
            profile = (pa.mean(dim=-1, keepdim=True) / xa.mean(dim=-1, keepdim=True)).clamp(max=8).expand_as(ratio)
        else:
            if global_profile.shape != (k, xa.shape[0]) or not torch.isfinite(global_profile).all():
                raise ValueError("invalid global query profile")
            profile = global_profile.clamp(0, 8)[:, :, None].expand_as(ratio)
        if global_anchor_affinity is not None:
            if global_anchor_affinity.shape != (k, xa.shape[0]) or not torch.isfinite(global_anchor_affinity).all():
                raise ValueError("invalid global anchor affinity")
            affinity = global_anchor_affinity.clamp(0, 1)[:, :, None].expand_as(ratio)
        activity = (pa.mean(dim=-2, keepdim=True) / xa.mean(dim=-2, keepdim=True)).clamp(max=8).expand_as(ratio)
        scale = xa.mean() if global_mixture_magnitude_mean is None else global_mixture_magnitude_mean
        log_mix = torch.log1p(xa / scale.clamp_min(1e-7)).expand_as(ratio)
        maximum_similarity = (similarity - torch.eye(k, device=mixture.device) * 3).max(dim=1).values
        frequency = torch.linspace(0, 1, xa.shape[0], device=mixture.device)[None, :, None].expand_as(ratio)
        features = torch.stack((
            torch.log1p(ratio), phase, torch.log1p(ratio.mean(dim=0, keepdim=True).expand_as(ratio)),
            torch.log1p(ratio.sum(dim=0, keepdim=True).expand_as(ratio)),
            collision, peer_phase, common_ratio, torch.log1p(residual_ratio), residual_phase,
            torch.log1p(profile), torch.log1p(activity), log_mix,
            F.pad(torch.diff(ratio, dim=-1), (1, 0)).clamp(-4, 4), frequency,
            signed[:, None, :].expand_as(ratio), affinity,
            maximum_similarity[:, None, None].expand_as(ratio),
            ratio.new_full(ratio.shape, math.log1p(k)),
            (ratio.sum(dim=0, keepdim=True) - 1).clamp(0, 4).expand_as(ratio)), dim=1)
        prediction = self.network(self._pool(features).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        controls = F.interpolate(prediction, size=xa.shape, mode="bilinear", align_corners=False)
        correction = (0.25 * torch.tanh(controls[:, 0]) * pz
                      - 0.5 * torch.tanh(controls[:, 1]) * common
                      + 0.25 * torch.complex(torch.tanh(controls[:, 2]),
                                              torch.tanh(controls[:, 3])) * unresolved)
        corrected = pz + collision * correction
        waveform = self._waveform(corrected, n)
        return waveform * envelope if apply_anchor_envelopes else waveform

    def save(self, path: str | Path, *, provenance: dict[str, Any]) -> None:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(destination)
        torch.save({"architecture": self.architecture, "config": asdict(self.config),
                    "state_dict": self.state_dict(), "provenance": provenance}, destination)

    @classmethod
    def load(cls, path: str | Path) -> "PeerRepairRefiner":
        path = Path(path)
        if path.stat().st_size > 1_000_000:
            raise ValueError("refiner artifact exceeds size bound")
        document = torch.load(path, map_location="cpu", weights_only=True)
        if document.get("architecture") != cls.architecture:
            raise ValueError("not a coherent-peer-repair-v2 artifact")
        result = cls(RefinerConfig(**document["config"]))
        result.load_state_dict(document["state_dict"], strict=True)
        result.provenance = document.get("provenance", {})
        with path.open("rb") as stream:
            result.artifact_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        result.eval()
        return result
