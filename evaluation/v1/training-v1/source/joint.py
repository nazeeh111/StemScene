"""Permutation-equivariant refinement over real source-separator proposals.

The inference boundary accepts mixed audio, proposals, query embeddings and user
anchors only. Clean sources are used by the separate training/evaluation path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import math

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RefinerAnchor:
    query_index: int
    start: float
    end: float
    present: bool


@dataclass
class RefinementResult:
    samples: np.ndarray
    residual: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RefinerConfig:
    n_fft: int = 1024
    hop: int = 256
    pool_frequency: int = 4
    pool_time: int = 2
    hidden: int = 16
    max_queries: int = 8
    max_samples: int = 32_000 * 30
    ambiguity_cosine: float = 0.98
    version: str = "joint-mask-residual-v1"


class JointRefiner(nn.Module):
    """Shared per-query controls with symmetric competition and residual inputs.

    There are no learned query slots and no exclusive softmax. Similar prompts
    remain uncertain, rather than becoming invented physical source identities.
    """

    feature_count = 15

    def __init__(self, config: RefinerConfig | None = None) -> None:
        super().__init__()
        self.config = config or RefinerConfig()
        if (not 32 <= self.config.n_fft <= 4096 or self.config.n_fft % 2
                or not 0 < self.config.hop <= self.config.n_fft // 2):
            raise ValueError("invalid STFT size/hop")
        if (not 1 <= self.config.hidden <= 64 or not 1 <= self.config.max_queries <= 8
                or not 1 <= self.config.pool_frequency <= 16 or not 1 <= self.config.pool_time <= 16
                or not self.config.n_fft <= self.config.max_samples <= 32_000 * 30):
            raise ValueError("refiner configuration exceeds memory bounds")
        self.network = nn.Sequential(
            nn.Linear(self.feature_count, self.config.hidden), nn.SiLU(),
            nn.Linear(self.config.hidden, self.config.hidden), nn.SiLU(),
            nn.Linear(self.config.hidden, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def _spectrum(self, x: Tensor) -> Tensor:
        return torch.stft(x, self.config.n_fft, self.config.hop,
                          window=torch.hann_window(self.config.n_fft, device=x.device),
                          center=True, pad_mode="constant", return_complex=True)

    def _waveform(self, z: Tensor, length: int) -> Tensor:
        return torch.istft(z, self.config.n_fft, self.config.hop,
                           window=torch.hann_window(self.config.n_fft, device=z.device),
                           center=True, length=length)

    def _anchors(self, mixture_z: Tensor, k: int, samples: int, sample_rate: int,
                 anchors: Sequence[RefinerAnchor]) -> tuple[Tensor, Tensor, Tensor]:
        if len(anchors) > 128:
            raise ValueError("at most 128 anchors are supported")
        f, t = mixture_z.shape
        signed = mixture_z.real.new_zeros(k, t)
        affinity = mixture_z.real.new_zeros(k, f, t)
        envelope = mixture_z.real.new_ones(k, samples)
        for index, anchor in enumerate(anchors):
            if (not isinstance(anchor.query_index, int) or isinstance(anchor.query_index, bool)
                    or not isinstance(anchor.present, bool) or not 0 <= anchor.query_index < k
                    or not math.isfinite(anchor.start) or not math.isfinite(anchor.end)):
                raise ValueError("anchor index or time is invalid")
            if not 0 <= anchor.start < anchor.end <= samples / sample_rate:
                raise ValueError("anchor must lie inside the mixed recording")
            for previous in anchors[:index]:
                if (previous.query_index == anchor.query_index and previous.present != anchor.present
                        and max(previous.start, anchor.start) < min(previous.end, anchor.end)):
                    raise ValueError("overlapping present and absent anchors conflict")
            first, last = int(round(anchor.start * sample_rate)), int(round(anchor.end * sample_rate))
            lo = max(0, int(math.floor(first / self.config.hop)))
            hi = min(t, max(lo + 1, int(math.ceil(last / self.config.hop))))
            qi = anchor.query_index
            signed[qi, lo:hi] = 1.0 if anchor.present else -1.0
            if anchor.present:
                template = mixture_z[:, lo:hi].abs().mean(dim=-1, keepdim=True)
                template = template / template.mean().clamp_min(1e-7)
                affinity[qi] += (template / (1.0 + template)).expand(f, t)
            else:
                local = envelope.new_zeros(last - first)
                fade = min(int(round(0.010 * sample_rate)), len(local) // 2)
                if fade:
                    local[:fade] = torch.linspace(1, 0, fade, device=local.device)
                    local[-fade:] = torch.linspace(0, 1, fade, device=local.device)
                envelope[qi, first:last] *= local
        return signed, affinity.clamp(max=1), envelope

    def forward(self, mixture: Tensor, proposals: Tensor, embeddings: Tensor,
                sample_rate: int = 32000,
                anchors: Sequence[RefinerAnchor] = (), *,
                global_profile: Tensor | None = None,
                global_anchor_affinity: Tensor | None = None,
                global_mixture_magnitude_mean: Tensor | None = None,
                apply_anchor_envelopes: bool = True) -> Tensor:
        """Return ``(queries, samples)`` estimates. References are not an argument."""
        if mixture.ndim != 1 or proposals.ndim != 2 or proposals.shape[1] != len(mixture):
            raise ValueError("expected mixture[N] and proposals[K,N]")
        k, n = proposals.shape
        if not 1 <= k <= self.config.max_queries or not self.config.n_fft <= n <= self.config.max_samples:
            raise ValueError("query count or sample count exceeds refiner bounds")
        if embeddings.shape != (k, 512) or not 8000 <= sample_rate <= 96000:
            raise ValueError("expected embeddings[K,512] and supported sample rate")
        if not all(torch.isfinite(a).all() for a in (mixture, proposals, embeddings)):
            raise ValueError("inference inputs must be finite")
        xz, pz = self._spectrum(mixture), self._spectrum(proposals)
        xa, pa = xz.abs().clamp_min(1e-7), pz.abs()
        ratio = (pa / xa).clamp(max=8)
        phase = ((pz * xz.conj()).real / (pa * xa).clamp_min(1e-7)).clamp(-1, 1)
        peer_mean = ratio.mean(dim=0, keepdim=True).expand_as(ratio)
        peer_sum = ratio.sum(dim=0, keepdim=True).expand_as(ratio)
        peer_max = ratio.max(dim=0, keepdim=True).values.expand_as(ratio)
        unit = F.normalize(embeddings, dim=-1)
        similarities = (unit @ unit.T).clamp(-1, 1)
        if k > 1:
            other = (similarities + 1) / 2 * (1 - torch.eye(k, device=mixture.device))
            similar_peer = torch.einsum("ij,jft->ift", other, ratio) / other.sum(dim=-1).clamp_min(1e-6)[:, None, None]
            max_similarity = (similarities - torch.eye(k, device=mixture.device) * 3).max(dim=1).values
        else:
            similar_peer, max_similarity = torch.zeros_like(ratio), ratio.new_zeros(k)
        # Offline query profiles span backend chunk boundaries. They identify a
        # requested class, not an otherwise indistinguishable physical source.
        if global_profile is None:
            profile = (pa.mean(dim=-1, keepdim=True) / xa.mean(dim=-1, keepdim=True)).clamp(max=8).expand_as(ratio)
        else:
            if global_profile.shape != (k, xa.shape[0]) or not torch.isfinite(global_profile).all():
                raise ValueError("global profile must have shape [queries,frequency_bins]")
            profile = global_profile.clamp(min=0, max=8)[:, :, None].expand_as(ratio)
        activity = (pa.mean(dim=-2, keepdim=True) / xa.mean(dim=-2, keepdim=True)).clamp(max=8).expand_as(ratio)
        mixture_scale = xa.mean() if global_mixture_magnitude_mean is None else global_mixture_magnitude_mean
        log_mix = torch.log1p(xa / mixture_scale.clamp_min(1e-7)).expand_as(ratio)
        time_delta = F.pad(torch.diff(ratio, dim=-1), (1, 0)).clamp(-4, 4)
        freq = torch.linspace(0, 1, xa.shape[0], device=mixture.device)[None, :, None].expand_as(ratio)
        signed, affinity, envelope = self._anchors(xz, k, n, sample_rate, anchors)
        if global_anchor_affinity is not None:
            if global_anchor_affinity.shape != (k, xa.shape[0]) or not torch.isfinite(global_anchor_affinity).all():
                raise ValueError("global anchor affinity has invalid dimensions or values")
            affinity = global_anchor_affinity.clamp(min=0, max=1)[:, :, None].expand_as(ratio)
        features = torch.stack((torch.log1p(ratio), phase, torch.log1p(peer_mean),
                                torch.log1p(peer_sum), torch.log1p(peer_max),
                                torch.log1p(similar_peer), torch.log1p(profile),
                                torch.log1p(activity), log_mix, time_delta, freq,
                                signed[:, None, :].expand_as(ratio), affinity,
                                max_similarity[:, None, None].expand_as(ratio),
                                ratio.new_full(ratio.shape, math.log1p(k))), dim=1)
        pooled = F.avg_pool2d(features, (self.config.pool_frequency, self.config.pool_time), ceil_mode=True)
        prediction = self.network(pooled.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        correction = F.interpolate(prediction, size=xa.shape, mode="bilinear", align_corners=False)
        gain = 1.0 + 0.75 * torch.tanh(correction[:, 0])
        coefficient = torch.complex(torch.tanh(correction[:, 1]), torch.tanh(correction[:, 2]))
        unresolved = xz - pz.sum(dim=0)
        corrected = gain * pz + 0.25 / math.sqrt(k) * coefficient * unresolved
        waveform = self._waveform(corrected, n)
        return waveform * envelope if apply_anchor_envelopes else waveform

    @torch.inference_mode()
    def refine(self, mixture: np.ndarray, sample_rate: int, proposals: np.ndarray,
               query_embeddings: np.ndarray,
               anchors: Sequence[RefinerAnchor] = ()) -> RefinementResult:
        x, p = np.asarray(mixture), np.asarray(proposals)
        if x.ndim != 2 or x.shape[1] != 1 or p.ndim != 3 or p.shape[1:] != x.shape:
            raise ValueError("supports mono mixture[N,1], proposals[K,N,1]")
        device = next(self.parameters()).device
        output = self(torch.as_tensor(x[:, 0], dtype=torch.float32, device=device),
                      torch.as_tensor(p[:, :, 0], dtype=torch.float32, device=device),
                      torch.as_tensor(query_embeddings, dtype=torch.float32, device=device),
                      sample_rate, anchors).cpu().numpy().astype(np.float64)[:, :, None]
        residual = x.astype(np.float64) - output.sum(axis=0)
        unit = np.asarray(query_embeddings, dtype=np.float64)
        unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
        similarity = unit @ unit.T
        ambiguous = [[i, j] for i in range(len(p)) for j in range(i + 1, len(p))
                     if similarity[i, j] >= self.config.ambiguity_cosine]
        return RefinementResult(output, residual, {
            "method": self.config.version, "ambiguous_query_pairs": ambiguous,
            "uncertainty": "Similarity indicates prompt overlap, not isolation or physical identity.",
            "context": "offline full-clip query profiles; not causal streaming",
            "anchors": [asdict(a) for a in anchors],
            "stft": {"n_fft": self.config.n_fft, "hop": self.config.hop, "center": True,
                     "window": "periodic Hann", "padding": "constant", "delay_correction_samples": 0},
        })

    @torch.inference_mode()
    def _global_context(self, mixture: np.ndarray, proposals: np.ndarray, sample_rate: int,
                        anchors: Sequence[RefinerAnchor], frame_batch: int = 256) -> tuple[Tensor, Tensor, Tensor]:
        """Accumulate exact centered-STFT statistics without a full-clip STFT."""
        n, k = len(mixture), len(proposals)
        n_fft, hop = self.config.n_fft, self.config.hop
        bins, frames = n_fft // 2 + 1, n // hop + 1
        sum_mixture = np.zeros(bins, dtype=np.float64)
        sum_proposals = np.zeros((k, bins), dtype=np.float64)
        positive = [anchor for anchor in anchors if anchor.present]
        anchor_sums = np.zeros((len(positive), bins), dtype=np.float64)
        anchor_counts = np.zeros(len(positive), dtype=np.int64)
        device = next(self.parameters()).device
        window = torch.hann_window(n_fft, device=device)
        for first in range(0, frames, frame_batch):
            count = min(frame_batch, frames - first)
            start, end = first * hop - n_fft // 2, (first + count - 1) * hop + n_fft // 2
            lo, hi = max(0, start), min(n, end)
            local_x = torch.as_tensor(mixture[lo:hi, 0], dtype=torch.float32, device=device)[None]
            local_p = torch.as_tensor(proposals[:, lo:hi, 0], dtype=torch.float32, device=device)
            block = torch.cat((local_x, local_p), dim=0)
            if not torch.isfinite(block).all():
                raise ValueError("inference inputs must be finite")
            block = F.pad(block, (max(0, -start), max(0, end - n)))
            magnitudes = torch.stft(block, n_fft, hop, window=window, center=False,
                                    return_complex=True).abs()
            # Small float32 block sums then float64 accumulation also work on MPS.
            mx = magnitudes[0].clamp_min(1e-7)
            sum_mixture += mx.sum(dim=-1).cpu().numpy().astype(np.float64)
            sum_proposals += magnitudes[1:].sum(dim=-1).cpu().numpy().astype(np.float64)
            for index, anchor in enumerate(positive):
                anchor_lo = max(0, int(math.floor(round(anchor.start * sample_rate) / hop)))
                anchor_hi = min(frames, max(anchor_lo + 1, int(math.ceil(round(anchor.end * sample_rate) / hop))))
                a, b = max(first, anchor_lo), min(first + count, anchor_hi)
                if a < b:
                    anchor_sums[index] += magnitudes[0, :, a - first:b - first].sum(dim=-1).cpu().numpy().astype(np.float64)
                    anchor_counts[index] += b - a
        profile = np.clip(sum_proposals / np.maximum(sum_mixture[None], 1e-7), 0, 8)
        affinity = np.zeros((k, bins), dtype=np.float64)
        for index, anchor in enumerate(positive):
            template = anchor_sums[index] / max(1, int(anchor_counts[index]))
            template /= max(1e-7, float(template.mean()))
            affinity[anchor.query_index] += template / (1 + template)
        affinity = np.clip(affinity, 0, 1)
        mixture_mean = float(sum_mixture.sum() / (bins * frames))
        return (torch.as_tensor(profile, dtype=torch.float32, device=device),
                torch.as_tensor(affinity, dtype=torch.float32, device=device),
                torch.tensor(mixture_mean, dtype=torch.float32, device=device))

    @torch.inference_mode()
    def refine_long(self, mixture: np.ndarray, sample_rate: int, proposals: np.ndarray,
                    query_embeddings: np.ndarray, anchors: Sequence[RefinerAnchor] = (), *,
                    max_seconds: float = 600, window_seconds: float = 20,
                    overlap_seconds: float = 2, max_array_bytes: int = 4_000_000_000) -> RefinementResult:
        """Bounded offline long-clip refinement with persistent query-class context.

        This preserves the short-clip path exactly. Long clips use bounded STFT
        passes for global class/anchor profiles and overlapping refinement windows.
        Correlations expose ambiguous associations but never reorder user queries.
        """
        x, p = np.asarray(mixture), np.asarray(proposals)
        e = np.asarray(query_embeddings)
        if x.ndim != 2 or x.shape[1] != 1 or p.ndim != 3 or p.shape[1:] != x.shape:
            raise ValueError("supports mono mixture[N,1], proposals[K,N,1]")
        k, n = len(p), len(x)
        if not 1 <= k <= self.config.max_queries or e.shape != (k, 512) or not np.isfinite(e).all():
            raise ValueError("invalid query count or embeddings")
        if not 8000 <= sample_rate <= 96000 or not 0 < max_seconds <= 600 or n > max_seconds * sample_rate:
            raise ValueError("long recording exceeds sample-rate or duration bound")
        if (not math.isfinite(window_seconds) or not math.isfinite(overlap_seconds)
                or window_seconds <= 0 or overlap_seconds <= 0
                or not 1 <= max_array_bytes <= 4_000_000_000):
            raise ValueError("invalid long-refinement resource parameters")
        # Account conservatively for caller inputs, output, weights, residual,
        # float64 input conversion and sum temporaries. Per-window model
        # workspace is additional. Reject before allocating output.
        estimated_bytes = x.nbytes + p.nbytes + (k * n + 4 * n) * 8 + e.nbytes
        if estimated_bytes > max_array_bytes:
            raise ValueError("long recording exceeds total audio-array memory bound")
        if n <= self.config.max_samples:
            return self.refine(x, sample_rate, p, e, anchors)
        if len(anchors) > 128:
            raise ValueError("at most 128 anchors are supported")
        for index, anchor in enumerate(anchors):
            if (not isinstance(anchor.query_index, int) or isinstance(anchor.query_index, bool)
                    or not isinstance(anchor.present, bool) or not 0 <= anchor.query_index < k
                    or not math.isfinite(anchor.start) or not math.isfinite(anchor.end)
                    or not 0 <= anchor.start < anchor.end <= n / sample_rate):
                raise ValueError("anchor must identify a valid interval in the mixed recording")
            for previous in anchors[:index]:
                if (previous.query_index == anchor.query_index and previous.present != anchor.present
                        and max(previous.start, anchor.start) < min(previous.end, anchor.end)):
                    raise ValueError("overlapping present and absent anchors conflict")
        window = min(self.config.max_samples, int(round(window_seconds * sample_rate)))
        overlap = int(round(overlap_seconds * sample_rate))
        if window < self.config.n_fft or not 1 <= overlap <= window // 2:
            raise ValueError("window must fit STFT and overlap must be at most half the window")
        # Align start positions to the analysis hop to avoid avoidable phase-grid
        # shifts. The final window is padded to at least one analysis window.
        step = max(self.config.hop, ((window - overlap) // self.config.hop) * self.config.hop)
        starts = list(range(0, n, step))
        profile, affinity, mixture_mean = self._global_context(x, p, sample_rate, anchors)
        device = next(self.parameters()).device
        embedding_tensor = torch.as_tensor(e, dtype=torch.float32, device=device)
        output = np.zeros((k, n), dtype=np.float64)
        weights = np.zeros(n, dtype=np.float64)
        associations = []
        previous_end = 0
        for window_index, start in enumerate(starts):
            end = min(n, start + window)
            local_anchors = []
            for anchor in anchors:
                a, b = max(start, int(round(anchor.start * sample_rate))), min(end, int(round(anchor.end * sample_rate)))
                if a < b:
                    local_anchors.append(RefinerAnchor(anchor.query_index, (a - start) / sample_rate,
                                                       (b - start) / sample_rate, anchor.present))
            local_x = torch.as_tensor(x[start:end, 0], dtype=torch.float32, device=device)
            local_p = torch.as_tensor(p[:, start:end, 0], dtype=torch.float32, device=device)
            padding = max(0, self.config.n_fft - len(local_x))
            if padding:
                local_x, local_p = F.pad(local_x, (0, padding)), F.pad(local_p, (0, padding))
            y = self(local_x, local_p,
                     embedding_tensor, sample_rate, local_anchors, global_profile=profile,
                     global_anchor_affinity=affinity, global_mixture_magnitude_mean=mixture_mean,
                     apply_anchor_envelopes=False).cpu().numpy().astype(np.float64)[:, :end - start]
            shared = max(0, previous_end - start)
            if shared:
                observed = weights[start:start + shared] > 1e-12
                previous = output[:, start:start + shared][:, observed] / weights[None, start:start + shared][:, observed]
                current = y[:, :shared][:, observed]
                norms = np.linalg.norm(previous, axis=1)[:, None] * np.linalg.norm(current, axis=1)[None]
                correlation = (previous @ current.T) / np.maximum(norms, 1e-12)
                same = np.diag(correlation)
                ambiguous_pairs = [[i, j] for i in range(k) for j in range(i + 1, k)
                                   if max(correlation[i, j], correlation[j, i]) > 0.6
                                   and max(correlation[i, j] - same[i], correlation[j, i] - same[j]) >= -0.05]
                associations.append({"start_sample": start, "overlap_samples": shared,
                                     "observed_overlap_samples": int(observed.sum()),
                                     "same_query_zero_lag_correlation": same.tolist(),
                                     "cross_query_correlation": correlation.tolist(),
                                     "ambiguous_pairs": ambiguous_pairs,
                                     "assignment": "persistent query index; no waveform-based reassignment"})
            weight = np.ones(end - start, dtype=np.float64)
            if start > 0:
                count = min(overlap, len(weight))
                weight[:count] = np.linspace(0, 1, count)
            if end < n:
                count = min(overlap, len(weight))
                weight[-count:] = np.minimum(weight[-count:], np.linspace(1, 0, count))
            output[:, start:end] += y * weight[None]
            weights[start:end] += weight
            previous_end = end
        if not np.all(weights > 0):
            raise RuntimeError("long-refinement overlap did not cover all samples")
        output /= weights[None]
        # Apply a single global absent-anchor envelope after overlap addition.
        # No window edge can restart a fade or leak an explicitly absent span.
        for anchor in anchors:
            if not anchor.present:
                a, b = int(round(anchor.start * sample_rate)), int(round(anchor.end * sample_rate))
                fade = min(int(round(0.010 * sample_rate)), (b - a) // 2)
                row = output[anchor.query_index]
                row[a + fade:b - fade] = 0
                if fade:
                    row[a:a + fade] *= np.linspace(1, 0, fade)
                    row[b - fade:b] *= np.linspace(0, 1, fade)
        samples = output[:, :, None]
        residual = x.astype(np.float64, copy=False) - samples.sum(axis=0)
        unit = e.astype(np.float64) / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        similarity = unit @ unit.T
        ambiguous = [[i, j] for i in range(k) for j in range(i + 1, k)
                     if similarity[i, j] >= self.config.ambiguity_cosine]
        return RefinementResult(samples, residual, {
            "method": self.config.version, "mode": "bounded offline overlapping windows",
            "context": "full-recording query-class profiles accumulated in bounded STFT passes",
            "sample_rate": sample_rate, "samples": n, "window_samples": window,
            "overlap_samples": overlap, "window_count": len(starts), "step_samples": step,
            "estimated_audio_array_bytes": estimated_bytes, "max_array_bytes": max_array_bytes,
            "anchors": [asdict(a) for a in anchors], "ambiguous_query_pairs": ambiguous,
            "chunk_association": associations,
            "identity_limit": "Stable requested classes, not proven physical source identity or separation accuracy.",
            "real_long_recording_quality": "requires measured real-proposal evaluation",
        })

    def save(self, path: str | Path, *, provenance: dict[str, Any]) -> None:
        target = Path(path)
        if target.exists():
            raise FileExistsError(target)
        torch.save({"config": asdict(self.config), "state_dict": self.state_dict(),
                    "provenance": provenance}, target)

    @classmethod
    def load(cls, path: str | Path) -> "JointRefiner":
        if Path(path).stat().st_size > 1_000_000:
            raise ValueError("refiner artifact exceeds size bound")
        document = torch.load(path, map_location="cpu", weights_only=True)
        result = cls(RefinerConfig(**document["config"]))
        result.load_state_dict(document["state_dict"], strict=True)
        result.eval()
        return result


def consistency_projection(mixture: np.ndarray, proposals: np.ndarray) -> np.ndarray:
    """Unweighted Euclidean mixture-consistency projection, a known baseline.

    This baseline assigns all mixture energy to the requested estimates, so its
    explicit residual is numerically zero. That limitation must be reported for
    incomplete query sets; it is not a novel algorithm or an accuracy guarantee.
    """
    x, p = np.asarray(mixture, dtype=np.float64), np.asarray(proposals, dtype=np.float64)
    if x.ndim != 2 or p.ndim != 3 or p.shape[1:] != x.shape:
        raise ValueError("expected mixture[N,C] and proposals[K,N,C]")
    if not 1 <= len(p) <= 8 or not np.isfinite(x).all() or not np.isfinite(p).all():
        raise ValueError("invalid query count or nonfinite projection input")
    return p + (x - p.sum(axis=0))[None] / len(p)
