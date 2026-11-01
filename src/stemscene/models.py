"""Explicit, bounded public model acquisition. Runtime inference never calls this."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4


@dataclass(frozen=True)
class PublicAsset:
    file: str
    url: str
    bytes: int
    sha256: str


ASSETS: tuple[PublicAsset, ...] = (
    PublicAsset("audiosep_base_4M_steps.ckpt",
                "https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/audiosep_base_4M_steps.ckpt",
                1264844076, "f8cda01bfd0ebd141eef45d41db7a3ada23a56568465840d3cff04b8010ce82c"),
    PublicAsset("roberta/config.json", "https://huggingface.co/FacebookAI/roberta-base/resolve/main/config.json",
                481, "ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9"),
    PublicAsset("roberta/tokenizer_config.json", "https://huggingface.co/FacebookAI/roberta-base/resolve/main/tokenizer_config.json",
                25, "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce"),
    PublicAsset("roberta/vocab.json", "https://huggingface.co/FacebookAI/roberta-base/resolve/main/vocab.json",
                898823, "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655"),
    PublicAsset("roberta/merges.txt", "https://huggingface.co/FacebookAI/roberta-base/resolve/main/merges.txt",
                456318, "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    PublicAsset("licenses/AudioSep-LICENSE",
                "https://raw.githubusercontent.com/Audio-AGI/AudioSep/944583f18b84589dc965de3ad77525c945334252/LICENSE",
                1126, "f38c0f36eebc626636decda302388054a78c0e89af440d3936ba626e0db05ead"),
    PublicAsset("licenses/AudioSep-weights-README.md", "https://huggingface.co/spaces/Audio-AGI/AudioSep/raw/main/README.md",
                142, "198e39aa917d10de3923b3662d92e3b39c976a85571874c2e4748ecaa8fec087"),
)


class DownloadCancelled(RuntimeError):
    pass


class _HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: object, code: int, msg: str,
                         headers: object, newurl: str) -> Request | None:
        parsed = urlsplit(newurl)
        if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None:
            raise ValueError("Model redirect must remain public HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_models(directory: str | Path, *, timeout: float = 30.0,
                    cancelled: Callable[[], bool] | None = None,
                    progress: Callable[[str, int, int], None] | None = None) -> Path:
    """Fetch pinned public bytes into a new directory, preserving failed partials.

    No API client, authentication, token lookup, implicit model cache or input
    audio is involved. Hash/length mismatches fail closed even if upstream main
    changes. Timeout applies to each network operation; cancellation is checked
    between at most one-MiB reads.
    """
    if not 0 < timeout <= 120:
        raise ValueError("Download timeout must be between 0 and 120 seconds")
    target = Path(directory).absolute()
    if os.path.lexists(target):
        raise FileExistsError(f"Model destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial-{uuid4().hex}")
    partial.mkdir()
    opener = build_opener(_HTTPSRedirects())
    records: list[dict[str, object]] = []
    status: dict[str, object] = {
        "schema": "stemscene-model-download/1", "status": "in-progress", "assets": records,
        "code_license": "AudioSep source MIT; exact license text retained separately",
        "weight_license_evidence": "Official weight-host README declares MIT; no separately written weight license was found",
        "tokenizer_source": "FacebookAI/roberta-base tokenizer/config only; no separate RoBERTa weight download",
        "total_expected_bytes": sum(asset.bytes for asset in ASSETS),
    }
    log = partial / "download-manifest.json"

    def save_status() -> None:
        # The fixed catalog and tiny status records are bounded independently of payload size.
        temporary = log.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, log)

    save_status()
    deadline = time.monotonic() + 1800
    try:
        for asset in ASSETS:
            if cancelled is not None and cancelled():
                raise DownloadCancelled("Model download cancelled")
            record: dict[str, object] = {**asdict(asset), "status": "in-progress", "received_bytes": 0}
            records.append(record)
            save_status()
            output = partial / asset.file
            output.parent.mkdir(parents=True, exist_ok=True)
            pending = output.with_name(output.name + ".partial")
            digest, count = hashlib.sha256(), 0
            started = time.monotonic()
            request = Request(asset.url, headers={"User-Agent": "STEMSCENE-model-setup/0.1"})
            with opener.open(request, timeout=timeout) as response, pending.open("xb") as stream:
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != asset.bytes:
                    raise ValueError(f"Unexpected declared size for {asset.file}")
                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Model setup exceeded its 1800-second total limit")
                    if cancelled is not None and cancelled():
                        raise DownloadCancelled("Model download cancelled")
                    chunk = response.read(min(1024 * 1024, asset.bytes + 1 - count))
                    if not chunk:
                        break
                    count += len(chunk)
                    record["received_bytes"] = count
                    if count > asset.bytes:
                        raise ValueError(f"Download exceeds pinned byte count: {asset.file}")
                    digest.update(chunk)
                    stream.write(chunk)
                    if progress:
                        progress(asset.file, count, asset.bytes)
                stream.flush()
                os.fsync(stream.fileno())
            if count != asset.bytes or digest.hexdigest() != asset.sha256:
                raise ValueError(f"Pinned length/hash verification failed: {asset.file}")
            os.rename(pending, output)
            record.update({"status": "verified", "received_bytes": count,
                           "measured_sha256": digest.hexdigest(), "seconds": time.monotonic() - started})
            save_status()
        status["status"] = "complete"
        save_status()
        if os.path.lexists(target):
            raise FileExistsError(f"Model destination appeared during download: {target}")
        os.rename(partial, target)
        return target
    except BaseException as exc:
        status["status"] = "cancelled" if isinstance(exc, (KeyboardInterrupt, DownloadCancelled)) else "failed"
        status["error_type"] = type(exc).__name__
        # Never persist redirected/signed request URLs or environment information.
        status["recovery"] = "Verified files and partial bytes retained here. Destination was not replaced."
        save_status()
        if isinstance(exc, KeyboardInterrupt):
            raise DownloadCancelled(f"Cancelled; partial model files retained at {partial}") from exc
        if isinstance(exc, (ValueError, FileExistsError, DownloadCancelled)):
            raise type(exc)(f"{exc}; partial model files retained at {partial}") from exc
        raise RuntimeError(f"Model download failed ({type(exc).__name__}); partial files retained at {partial}") from exc
