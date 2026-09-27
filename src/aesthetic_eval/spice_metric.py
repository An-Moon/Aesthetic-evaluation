import json
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple


def compute_spice_strict(
    preds: List[str],
    refs: List[str],
    java_path: str,
    cache_dir: str,
    timeout_seconds: int,
    chunk_size: int,
    cache_enabled: bool,
    threads: int,
    java_heap_gb: int,
    checkpoint_enabled: bool,
    checkpoint_dir: str,
) -> Tuple[float, List[float]]:
    import pycocoevalcap.spice

    spice_dir = Path(pycocoevalcap.spice.__file__).resolve().parent
    spice_jar = spice_dir / "spice-1.0.jar"
    required = [spice_jar, spice_dir / "lib" / "stanford-corenlp-3.6.0.jar", spice_dir / "lib" / "stanford-corenlp-3.6.0-models.jar", Path(java_path)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing strict SPICE resources: {missing}")
    if len(preds) != len(refs):
        raise ValueError(f"SPICE input mismatch: {len(preds)} != {len(refs)}")
    if timeout_seconds <= 0:
        raise ValueError(f"SPICE timeout_seconds must be positive, got {timeout_seconds}")
    if chunk_size <= 0:
        raise ValueError(f"SPICE chunk_size must be positive, got {chunk_size}")
    if threads <= 0:
        raise ValueError(f"SPICE threads must be positive, got {threads}")
    if java_heap_gb <= 0:
        raise ValueError(f"SPICE java_heap_gb must be positive, got {java_heap_gb}")
    if cache_enabled:
        if not cache_dir:
            raise ValueError("SPICE cache_enabled requires a non-empty cache_dir")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_enabled else None
    if checkpoint_enabled:
        if not checkpoint_dir:
            raise ValueError("SPICE checkpoint_enabled requires a non-empty checkpoint_dir")
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    resource_identity = {
        str(path.name): file_sha256(path)
        for path in required
    }
    by_id = {}
    total = len(preds)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        input_data = [
            {"image_id": i, "test": preds[i], "refs": [refs[i]]}
            for i in range(start, end)
        ]
        input_json = json.dumps(input_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        checkpoint_identity = {
            "runner": "aesthetic_eval.spice_metric_strict_v2_chunk_checkpoint",
            "input_sha256": hashlib.sha256(input_json.encode("utf-8")).hexdigest(),
            "resources": resource_identity,
            "start": start,
            "end": end,
        }
        checkpoint_key = hashlib.sha256(
            json.dumps(checkpoint_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        checkpoint_path = (
            checkpoint_root / f"chunk_{start:06d}_{end:06d}_{checkpoint_key[:16]}.json"
            if checkpoint_root is not None else None
        )
        print(f"[SPICE] chunk {start}:{end}/{total}", flush=True)
        if checkpoint_path is not None and checkpoint_path.is_file():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if saved.get("identity") != checkpoint_identity:
                raise RuntimeError(f"SPICE checkpoint identity mismatch: {checkpoint_path}")
            saved_results = saved.get("results")
            if not isinstance(saved_results, list):
                raise RuntimeError(f"SPICE checkpoint results are invalid: {checkpoint_path}")
            chunk_by_id = {int(item["image_id"]): float(item["score"]) for item in saved_results}
            expected_ids = set(range(start, end))
            if set(chunk_by_id) != expected_ids:
                raise RuntimeError(
                    f"SPICE checkpoint coverage mismatch for {start}:{end}: "
                    f"expected={len(expected_ids)} actual={len(chunk_by_id)}"
                )
            print(f"[SPICE] checkpoint hit {checkpoint_path}", flush=True)
            by_id.update(chunk_by_id)
            continue
        with tempfile.TemporaryDirectory(prefix="aesthetic_spice_") as temp_dir:
            input_path, output_path = Path(temp_dir) / "input.json", Path(temp_dir) / "output.json"
            input_path.write_text(input_json, encoding="utf-8")
            command = [
                str(Path(java_path).resolve()), f"-Xmx{java_heap_gb}G",
                "-jar", str(spice_jar), str(input_path), "-threads", str(threads),
            ]
            if cache_enabled:
                command.extend(["-cache", str(Path(cache_dir).resolve())])
            command.extend(["-out", str(output_path), "-subset", "-silent"])
            subprocess.run(command, cwd=str(spice_dir), check=True, timeout=timeout_seconds)
            results = json.loads(output_path.read_text(encoding="utf-8"))
        chunk_by_id = {int(item["image_id"]): float(item["scores"]["All"]["f"]) for item in results}
        expected_ids = set(range(start, end))
        if set(chunk_by_id) != expected_ids:
            raise RuntimeError(
                f"SPICE chunk coverage mismatch for {start}:{end}: "
                f"expected={len(expected_ids)} actual={len(chunk_by_id)}"
            )
        if checkpoint_path is not None:
            payload = {
                "identity": checkpoint_identity,
                "results": [
                    {"image_id": image_id, "score": chunk_by_id[image_id]}
                    for image_id in sorted(chunk_by_id)
                ],
            }
            temp_checkpoint = checkpoint_path.with_name(
                f".{checkpoint_path.name}.tmp.{os.getpid()}"
            )
            temp_checkpoint.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            temp_checkpoint.replace(checkpoint_path)
            print(f"[SPICE] checkpoint saved {checkpoint_path}", flush=True)
        by_id.update(chunk_by_id)
    scores = [by_id[i] for i in range(len(preds))]
    if len(scores) != len(preds):
        raise RuntimeError(f"SPICE coverage mismatch: {len(scores)} != {len(preds)}")
    return sum(scores) / len(scores), scores
