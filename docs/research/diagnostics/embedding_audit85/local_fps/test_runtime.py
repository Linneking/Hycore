"""Bounded local-CUDA validation against an independent CPU kernel reference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from fps import OriginalFPS, verified_source


def cpu_reference(xyz, npoint):
    """Float32 reference retaining per-thread first maximum and reduction ties."""
    xyz = np.asarray(xyz, dtype=np.float32)
    batch, count, _ = xyz.shape
    block = min(512, 1 << (count.bit_length() - 1))
    columns = (count + block - 1) // block
    positions = np.arange(block)[:, None] + block * np.arange(columns)[None, :]
    bounded_positions = np.minimum(positions, count - 1)
    output = np.zeros((batch, npoint), dtype=np.int32)
    for row, points in enumerate(xyz):
        distances = np.full(count, np.float32(1e10), dtype=np.float32)
        magnitude = (points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]) + points[:, 2] * points[:, 2]
        valid = magnitude > np.float32(1e-3)
        old = 0
        for index in range(1, npoint):
            difference = points - points[old]
            squared = ((difference[:, 0] * difference[:, 0] + difference[:, 1] * difference[:, 1])
                       + difference[:, 2] * difference[:, 2])
            distances[valid] = np.minimum(squared[valid], distances[valid])
            thread_values = distances[bounded_positions].copy()
            thread_valid = (positions < count) & valid[bounded_positions]
            thread_values[~thread_valid] = np.float32(-1)
            local_column = np.argmax(thread_values, axis=1)
            best_values = thread_values[np.arange(block), local_column]
            best_indices = positions[np.arange(block), local_column].astype(np.int32)
            best_indices[best_values < 0] = 0
            stride = block // 2
            while stride:
                replace = best_values[stride:2 * stride] > best_values[:stride]
                best_values[:stride] = np.maximum(best_values[:stride], best_values[stride:2 * stride])
                best_indices[:stride] = np.where(replace, best_indices[stride:2 * stride], best_indices[:stride])
                stride //= 2
            old = int(best_indices[0])
            output[row, index] = old
    return output


def cases():
    rng = np.random.default_rng(4780)
    result = []
    for count, samples in ((1, 3), (2, 5), (7, 9), (31, 20), (128, 32), (513, 64), (1024, 64)):
        result.append((f"random_n{count}_m{samples}", rng.normal(size=(2, count, 3)).astype(np.float32), samples))
    points = np.zeros((2, 1024, 3), dtype=np.float32)
    points[:, :256, 0] = 1
    points[:, 256:512, 0] = -1
    points[:, 512:768, 1] = 1
    points[:, 768:, 1] = -1
    result.append(("duplicate_symmetric_thread_ties", points, 48))
    points = rng.normal(size=(2, 128, 3)).astype(np.float32)
    points[:, :32] = 0
    points[:, 32:40] = np.float32(.001)
    points[:, 40:48] = np.float32(.02)
    result.append(("near_origin_filtered_first_zero", points, 40))
    result.append(("all_origin", np.zeros((2, 16, 3), dtype=np.float32), 20))
    result.append(("zero_output", rng.normal(size=(2, 16, 3)).astype(np.float32), 0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    args = parser.parse_args()
    verified_source()
    runtime = OriginalFPS(args.runtime_dir)
    passed = []
    for name, points, count in cases():
        expected = cpu_reference(points, count)
        tensor = torch.from_numpy(points).cuda()
        rng_before = torch.cuda.get_rng_state().clone()
        actual = runtime(tensor, count).cpu().numpy()
        np.testing.assert_array_equal(actual, expected, err_msg=name)
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng_before, rtol=0, atol=0)
        passed.append(dict(name=name, shape=list(points.shape), samples=count, exact_index_match=True))
    # Match the production 1024 -> 512 stage and a non-default Torch stream.
    points = torch.randn((8, 1024, 3), device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        first = runtime(points, 512)
    stream.synchronize()
    second = runtime(points, 512)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for _ in range(5):
        runtime(points, 512)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(50):
        runtime(points, 512)
    torch.cuda.synchronize()
    milliseconds = 1000 * (time.perf_counter() - started) / 50
    report = dict(accepted=True, kernel_source_verbatim=True, cpu_reference_exact_cases=passed,
                  nondefault_stream_exact_match=True, CUDA_RNG_unchanged=True,
                  benchmark_shape=[8, 1024, 3], benchmark_samples=512,
                  mean_FPS_milliseconds=milliseconds,
                  note="Kernel/operator gate only. Parent must still compare frozen-model outputs with a reliable existing cache.",
                  test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime=runtime.identity)
    output = args.runtime_dir / "fps_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"accepted":True, "cases":len(passed), "FPS_ms_B8_N1024_M512":milliseconds,
                      "validation":str(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
