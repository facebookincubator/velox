# Copyright (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Compare PyTorch EAGER (CPU) vs PyTorch NATIVERT (CPU) numerically, with a
# focus on float/double rounding differences.
#
# This deliberately does NOT involve torchwave or sigmoid. It runs a battery of
# float32 and float64 ops:
#   1. eagerly on CPU (the reference), and
#   2. through torch.export -> package_pt2 -> torch._C._nativert.PyModelRunner
#      (the pure-PyTorch native runtime) on CPU,
# and reports, per output and per dtype:
#   - bit-exact fraction (results that are byte-for-byte identical),
#   - max / mean absolute difference,
#   - max relative difference,
#   - max ULP (units-in-the-last-place) difference.
#
# It does this for two export variants:
#   (A) the exported graph as-is (pre-dispatch / training IR), and
#   (B) the same graph after run_decompositions() (core-aten IR),
# because most eager-vs-nativert rounding differences come from op
# DECOMPOSITION (e.g. softmax / layer_norm / var being rewritten into
# exp/sum/div sequences), not from the runtime itself.

import os
import tempfile
from typing import Dict, Tuple

import torch
from torch._C._nativert import PyModelRunner
from torch.export.pt2_archive._package import package_pt2

MODEL_NAME = "compare"


def _battery(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """A battery of ops prone to accumulation / rounding differences.

    x: (N, K), w: (K, M), b: (M,) all of one floating dtype.
    """
    out: Dict[str, torch.Tensor] = {}
    # GEMM-style accumulation (order/blocking sensitive).
    out["matmul"] = x @ w
    out["addmm"] = torch.addmm(b, x, w)  # linear: b + x @ w
    out["linear"] = torch.nn.functional.linear(x, w.t(), b)
    # Reductions (parallel reduction order sensitive).
    out["sum_dim"] = x.sum(dim=1)
    out["mean_dim"] = x.mean(dim=1)
    out["var_dim"] = x.var(dim=1, unbiased=True)
    out["cumsum"] = torch.cumsum(x, dim=1)
    # Fused normalizations that get decomposed.
    out["softmax"] = torch.softmax(x, dim=1)
    out["log_softmax"] = torch.log_softmax(x, dim=1)
    out["layer_norm"] = torch.nn.functional.layer_norm(x, (x.shape[1],))
    # Transcendentals / elementwise.
    out["sigmoid"] = torch.sigmoid(x)
    out["tanh"] = torch.tanh(x)
    out["gelu"] = torch.nn.functional.gelu(x)
    out["gelu_tanh"] = torch.nn.functional.gelu(x, approximate="tanh")
    out["exp"] = torch.exp(x * 0.5)
    out["log"] = torch.log(torch.abs(x) + 1.0)
    out["rsqrt"] = torch.rsqrt(torch.abs(x) + 1.0)
    out["div"] = x / (torch.abs(x) + 0.5)
    return out


class CompareModule(torch.nn.Module):
    def forward(
        self,
        x32: torch.Tensor,
        w32: torch.Tensor,
        b32: torch.Tensor,
        x64: torch.Tensor,
        w64: torch.Tensor,
        b64: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in _battery(x32, w32, b32).items():
            out[f"f32::{k}"] = v
        for k, v in _battery(x64, w64, b64).items():
            out[f"f64::{k}"] = v
        return out


def _ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Max ULP distance between two same-dtype float tensors (total-order)."""
    if a.dtype == torch.float32:
        ia = a.contiguous().view(torch.int32).to(torch.int64)
        ib = b.contiguous().view(torch.int32).to(torch.int64)
        maxv = 0x7FFFFFFF
    elif a.dtype == torch.float64:
        ia = a.contiguous().view(torch.int64)
        ib = b.contiguous().view(torch.int64)
        maxv = 0x7FFFFFFFFFFFFFFF
    else:
        return torch.zeros((), dtype=torch.int64)

    def monotone(i: torch.Tensor) -> torch.Tensor:
        # IEEE-754 stored as signed int -> a monotone (total) ordering:
        # non-negative stays, negative gets its lower bits flipped.
        neg = i < 0
        return torch.where(neg, maxv - i, i)

    return (monotone(ia) - monotone(ib)).abs()


def _compare(
    name: str, a: torch.Tensor, b: torch.Tensor
) -> Tuple[float, float, float, int, int]:
    """Return (max_abs, mean_abs, max_rel, max_ulp, n) comparing a (eager) vs b (nativert)."""
    a = a.detach()
    b = b.detach()
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    assert a.dtype == b.dtype, f"{name}: dtype {a.dtype} vs {b.dtype}"
    af = a.to(torch.float64)
    bf = b.to(torch.float64)
    diff = (af - bf).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    denom = af.abs().clamp_min(1e-30)
    max_rel = float((diff / denom).max())
    ulp = _ulp_diff(a, b)
    max_ulp = int(ulp.max())
    n = a.numel()
    return max_abs, mean_abs, max_rel, max_ulp, n


def _run_nativert(
    ep: torch.export.ExportedProgram, args: Tuple[torch.Tensor, ...]
) -> Dict[str, torch.Tensor]:
    with tempfile.TemporaryDirectory() as d:
        archive = os.path.join(d, "model.pt2")
        package_pt2(archive, exported_programs={MODEL_NAME: ep})
        runner = PyModelRunner(archive, MODEL_NAME)
        return runner.run(*args)


def _report(
    title: str,
    eager: Dict[str, torch.Tensor],
    nativert: Dict[str, torch.Tensor],
) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    header = f"{'output':<22}{'dtype':<9}{'bit-exact':<12}{'max_abs':<13}{'mean_abs':<13}{'max_rel':<13}{'max_ulp':<9}"
    print(header)
    print("-" * 100)
    keys = list(eager.keys())
    n_exact = 0
    for k in keys:
        a = eager[k]
        b = nativert[k]
        max_abs, mean_abs, max_rel, max_ulp, n = _compare(k, a, b)
        exact = bool(torch.equal(a, b))
        n_exact += int(exact)
        dtype = "f32" if a.dtype == torch.float32 else "f64"
        print(
            f"{k:<22}{dtype:<9}{('yes' if exact else 'NO'):<12}"
            f"{max_abs:<13.3e}{mean_abs:<13.3e}{max_rel:<13.3e}{max_ulp:<9}"
        )
    print("-" * 100)
    print(f"bit-exact outputs: {n_exact}/{len(keys)}")


def main() -> None:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    N, K, M = 64, 128, 96
    x32 = torch.randn(N, K, dtype=torch.float32)
    w32 = torch.randn(K, M, dtype=torch.float32)
    b32 = torch.randn(M, dtype=torch.float32)
    x64 = x32.to(torch.float64)
    w64 = w32.to(torch.float64)
    b64 = b32.to(torch.float64)
    args = (x32, w32, b32, x64, w64, b64)

    module = CompareModule().eval()

    print(f"torch version: {torch.__version__}")
    print(f"intra-op threads: {torch.get_num_threads()}")

    # Reference: eager CPU.
    with torch.no_grad():
        eager_out = module(*args)

    # Export once.
    with torch.no_grad():
        ep = torch.export.export(module, args, strict=False)
    print(f"exported graph: {len(list(ep.graph.nodes))} nodes")

    # (A) nativert on the exported graph as-is.
    nativert_a = _run_nativert(ep, args)
    _report(
        "(A) EAGER CPU vs NATIVERT CPU  -- exported graph as-is (pre-dispatch IR)",
        eager_out,
        nativert_a,
    )

    # (B) nativert after run_decompositions() (core-aten IR).
    with torch.no_grad():
        ep_decomp = ep.run_decompositions()
    print(f"\ndecomposed graph: {len(list(ep_decomp.graph.nodes))} nodes")
    nativert_b = _run_nativert(ep_decomp, args)
    _report(
        "(B) EAGER CPU vs NATIVERT CPU  -- after run_decompositions() (core-aten IR)",
        eager_out,
        nativert_b,
    )


if __name__ == "__main__":
    main()
