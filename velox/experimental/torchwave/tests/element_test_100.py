# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import random

import torch
from torch import nn, Tensor

# Input layout (100 1-D tensors initialized to arange(0, size)):
#   inputs[0..49]  = s0..s49  : 50 tensors of 1000 elements
#   inputs[50..79] = m0..m29  : 30 tensors of 10000 elements
#   inputs[80..99] = l0..l19  : 20 tensors of 100000 elements
#
# Output layout (110 tensors):
#   60 outputs computed from s inputs (indices 0..49)
#   30 outputs computed from m inputs (indices 50..79)
#   20 outputs computed from l inputs (indices 80..99)
#
# Each output is a random expression of 2-4 inputs using add, sub, mul.

# Operation codes
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2

S_INDICES = list(range(0, 50))
M_INDICES = list(range(50, 80))
L_INDICES = list(range(80, 100))


def _generate_expr_specs() -> list[tuple[list[int], list[int]]]:
    """Generate deterministic random expression specs with seed 42."""
    rng = random.Random(42)
    specs: list[tuple[list[int], list[int]]] = []
    for count, indices in [(60, S_INDICES), (30, M_INDICES), (20, L_INDICES)]:
        for _ in range(count):
            n = rng.randint(2, 4)
            inp = [rng.choice(indices) for _ in range(n)]
            ops = [rng.randint(0, 2) for _ in range(n - 1)]
            specs.append((inp, ops))
    return specs


# Pre-compute so every instance shares the same graph structure.
_EXPR_SPECS: list[tuple[list[int], list[int]]] = _generate_expr_specs()


class ElementTest100(nn.Module):
    """Element-wise test with 100 inputs and 110 outputs.

    Inputs: 100 1-D tensors (s0-s49 @1000, m0-m29 @10000, l0-l19 @100000).
    Outputs: 110 tensors, each a random combination of 2-4 same-group inputs
    using add, sub, and mul operations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.expr_specs = _EXPR_SPECS

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        results: list[Tensor] = []
        for inp_indices, ops in self.expr_specs:
            val = inputs[inp_indices[0]]
            for i, op in enumerate(ops):
                next_val = inputs[inp_indices[i + 1]]
                if op == OP_ADD:
                    val = val + next_val
                elif op == OP_SUB:
                    val = val - next_val
                else:
                    val = val * next_val
            results.append(val)
        return tuple(results)
