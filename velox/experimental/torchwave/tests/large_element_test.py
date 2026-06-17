# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

from torch import nn, Tensor

OP_AND = 0
OP_OR = 1
OP_ADD = 2
OP_SUB = 3
OP_MUL = 4


def _generate_expr_specs() -> list[tuple[int, list[tuple[int, int, int, bool]]]]:
    """Generate deterministic random expression specs with seed 42.

    Returns list of (start_input_idx, [(op_code, input_idx_or_-1, scalar_val,
    nest_right)]).
    When input_idx is -1, scalar_val is used as the RHS operand.
    When nest_right is True, the accumulator becomes the RHS and a new
    sub-expression is built on the LHS.
    """
    rng = random.Random(42)
    all_indices = list(range(100))
    ops_per_expr = [100, 100, 100]
    scalar_positions = set(rng.sample(range(300), 60))

    specs: list[tuple[int, list[tuple[int, int, int, bool]]]] = []
    global_op = 0
    for expr_idx in range(3):
        n_ops = ops_per_expr[expr_idx]
        start_idx = rng.choice(all_indices)
        ops: list[tuple[int, int, int, bool]] = []
        for _ in range(n_ops):
            op_code = rng.randint(0, 4)
            is_scalar = global_op in scalar_positions
            nest_right = rng.random() < 0.3
            if is_scalar:
                scalar_val = rng.randint(1, 15)
                ops.append((op_code, -1, scalar_val, False))
            else:
                inp_idx = rng.choice(all_indices)
                ops.append((op_code, inp_idx, 0, nest_right))
            global_op += 1
        specs.append((start_idx, ops))

    return specs


_EXPR_SPECS = _generate_expr_specs()


def _apply_op(op_code: int, lhs: Tensor, rhs: Tensor | int) -> Tensor:
    if op_code == OP_AND:
        return lhs & rhs
    elif op_code == OP_OR:
        return lhs | rhs
    elif op_code == OP_ADD:
        return lhs + rhs
    elif op_code == OP_SUB:
        return lhs - rhs
    else:
        return lhs * rhs


class LargeElementTest(nn.Module):
    """Element-wise test with 100 int64 inputs and 3 outputs.

    Inputs: 100 1-D tensors of 500 int64 elements each.
    Outputs: 3 tensors, each a chain of 100 operations using
    bitwise_and, bitwise_or, add, sub, and mul. 60 of the 300
    total operations use scalar operands. About 30% of tensor
    operations nest on the right side to test non-left-deep trees.
    """

    def __init__(self) -> None:
        super().__init__()
        self.expr_specs = _EXPR_SPECS

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        results: list[Tensor] = []
        for start_idx, ops in self.expr_specs:
            val = inputs[start_idx]
            for op_code, inp_idx, scalar_val, nest_right in ops:
                rhs = scalar_val if inp_idx == -1 else inputs[inp_idx]
                if nest_right:
                    # pyrefly: ignore [bad-argument-type]
                    val = _apply_op(op_code, rhs, val)
                else:
                    val = _apply_op(op_code, val, rhs)
            results.append(val)
        return tuple(results)
