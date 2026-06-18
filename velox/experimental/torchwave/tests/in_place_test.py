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

from torch import nn, Tensor


class InPlaceTestPreproc(nn.Module):
    """Exercises in-place mutation through views and clones.

    a, b, y, yy are 1-D tensors of equal length.

    Chain on a: va aliases a (a view), so every va.add_ also mutates a. The
    clones snapshot a at distinct points; c aliases a (it is the result of an
    in-place add_). Correct output requires honoring the imperative order of the
    in-place ops and the aliasing between a, va, and c.

      va = a.view(-1)
      a.add_(1)          # a += 1
      a2 = a.clone()     # snapshot (dead: not returned)
      c = va.add_(2)     # a += 2; c aliases a
      d = a.clone()      # snapshot of a after +3
      va.add_(b)         # a += b
      e = a.clone()      # snapshot of a after +3+b

    View-return of a mutated tensor whose in-place op output is unreferenced:

      x = y.view(-1)
      y.add_(1)          # output unreferenced; x must still see y+1

    Multiply-referenced output of an in-place op (xx aliases yy):

      xx = yy.add_(5)
      yy2 = xx + 2
      yy3 = xx + 10

    In-place output returned, then self mutated again (strict alias test: a
    copy of the result would miss the later +100):

      zz = ww.add_(5)
      ww.add_(100)       # zz must reflect this; zz aliases ww

    With a0/b/y0/yy0/ww0 the inputs:
      a == c == e == a0 + 3 + b
      d == a0 + 3
      x == y0 + 1
      xx == yy0 + 5
      yy2 == yy0 + 7
      yy3 == yy0 + 15
      zz == ww0 + 105
    """

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        y: Tensor,
        yy: Tensor,
        ww: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        va = a.view(-1)
        a.add_(1)
        a2 = a.clone()  # noqa: F841 (intentionally dead clone)
        c = va.add_(2)
        d = a.clone()
        va.add_(b)
        e = a.clone()

        # View returned; the in-place op's own output is unreferenced, but the
        # mutation must still be visible through the returned view x.
        x = y.view(-1)
        y.add_(1)

        # Multiply-referenced output of an in-place op: xx aliases yy and feeds
        # two consumers and is itself returned.
        xx = yy.add_(5)
        yy2 = xx + 2
        yy3 = xx + 10

        # Strict alias test: zz is the in-place result and is returned, but ww
        # is mutated again afterward. zz must reflect the later mutation (it
        # aliases ww), which a freshly-copied output would not.
        zz = ww.add_(5)
        ww.add_(100)

        return a, b, c, d, e, x, xx, yy2, yy3, zz
