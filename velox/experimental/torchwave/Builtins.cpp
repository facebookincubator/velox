/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

void registerBuiltins() {
  // Binary arithmetic.
  Registry::registerElementwise("torch.ops.aten.add.Tensor", {"alpha"});
  Registry::registerElementwise("torch.ops.aten.sub.Tensor", {"alpha"});
  Registry::registerElementwise("torch.ops.aten.mul.Tensor");
  Registry::registerElementwise("torch.ops.aten.div.Tensor");
  Registry::registerElementwise("torch.ops.aten.remainder.Tensor");
  Registry::registerElementwise("torch.ops.aten.fmod.Tensor");
  Registry::registerElementwise("torch.ops.aten.pow.Tensor_Tensor");

  // Comparison.
  Registry::registerElementwise("torch.ops.aten.eq.Tensor");
  Registry::registerElementwise("torch.ops.aten.ne.Tensor");
  Registry::registerElementwise("torch.ops.aten.lt.Tensor");
  Registry::registerElementwise("torch.ops.aten.le.Tensor");
  Registry::registerElementwise("torch.ops.aten.gt.Tensor");
  Registry::registerElementwise("torch.ops.aten.ge.Tensor");

  // Bitwise.
  Registry::registerElementwise("torch.ops.aten.bitwise_and.Tensor");
  Registry::registerElementwise("torch.ops.aten.bitwise_or.Tensor");
  Registry::registerElementwise("torch.ops.aten.bitwise_xor.Tensor");
  Registry::registerElementwise("torch.ops.aten.bitwise_not.default");

  // Logical.
  Registry::registerElementwise("torch.ops.aten.logical_and.default");
  Registry::registerElementwise("torch.ops.aten.logical_or.default");
  Registry::registerElementwise("torch.ops.aten.logical_xor.default");
  Registry::registerElementwise("torch.ops.aten.logical_not.default");

  // Unary math.
  Registry::registerElementwise("torch.ops.aten.abs.default");
  Registry::registerElementwise("torch.ops.aten.neg.default");
  Registry::registerElementwise("torch.ops.aten.ceil.default");
  Registry::registerElementwise("torch.ops.aten.floor.default");
  Registry::registerElementwise("torch.ops.aten.round.default");
  Registry::registerElementwise("torch.ops.aten.trunc.default");
  Registry::registerElementwise("torch.ops.aten.sign.default");
  Registry::registerElementwise("torch.ops.aten.sqrt.default");
  Registry::registerElementwise("torch.ops.aten.rsqrt.default");
  Registry::registerElementwise("torch.ops.aten.reciprocal.default");
  Registry::registerElementwise("torch.ops.aten.exp.default");
  Registry::registerElementwise("torch.ops.aten.log.default");
  Registry::registerElementwise("torch.ops.aten.log2.default");
  Registry::registerElementwise("torch.ops.aten.log10.default");
  Registry::registerElementwise("torch.ops.aten.log1p.default");

  // Trigonometric.
  Registry::registerElementwise("torch.ops.aten.sin.default");
  Registry::registerElementwise("torch.ops.aten.cos.default");
  Registry::registerElementwise("torch.ops.aten.tan.default");
  Registry::registerElementwise("torch.ops.aten.asin.default");
  Registry::registerElementwise("torch.ops.aten.acos.default");
  Registry::registerElementwise("torch.ops.aten.atan.default");
  Registry::registerElementwise("torch.ops.aten.atan2.default");
  Registry::registerElementwise("torch.ops.aten.sinh.default");
  Registry::registerElementwise("torch.ops.aten.cosh.default");
  Registry::registerElementwise("torch.ops.aten.tanh.default");

  // Activation functions.
  Registry::registerElementwise("torch.ops.aten.relu.default");
  Registry::registerElementwise("torch.ops.aten.sigmoid.default");
  Registry::registerElementwise("torch.ops.aten.clamp.default", {"min", "max"});

  // Min/max.
  Registry::registerElementwise("torch.ops.aten.minimum.default");
  Registry::registerElementwise("torch.ops.aten.maximum.default");
}

} // namespace torch::wave
