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

// Compile-time test: verifies that Velox function headers can be
// compiled by nvcc when the gpu_shadows/ directory is on the include
// path BEFORE the Velox source root.
//
// This .cu file is compiled by nvcc as CUDA C++.  It includes the
// real Velox Arithmetic.h header, which transitively pulls in Macros.h,
// Exceptions.h, CPortability.h, etc.  Shadow headers intercept those
// and replace them with GPU-compatible stubs.
//
// The functions themselves are instantiated on the host side only here.
// Later PRs (GpuSimpleFunctionAdapter) will add __device__ annotations
// and call them from kernels.

#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/functions/prestosql/Arithmetic.h"

namespace {

using namespace facebook::velox::gpu;

// Verify that PlusFunction can be instantiated with GpuExec.
void hostSidePlusTest() {
  facebook::velox::functions::PlusFunction<GpuExec> fn;
  double result = 0;
  fn.call(result, 1.0, 2.0);
  (void)result;
}

// Verify MinusFunction instantiation.
void hostSideMinusTest() {
  facebook::velox::functions::MinusFunction<GpuExec> fn;
  double result = 0;
  fn.call(result, 5.0, 3.0);
  (void)result;
}

// Verify MultiplyFunction instantiation.
void hostSideMultiplyTest() {
  facebook::velox::functions::MultiplyFunction<GpuExec> fn;
  double result = 0;
  fn.call(result, 2.0, 3.0);
  (void)result;
}

// Verify DivideFunction instantiation.
void hostSideDivideTest() {
  facebook::velox::functions::DivideFunction<GpuExec> fn;
  double result = 0;
  fn.call(result, 6.0, 2.0);
  (void)result;
}

// Verify CeilFunction instantiation.
void hostSideCeilTest() {
  facebook::velox::functions::CeilFunction<GpuExec> fn;
  double result = 0;
  fn.call(result, 1.5);
  (void)result;
}

// Verify FloorFunction instantiation.
void hostSideFloorTest() {
  facebook::velox::functions::FloorFunction<GpuExec> fn;
  double result = 0;
  fn.call(result, 1.5);
  (void)result;
}

// Verify AbsFunction instantiation.
void hostSideAbsTest() {
  facebook::velox::functions::AbsFunction<GpuExec> fn;
  double result = 0;
  fn.call(result, -5.0);
  (void)result;
}

// Verify NegateFunction instantiation.
void hostSideNegateTest() {
  facebook::velox::functions::NegateFunction<GpuExec> fn;
  double result = 0;
  fn.call(result, 5.0);
  (void)result;
}

} // namespace
