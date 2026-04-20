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

// Compile-time test: verifies that Velox function headers compile under
// nvcc when the gpu_shadows/ directory precedes the Velox source root
// on the include path.
//
// Functions are instantiated on the host side.  Later PRs
// (GpuSimpleFunctionAdapter) will call them from __device__ kernels.

#include "velox/experimental/cudf/functions/GpuExec.h"
#include "velox/functions/prestosql/Arithmetic.h"
#include "velox/functions/prestosql/Comparisons.h"
// sparksql/Arithmetic.h deferred -- requires ToHexUtil host-only dependency
// sparksql/Comparisons.h deferred -- mixes VectorFunction factories

using namespace facebook::velox::gpu;

namespace {

// --- Presto Arithmetic ---
void verifyPlusFunction() {
  facebook::velox::functions::PlusFunction<GpuExec> fn;
  double r = 0;
  fn.call(r, 1.0, 2.0);
  (void)r;
}

void verifyMinusFunction() {
  facebook::velox::functions::MinusFunction<GpuExec> fn;
  int64_t r = 0;
  fn.call(r, int64_t{5}, int64_t{3});
  (void)r;
}

void verifyMultiplyFunction() {
  facebook::velox::functions::MultiplyFunction<GpuExec> fn;
  float r = 0;
  fn.call(r, 2.0f, 3.0f);
  (void)r;
}

void verifyDivideFunction() {
  facebook::velox::functions::DivideFunction<GpuExec> fn;
  double r = 0;
  fn.call(r, 6.0, 2.0);
  (void)r;
}

void verifyCeilFunction() {
  facebook::velox::functions::CeilFunction<GpuExec> fn;
  double r = 0;
  fn.call(r, 1.5);
  (void)r;
}

void verifyFloorFunction() {
  facebook::velox::functions::FloorFunction<GpuExec> fn;
  int64_t r = 0;
  fn.call(r, int64_t{5});
  (void)r;
}

void verifyAbsFunction() {
  facebook::velox::functions::AbsFunction<GpuExec> fn;
  double r = 0;
  fn.call(r, -5.0);
  (void)r;
}

void verifyNegateFunction() {
  facebook::velox::functions::NegateFunction<GpuExec> fn;
  int32_t r = 0;
  fn.call(r, int32_t{5});
  (void)r;
}

void verifyModulusFunction() {
  facebook::velox::functions::ModulusFunction<GpuExec> fn;
  int64_t r = 0;
  fn.call(r, int64_t{10}, int64_t{3});
  (void)r;
}

// --- Presto Comparisons ---
void verifyLtFunction() {
  facebook::velox::functions::LtFunction<GpuExec> fn;
  bool r = false;
  fn.call(r, 1.0, 2.0);
  (void)r;
}

void verifyGtFunction() {
  facebook::velox::functions::GtFunction<GpuExec> fn;
  bool r = false;
  fn.call(r, 3.0, 1.0);
  (void)r;
}

void verifyEqFunction() {
  facebook::velox::functions::EqFunction<GpuExec> fn;
  bool r = false;
  fn.call(r, 42, 42);
  (void)r;
}

void verifyNeqFunction() {
  facebook::velox::functions::NeqFunction<GpuExec> fn;
  bool r = false;
  fn.call(r, 1, 2);
  (void)r;
}

} // namespace
